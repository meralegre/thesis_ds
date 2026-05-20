import pandas as pd
import numpy as np
import keras
from keras import layers, ops
from sklearn.model_selection import KFold

import os
import ast
import json
import argparse
import tempfile

# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_melodies(melodies, max_len=128, pitch_to_idx=None):
    """
    Convert list of pitch sequences into transformer-ready format.
    Melodies are padded to max_len.
    """
    # Build or reuse vocabulary
    if pitch_to_idx is None:
        all_pitches = sorted(set(p for mel in melodies for p in mel))
        pitch_to_idx = {p: i + 1 for i, p in enumerate(all_pitches)}

    idx_to_pitch = {i: p for p, i in pitch_to_idx.items()}
    # +1 since index 0 is PAD
    vocab_size = max(pitch_to_idx.values()) + 1

    PAD = 0

    xs = []
    ys = []
    masks = []
    skipped = 0

    for mel in melodies:
        # Convert to indices, skipping pitches not in vocabulary
        indexed = [pitch_to_idx[p] for p in mel if p in pitch_to_idx]

        if len(indexed) < 2:
            skipped += 1
            continue

        # Input: all tokens except the last
        # Target: all tokens except the first (shifted by 1)
        inp = indexed[:-1]
        tgt = indexed[1:]

        # Pad or truncate to max_len
        if len(inp) >= max_len:
            # Truncate: take last max_len tokens
            inp = inp[-max_len:]
            tgt = tgt[-max_len:]
            m = [True] * max_len
        else:
            # Pad from the left
            pad_len = max_len - len(inp)
            m = [False] * pad_len + [True] * len(inp)
            inp = [PAD] * pad_len + inp
            tgt = [PAD] * pad_len + tgt

        xs.append(inp)
        ys.append(tgt)
        masks.append(m)

    if skipped > 0:
        print(f"  Warning: skipped {skipped} melodies (too short or unknown pitches)")

    return (np.array(xs), np.array(ys), np.array(masks),
            vocab_size, pitch_to_idx, idx_to_pitch)


def parse_lisp_melodies(filepath, viewpoint=":CPITCH"):
    """Parse IDyOM lisp export and extract pitch sequences per melody."""
    import re
    with open(filepath, 'r') as f:
        text = f.read()

    melody_pattern = r'\("([^"]+)"\s*\n((?:\s*\((?:\(:[A-Z].*?\)\s*)+\)\s*)+)'
    melodies = {}

    for match in re.finditer(melody_pattern, text):
        name = match.group(1)
        if "Dataset" in name or "Collection" in name:
            continue
        block = match.group(2)

        pitch_pattern = rf'\({viewpoint}\s+(\d+)\)'
        pitches = [int(p) for p in re.findall(pitch_pattern, block)]

        if pitches:
            melodies[name] = pitches

    return melodies


# ============================================================
# MODEL COMPONENTS
# ============================================================
class TokenAndPositionEmbedding(layers.Layer):
    """Embeds tokens and adds learned positional encoding."""

    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=False
        )
        self.pos_emb = layers.Embedding(
            input_dim=max_len, output_dim=embed_dim
        )

    def call(self, x):
        seq_len = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=seq_len, step=1)
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(positions)
        return token_embeddings + position_embeddings


class CausalTransformerBlock(layers.Layer):
    """
    Transformer block with CAUSAL attention mask.

    The causal mask ensures position i can only attend to positions <= i,
    matching IDyOM's left-to-right prediction constraint.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
#             dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Causal mask: each position can only attend to itself and earlier positions
        attn_output = self.att(
            inputs, inputs, use_causal_mask=True
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer(
    vocab_size,
    max_len=64,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    dropout_rate=0.1,
):
    """
    Build a simple causal transformer for next-pitch prediction.
    """
    inputs = keras.Input(shape=(max_len,))

    # Embedding
    x = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)(inputs)

    # Transformer blocks with causal masking
    for _ in range(num_layers):
        x = CausalTransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # Per-position prediction head
    # Dense applied to (batch, seq_len, embed_dim) works per-position automatically
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================
# LOSS & IC COMPUTATION
# ============================================================
def masked_sparse_crossentropy(y_true, y_pred):
    """
    Sparse categorical crossentropy that ignores PAD tokens (index 0).

    This ensures padding doesn't artificially lower the loss.
    """
    # Create mask: True where target is not PAD
    mask = ops.cast(y_true != 0, dtype="float32")

    # Compute per-position loss
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Apply mask and average over non-padded positions only
    masked_loss = loss * mask
    return ops.sum(masked_loss) / (ops.sum(mask) + 1e-8)


def compute_ic(model, xs, ys, masks):
    """
    Compute Information Content (IC) in bits for each note prediction.
    """
    # Get model predictions
    probs = model.predict(xs, verbose=0)

    all_ics = []

    for i in range(len(xs)):
        for t in range(len(xs[i])):
            if masks[i][t]:
                # Probability assigned to the correct next token
                true_token = ys[i][t]
                p = probs[i][t][true_token]

                # IC in bits (p is a clip to avoid log(0))
                p = max(p, 1e-10)
                ic = -np.log2(p)
                all_ics.append(ic)

    mean_ic = np.mean(all_ics)
    return mean_ic, all_ics


def compute_per_melody_ic(model, xs, ys, masks, melody_indices):
    """
    Compute mean IC per melody (useful for per-piece comparison with IDyOM).
    """
    probs = model.predict(xs, verbose=0)
    melody_ics = {}

    for i in range(len(xs)):
        mel_id = melody_indices[i]
        ics = []
        for t in range(len(xs[i])):
            if masks[i][t]:
                true_token = ys[i][t]
                p = max(probs[i][t][true_token], 1e-10)
                ics.append(-np.log2(p))
        melody_ics[mel_id] = np.mean(ics)

    return melody_ics


def compute_per_note_ic(model, melodies, melody_names, max_len, pitch_to_idx, idx_to_pitch):
    """
    Compute IC at every note position for each melody (full-window version).
    Uses prepare_melodies to handle padding/masking, then extracts per-note IC.
    """
    rows = []

    for name in melody_names:
        mel = [int(x) for x in melodies[name]]

        indexed = []
        valid_positions = []
        for pos, p in enumerate(mel):
            if p in pitch_to_idx:
                indexed.append(pitch_to_idx[p])
                valid_positions.append(pos)

        if len(indexed) < 2:
            print(f"Warning: {name} has fewer than 2 valid pitches, skipping")
            continue

        # Build input/target/mask for this single melody
        inp = indexed[:-1]
        tgt = indexed[1:]

        PAD = 0
        if len(inp) >= max_len:
            inp = inp[-max_len:]
            tgt = tgt[-max_len:]
            mask = [True] * max_len
            # offset: how many notes were truncated from the start
            offset = len(indexed) - 1 - max_len
        else:
            pad_len = max_len - len(inp)
            mask = [False] * pad_len + [True] * len(inp)
            inp = [PAD] * pad_len + inp
            tgt = [PAD] * pad_len + tgt
            offset = 0

        xs = np.array([inp])
        ys = np.array([tgt])
        probs = model.predict(xs, verbose=0)

        note_idx = 0
        for t in range(max_len):
            if mask[t]:
                true_token = ys[0][t]
                p = max(float(probs[0][t][true_token]), 1e-10)
                ic = -np.log2(p)
                # Map back to original position in melody
                original_pos = valid_positions[offset + note_idx + 1]
                rows.append({
                    "melody": name,
                    "note": original_pos,
                    "pitch": mel[original_pos],
                    "ic": ic,
                })
                note_idx += 1

    return pd.DataFrame(rows)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_model(
    melodies,
    max_len=128,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    dropout_rate=0.1,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    patience=5,
    save_models_dir=None,
):
    """
    Simple full-corpus training (for initial testing only).
    For proper evaluation, use run_kfold or run_cross_corpus.
    """
    xs, ys, masks, vocab_size, pitch_to_idx, idx_to_pitch = \
        prepare_melodies(melodies, max_len=max_len)

    print("Data prepared:")
    print(f"  Melodies: {len(melodies)}")
    print(f"  Vocab size: {vocab_size} ({vocab_size - 1} pitches + PAD)")
    print(f"  Sequence length: {max_len}")
    print(f"  Input shape: {xs.shape}")

    model = build_transformer(
        vocab_size=vocab_size, max_len=max_len, embed_dim=embed_dim,
        num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
        dropout_rate=dropout_rate,
    )

    print(f"\n  Total parameters: {model.count_params():,}")
    model.summary()

    model.compile(
        loss=masked_sparse_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True,
    )

    history = model.fit(
        xs, ys,
        sample_weight=masks.astype(np.float32),
        batch_size=batch_size, epochs=epochs,
        validation_split=validation_split, callbacks=[early_stop],
    )

    # Save model for reproducibility
    if save_models_dir:
        os.makedirs(save_models_dir, exist_ok=True)
        model_path = os.path.join(save_models_dir, "model.keras")
        model.save(model_path)
        print(f"Model saved to {model_path}")

        vocab_path = os.path.join(save_models_dir, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump({
                "pitch_to_idx": pitch_to_idx,
                "idx_to_pitch": {str(k): v for k, v in idx_to_pitch.items()},
                "vocab_size": vocab_size,
                "max_len": max_len,          # fixed: was window_size
            }, f, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

    return model, history, {
        "vocab_size": vocab_size,
        "pitch_to_idx": pitch_to_idx,
        "idx_to_pitch": idx_to_pitch,
        "max_len": max_len,
    }


def run_kfold(
    melodies,
    melody_ids=None,
    k=10,
    max_len=128,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    dropout_rate=0.1,
    batch_size=32,
    epochs=50,
    patience=5,
    random_state=42,
    export_folds_path=None,
    save_models_dir=None,
):
    """
    K-fold cross-validation for fair comparison with IDyOM.
    """
    if melody_ids is None:
        melody_ids = list(range(len(melodies)))

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    all_ics = []
    fold_results = []
    fold_assignments = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(melodies)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{k}")
        print(f"{'='*60}")

        train_melodies = [melodies[i] for i in train_idx]
        test_melodies = [melodies[i] for i in test_idx]
        train_ids = [melody_ids[i] for i in train_idx]
        test_ids = [melody_ids[i] for i in test_idx]

        print(f"Train: {len(train_melodies)} melodies")
        print(f"Test:  {len(test_melodies)} melodies")

        # Record fold assignments
        for idx in train_idx:
            fold_assignments.append({
                "melody_id": melody_ids[idx],
                "fold": fold + 1,
                "split": "train",
            })
        for idx in test_idx:
            fold_assignments.append({
                "melody_id": melody_ids[idx],
                "fold": fold + 1,
                "split": "test",
            })

        # Prepare data: vocabulary built from training set only
        xs_train, ys_train, masks_train, vocab_size, pitch_to_idx, idx_to_pitch = \
            prepare_melodies(train_melodies, max_len=max_len)

        # Test data uses same vocabulary
        xs_test, ys_test, masks_test, _, _, _ = \
            prepare_melodies(test_melodies, max_len=max_len, pitch_to_idx=pitch_to_idx)

        print(f"Vocab size: {vocab_size}")

        # Build and train
        model = build_transformer(
            vocab_size=vocab_size, max_len=max_len, embed_dim=embed_dim,
            num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        if fold == 0:
            print(f"Total parameters: {model.count_params():,}")

        model.compile(
            loss=masked_sparse_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        )

        checkpoint_path = tempfile.mktemp(suffix=".weights.h5")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=False
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_loss",
                save_best_only=True, save_weights_only=True,
            ),
        ]

        model.fit(
            xs_train, ys_train,
            sample_weight=masks_train.astype(np.float32),
            batch_size=batch_size, epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks, verbose=1,
        )

        # Load best weights
        model.load_weights(checkpoint_path)
        os.remove(checkpoint_path)

        # Save fold model for reproducibility
        if save_models_dir:
            os.makedirs(save_models_dir, exist_ok=True)
            model_path = os.path.join(save_models_dir, f"fold_{fold + 1}.keras")
            model.save(model_path)
            print(f"  Model saved to {model_path}")

            vocab_path = os.path.join(
                save_models_dir, f"fold_{fold + 1}_vocab.json"
            )
            with open(vocab_path, "w") as f:
                json.dump({
                    "pitch_to_idx": pitch_to_idx,
                    "idx_to_pitch": {str(k): v for k, v in idx_to_pitch.items()},
                    "vocab_size": vocab_size,
                    "max_len": max_len,
                }, f, indent=2)

        # Evaluate on test fold
        mean_ic, ics = compute_ic(model, xs_test, ys_test, masks_test)
        melody_ics = compute_per_melody_ic(
            model, xs_test, ys_test, masks_test, test_ids,
        )

        print(f"\nFold {fold + 1} mean IC: {mean_ic:.3f} bits")
        all_ics.extend(ics)
        fold_results.append({
            "fold": fold + 1,
            "mean_ic": mean_ic,
            "train_size": len(train_melodies),
            "test_size": len(test_melodies),
            "melody_ics": melody_ics,
        })

    # Export fold assignments for IDyOM replication
    df_folds = pd.DataFrame(fold_assignments)
    if export_folds_path:
        df_folds.to_csv(export_folds_path, index=False)
        print(f"\nFold assignments saved to: {export_folds_path}")

    # Summary
    overall_mean_ic = np.mean(all_ics)
    overall_std_ic = np.std(all_ics)

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS ({k}-fold cross-validation)")
    print(f"{'='*60}")
    print(f"Mean IC: {overall_mean_ic:.3f} bits")
    print(f"Std IC:  {overall_std_ic:.3f} bits")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['mean_ic']:.3f} bits"
              f" ({r['test_size']} test melodies)")

    return {
        "overall_mean_ic": overall_mean_ic,
        "overall_std_ic": overall_std_ic,
        "all_ics": all_ics,
        "fold_results": fold_results,
        "fold_assignments": df_folds,
    }


def run_cross_corpus(
    train_melodies,
    test_melodies,
    train_ids=None,
    test_ids=None,
    test_meta=None,
    max_len=128,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    dropout_rate=0.1,
    batch_size=32,
    epochs=75,
    patience=5,
    validation_split=0.1,
    test_subset=None,
    random_state=42,
    export_test_csv=None,
    save_models_dir=None,
):
    """
    Cross-corpus evaluation with full window: train on one corpus,
    test on another.

    Vocabulary is built from the training corpus only. Test melodies
    with pitches outside the training vocabulary are skipped.
    """
    if train_ids is None:
        train_ids = list(range(len(train_melodies)))
    if test_ids is None:
        test_ids = list(range(len(test_melodies)))

    # Optionally subsample test corpus
    if test_subset is not None and test_subset < len(test_melodies):
        rng = np.random.RandomState(random_state)
        subset_idx = sorted(rng.choice(len(test_melodies), size=test_subset, replace=False))
        test_melodies = [test_melodies[i] for i in subset_idx]
        test_ids = [test_ids[i] for i in subset_idx]
        print(f"Subsampled test corpus to {test_subset} melodies")

    # Export test subset metadata for IDyOM
    if export_test_csv is not None and test_meta is not None:
        subset_meta = test_meta[test_meta["melody_id"].isin(test_ids)].copy()
        subset_meta.to_csv(export_test_csv, index=False)
        print(f"Test subset saved to: {export_test_csv} ({len(subset_meta)} melodies)")

    print(f"{'='*60}")
    print(f"CROSS-CORPUS (full-window, max_len={max_len}): "
          f"Train={len(train_melodies)}  Test={len(test_melodies)}")
    print(f"{'='*60}")

    # Vocabulary from training corpus only
    xs_train, ys_train, masks_train, vocab_size, pitch_to_idx, idx_to_pitch = \
        prepare_melodies(train_melodies, max_len=max_len)

    # Test corpus reuses training vocabulary
    xs_test, ys_test, masks_test, _, _, _ = \
        prepare_melodies(test_melodies, max_len=max_len, pitch_to_idx=pitch_to_idx)

    print(f"Vocab size (from train): {vocab_size}")
    print(f"Train shape: {xs_train.shape}  Test shape: {xs_test.shape}")

    model = build_transformer(
        vocab_size=vocab_size, max_len=max_len, embed_dim=embed_dim,
        num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
        dropout_rate=dropout_rate,
    )
    print(f"Total parameters: {model.count_params():,}")

    model.compile(
        loss=masked_sparse_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True,
    )

    model.fit(
        xs_train,
        ys_train,
        sample_weight=masks_train.astype(np.float32),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1,
    )

    # Save model for reproducibility
    if save_models_dir:
        os.makedirs(save_models_dir, exist_ok=True)
        model_path = os.path.join(save_models_dir, "model.keras")
        model.save(model_path)
        print(f"Model saved to {model_path}")

        vocab_path = os.path.join(save_models_dir, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump({
                "pitch_to_idx": pitch_to_idx,
                "idx_to_pitch": {str(k): v for k, v in idx_to_pitch.items()},
                "vocab_size": vocab_size,
                "max_len": max_len,
            }, f, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

    # Evaluate on test corpus
    mean_ic, all_ics = compute_ic(model, xs_test, ys_test, masks_test)
    melody_ics = compute_per_melody_ic(model, xs_test, ys_test, masks_test, test_ids)

    # Per-note IC
    id_to_melody = dict(zip(test_ids, test_melodies))
    per_note_df = compute_per_note_ic(
        model, id_to_melody, test_ids, max_len, pitch_to_idx, idx_to_pitch,
    )

    # Add melody names from metadata for alignment with IDyOM
    if test_meta is not None:
        name_map = dict(zip(test_meta["melody_id"], test_meta["filename"]))
        per_note_df["melody_name"] = per_note_df["melody_id"].map(name_map)
        # Strip file extension if present
        per_note_df["melody_name"] = per_note_df["melody_name"].apply(
            lambda x: x.rsplit(".", 1)[0] if isinstance(x, str) and "." in x else x
        )

    # Save CSVs
    melody_ic_df = pd.DataFrame([
        {"melody_id": mel_id, "mean_ic": ic}
        for mel_id, ic in melody_ics.items()
    ])
    if test_meta is not None:
        melody_ic_df["melody_name"] = melody_ic_df["melody_id"].map(name_map)
        melody_ic_df["melody_name"] = melody_ic_df["melody_name"].apply(
            lambda x: x.rsplit(".", 1)[0] if isinstance(x, str) and "." in x else x
        )

    melody_ic_path = "cross_corpus_full_per_melody_ic.csv"
    melody_ic_df.to_csv(melody_ic_path, index=False)
    print(f"Per-melody IC saved to: {melody_ic_path} ({len(melody_ic_df)} melodies)")

    per_note_path = "cross_corpus_full_per_note_ic.csv"
    per_note_df.to_csv(per_note_path, index=False)
    print(f"Per-note IC saved to: {per_note_path} ({len(per_note_df)} notes)")

    print(f"\nCross-corpus mean IC: {mean_ic:.3f} bits")
    print(f"Cross-corpus std  IC: {np.std(all_ics):.3f} bits")

    return {
        "mean_ic": mean_ic,
        "std_ic": np.std(all_ics),
        "all_ics": all_ics,
        "melody_ics": melody_ics,
        "per_note_ic": per_note_df,
        "vocab_size": vocab_size,
        "pitch_to_idx": pitch_to_idx,
        "idx_to_pitch": idx_to_pitch,
        "test_ids": test_ids,
        "model": model,
    }


def run_hymn_ic(
    model_dir,
    hymn_lisp_path="data/hymns.lisp",
):
    """
    Compute per-note and per-melody IC for hymn melodies
    using a pre-trained full-window model.
    """
    model_path = os.path.join(model_dir, "model.keras")
    vocab_path = os.path.join(model_dir, "vocab.json")

    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        print("Run 'full_essen' experiment first.")
        exit(1)

    if not os.path.exists(hymn_lisp_path):
        print(f"Error: hymn lisp file not found at {hymn_lisp_path}")
        exit(1)

    # Load vocab and model
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    pitch_to_idx = {int(k): v for k, v in vocab_data["pitch_to_idx"].items()}
    idx_to_pitch = {int(k): v for k, v in vocab_data["idx_to_pitch"].items()}
    max_len = vocab_data["max_len"]
    vocab_size = vocab_data["vocab_size"]

    model = build_transformer(vocab_size=vocab_size, max_len=max_len)
    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Vocab size: {vocab_size}, Max len: {max_len}")

    # Load hymn melodies
    hymn_melodies = parse_lisp_melodies(hymn_lisp_path)
    melody_names = sorted(hymn_melodies.keys())

    # Out of vocab check
    oov = set(p for mel in hymn_melodies.values() for p in mel) - set(pitch_to_idx.keys())
    if oov:
        print(f"Warning: {len(oov)} OOV pitches in hymns: {sorted(oov)}")

    # Prepare hymn data using saved vocabulary
    hymn_melodies_list = [hymn_melodies[name] for name in melody_names]
    xs, ys, masks, _, _, _ = prepare_melodies(
        hymn_melodies_list, max_len=max_len, pitch_to_idx=pitch_to_idx,
    )

    # Per-note IC
    per_note_df = compute_per_note_ic(
        model, hymn_melodies, melody_names, max_len, pitch_to_idx, idx_to_pitch,
    )
    per_note_df.to_csv("hymn_transformer_full_per_note_ic.csv", index=False)
    print(f"Per-note IC saved ({len(per_note_df)} notes)")

    # Per-melody IC
    melody_ics = compute_per_melody_ic(model, xs, ys, masks, melody_names)
    melody_ic_df = pd.DataFrame([
        {"melody_id": mid, "mean_ic": mic} for mid, mic in melody_ics.items()
    ])
    melody_ic_df.to_csv("hymn_transformer_full_per_melody_ic.csv", index=False)
    print(f"Per-melody IC saved ({len(melody_ic_df)} melodies)")

    print(f"\nMean IC: {per_note_df['ic'].mean():.3f} bits")
    print(f"\nPer-melody summary:")
    print(per_note_df.groupby("melody")["ic"].agg(["mean", "std", "count"]))

    return {"per_note_ic": per_note_df, "melody_ics": melody_ic_df}


# ============================================================
# DATA LOADING HELPERS
# ============================================================
def load_corpus(name):
    """Load melodies and IDs from the pre-exported CSVs."""
    df = pd.read_csv(f"data/{name}_unique_melodies.csv")
    melodies = df["pitch"].apply(ast.literal_eval).tolist()
    melody_ids = df["melody_id"].tolist()
    return melodies, melody_ids


def load_meta(name):
    """Load melody metadata (filename, path) for IDyOM export."""
    return pd.read_csv(f"data/{name}_unique_meta_melodies.csv")


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-window transformer experiments"
    )
    parser.add_argument(
        "experiment",
        choices=[
            "full_essen",
            "full_meertens",
            "kfold_essen",
            "kfold_meertens",
            "cross_essen2meertens",
            "cross_meertens2essen",
            "hymn_ic"
        ],
        help="Which experiment to run",
    )
    parser.add_argument("--max-len", type=int, default=None,
                        help="Max sequence length (default: longest melody)")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--test-subset", type=int, default=None,
                        help="Subsample test corpus to this many melodies")
    parser.add_argument("--save-models-dir", type=str, default=None,
                        help="Directory to save trained models")
    parser.add_argument("--hymn-lisp", type=str, default="data/hymns.lisp")
    parser.add_argument("--trained-model-dir", type=str, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    if args.experiment == "full_essen":
        melodies, ids = load_corpus("essen")
        max_len = args.max_len or max(len(m) for m in melodies)
        model, history, data_info = train_model(
            melodies,
	    max_len=max_len,
            epochs=args.epochs,
            patience=args.patience,
            save_models_dir=args.save_models_dir or "models/full_window_essen",
        )

    # ------------------------------------------------------------------
    elif args.experiment == "full_meertens":
        melodies, ids = load_corpus("meertens")
        max_len = args.max_len or max(len(m) for m in melodies)
        model, history, data_info = train_model(
            melodies,
            max_len=max_len,
            epochs=args.epochs,
            patience=args.patience,
            save_models_dir=args.save_models_dir or "models/full_window_meertens",
        )

    # ------------------------------------------------------------------
    elif args.experiment == "kfold_essen":
        melodies, ids = load_corpus("essen")
        max_len = args.max_len or max(len(m) for m in melodies)
        run_kfold(
            melodies, melody_ids=ids, k=10,
            max_len=max_len,
            epochs=args.epochs,
            patience=args.patience,
            export_folds_path="full_essen_fold_assignments.csv",
            save_models_dir=args.save_models_dir or "models/kfold_full_essen",
        )

    # ------------------------------------------------------------------
    elif args.experiment == "kfold_meertens":
        melodies, ids = load_corpus("meertens")
        max_len = args.max_len or max(len(m) for m in melodies)
        run_kfold(
            melodies,
            melody_ids=ids,
            k=10,
            max_len=max_len,
            epochs=args.epochs,
            patience=args.patience,
            export_folds_path="full_meertens_fold_assignments.csv",
            save_models_dir=args.save_models_dir or "models/kfold_full_meertens",
        )

    # ------------------------------------------------------------------
    elif args.experiment == "cross_essen2meertens":
        train_mel, train_ids = load_corpus("essen")
        test_mel, test_ids = load_corpus("meertens")
        test_meta = load_meta("meertens")
        max_len = args.max_len or max(len(m) for m in train_mel)

        run_cross_corpus(
            train_melodies=train_mel,
            test_melodies=test_mel,
            train_ids=train_ids,
            test_ids=test_ids,
            test_meta=test_meta,
            max_len=max_len,
            epochs=args.epochs,
            patience=args.patience,
            test_subset=args.test_subset,
            export_test_csv="cross_corpus_full_essen2meertens.csv",
            save_models_dir=(
                args.save_models_dir
                or f"models/cross_corpus_full_essen2meertens"
            )
        )

    # ------------------------------------------------------------------
    elif args.experiment == "cross_meertens2essen":
        train_mel, train_ids = load_corpus("meertens")
        test_mel, test_ids = load_corpus("essen")
        test_meta = load_meta("essen")
        max_len = args.max_len or max(len(m) for m in train_mel)

        run_cross_corpus(
            train_melodies=train_mel,
            test_melodies=test_mel,
            train_ids=train_ids,
            test_ids=test_ids,
            test_meta=test_meta,
            max_len=max_len,
            epochs=args.epochs,
            patience=args.patience,
            export_test_csv="cross_corpus_full_meertens2essen.csv",
            save_models_dir=(
                args.save_models_dir
                or f"models/cross_corpus_full_meertens2essen"
            )
        )

    # ------------------------------------------------------------------
    elif args.experiment == "hymn_ic":
        run_hymn_ic(
            model_dir=args.trained_model_dir or "models/full_window/full_window_essen",
            hymn_lisp_path=args.hymn_lisp or "data/hymns.lisp",
        )
