import pandas as pd
import numpy as np
import keras
from keras import layers, ops
from sklearn.model_selection import KFold

import ast
import os
import re
import json
import random
import argparse
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    """Set random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_melodies(melodies, window_size=10, pitch_to_idx=None):
    """
    Create sliding-window training examples from melodies.

    Instead of truncating long melodies, each melody produces multiple
    overlapping windows. Every note in the corpus gets predicted exactly once.

    For predicting note at position t, the input is the window_size notes
    before it (left-padded if t < window_size).
    """
    if pitch_to_idx is None:
        all_pitches = sorted(set(p for mel in melodies for p in mel))
        pitch_to_idx = {p: i + 1 for i, p in enumerate(all_pitches)}

    idx_to_pitch = {i: p for p, i in pitch_to_idx.items()}
    vocab_size = max(pitch_to_idx.values()) + 1
    PAD = 0

    xs = []
    ys = []
    skipped = 0

    for mel in melodies:
        indexed = [pitch_to_idx[p] for p in mel if p in pitch_to_idx]

        if len(indexed) < 2:
            skipped += 1
            continue

        for t in range(1, len(indexed)):
            start = max(0, t - window_size)
            context = indexed[start:t]
            pad_len = window_size - len(context)
            inp = [PAD] * pad_len + context

            xs.append(inp)
            ys.append(indexed[t])

    if skipped > 0:
        print(f"Warning: skipped {skipped} melodies (too short or unknown pitches)")

    print(f"Total training examples: {len(xs)} (from {len(melodies)} melodies)")

    return (np.array(xs), np.array(ys),
            vocab_size, pitch_to_idx, idx_to_pitch)


def parse_lisp_melodies(filepath, viewpoint=":CPITCH"):
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


class SlidingWindowTransformerBlock(layers.Layer):
    """
    Transformer block where position i can only attend to
    positions [i-window_size+1, ..., i].

    This matches IDyOM's order bound: at each prediction step,
    only the previous `window_size` notes are visible.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, window_size, dropout_rate=0.1):
        super().__init__()
        self.window_size = window_size
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def _sliding_window_mask(self, seq_len):
        """Create a causal sliding window attention mask.

        mask[i, j] = True (attend) if j <= i AND j >= i - window_size + 1
        """
        causal = ops.tril(ops.ones((seq_len, seq_len), dtype="bool"))
        rows = ops.arange(seq_len)[:, None]
        cols = ops.arange(seq_len)[None, :]
        window = (rows - cols) < self.window_size
        mask = ops.cast(causal & window, dtype="bool")
        return mask

    def call(self, inputs, training=False):
        seq_len = ops.shape(inputs)[1]
        mask = self._sliding_window_mask(seq_len)

        attn_output = self.att(
            inputs, inputs, attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer(
    vocab_size,
    window_size=16,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    dropout_rate=0.1,
):
    """
    The window_size parameter is analogous to IDyOM's order bound:
    each position can only attend to the previous window_size positions.

    Default configuration is deliberately small for fair comparison with IDyOM
    """
    inputs = keras.Input(shape=(window_size,))
    x = TokenAndPositionEmbedding(window_size, vocab_size, embed_dim)(inputs)

    for _ in range(num_layers):
        x = SlidingWindowTransformerBlock(
            embed_dim, num_heads, ff_dim, window_size, dropout_rate
        )(x)

    # Output: predict from last position only
    x = x[:, -1, :]
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================
# IC COMPUTATION
# ============================================================
def compute_ic(model, xs, ys):
    """
    Compute Information Content (IC) in bits for each note prediction.
    """
    probs = model.predict(xs, verbose=0)

    all_ics = []
    for i in range(len(xs)):
        p = probs[i][ys[i]]
        p = max(p, 1e-10)
        all_ics.append(-np.log2(p))

    return np.mean(all_ics), all_ics


def compute_per_melody_ic(model, melodies, window_size, pitch_to_idx):
    """
    Compute mean IC per melody using sliding windows.
    """
    PAD = 0
    melody_ics = []

    for mel_idx, mel in enumerate(melodies):
        indexed = [pitch_to_idx[p] for p in mel if p in pitch_to_idx]
        if len(indexed) < 2:
            continue

        mel_xs = []
        mel_ys = []

        for t in range(1, len(indexed)):
            start = max(0, t - window_size)
            context = indexed[start:t]
            pad_len = window_size - len(context)
            inp = [PAD] * pad_len + context

            mel_xs.append(inp)
            mel_ys.append(indexed[t])

        mel_xs = np.array(mel_xs)
        mel_ys = np.array(mel_ys)
        probs = model.predict(mel_xs, verbose=0)

        ics = []
        for i in range(len(mel_xs)):
            p = max(probs[i][mel_ys[i]], 1e-10)
            ics.append(-np.log2(p))

        melody_ics.append({
            "melody_idx": mel_idx,
            "mean_ic": np.mean(ics),
            "n_notes": len(ics),
        })

    return melody_ics


def compute_per_note_ic(
        model,
        melodies,
        melody_names,
        window_size,
        pitch_to_idx,
        idx_to_pitch
):
    """
    Compute IC at every note position for each melody.
    """
    PAD = 0
    rows = []

    for name in melody_names:
        mel = melodies[name]

        # Map pitches to vocab indices; track OOV
        indexed = []
        valid_positions = []
        for pos, p in enumerate(mel):
            if int(p) in pitch_to_idx:
                indexed.append(pitch_to_idx[int(p)])
                valid_positions.append(pos)
            else:
                print(f"Warning: pitch {p} in {name} not in vocabulary, skipping")

        if len(indexed) < 2:
            print(f"Warning: {name} has fewer than 2 valid pitches, skipping")
            continue

        # Build sliding window inputs for each note position
        mel_xs = []
        mel_ys = []
        note_positions = []

        for t in range(1, len(indexed)):
            start = max(0, t - window_size)
            context = indexed[start:t]
            pad_len = window_size - len(context)
            inp = [PAD] * pad_len + context

            mel_xs.append(inp)
            mel_ys.append(indexed[t])
            note_positions.append(valid_positions[t])

        mel_xs = np.array(mel_xs)
        mel_ys = np.array(mel_ys)

        # Batch predict
        probs = model.predict(mel_xs, verbose=0)

        for i in range(len(mel_xs)):
            p = max(probs[i][mel_ys[i]], 1e-10)
            ic = -np.log2(p)

            # Predicted pitch
            predicted_token = int(np.argmax(probs[i]))
            predicted_pitch = idx_to_pitch[predicted_token]
            
            rows.append({
                "melody": name,
                "note": note_positions[i],
                "pitch": mel[note_positions[i]],
                "ic": ic,
                "predicted_pitch": predicted_pitch,
                "correct_pitch": predicted_pitch == mel[note_positions[i]],
            })

    return pd.DataFrame(rows)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_model(
    melodies,
    window_size=16,
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
    Full-corpus training (no cross-validation).
    Use for initial testing or when training on an entire corpus
    to evaluate on a separate test set (e.g., hymn melodies).
    """
    xs, ys, vocab_size, pitch_to_idx, idx_to_pitch = \
        prepare_melodies(melodies, window_size=window_size)

    print(f"Data prepared:")
    print(f"  Melodies: {len(melodies)}")
    print(f"  Vocab size: {vocab_size} ({vocab_size - 1} pitches + PAD)")
    print(f"  Window size: {window_size}")
    print(f"  Training examples: {xs.shape[0]}")

    model = build_transformer(
        vocab_size=vocab_size, window_size=window_size, embed_dim=embed_dim,
        num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
        dropout_rate=dropout_rate,
    )

    print(f"\n  Total parameters: {model.count_params():,}")
    model.summary()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True,
    )

    history = model.fit(
        xs, ys,
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
                "window_size": window_size,
            }, f, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

    return model, history, {
        "vocab_size": vocab_size,
        "pitch_to_idx": pitch_to_idx,
        "idx_to_pitch": idx_to_pitch,
        "window_size": window_size,
    }



def run_kfold(
    melodies,
    melody_ids=None,
    k=10,
    window_size=16,
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

    Both models must use identical train/test splits. This function:
    1. Splits melodies into k folds at the piece level
    2. For each fold, trains on training melodies, evaluates on test melodies
    3. Exports fold assignments so IDyOM can use the same splits
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

        print(f"Train: {len(train_melodies)} melodies")
        print(f"Test: {len(test_melodies)} melodies")

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
        xs_train, ys_train, vocab_size, pitch_to_idx, idx_to_pitch = \
            prepare_melodies(train_melodies, window_size=window_size)

        # Test data uses same vocabulary
        xs_test, ys_test, _, _, _ = \
            prepare_melodies(test_melodies, window_size=window_size, pitch_to_idx=pitch_to_idx)

        print(f"Vocab size: {vocab_size}")

        # Build and train
        model = build_transformer(
            vocab_size=vocab_size, window_size=window_size, embed_dim=embed_dim,
            num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        if fold == 0:
            print(f"Total parameters: {model.count_params():,}")

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
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
            batch_size=batch_size, epochs=epochs,
            validation_split=0.1, callbacks=callbacks, verbose=1,
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
                    "window_size": window_size,
                }, f, indent=2)

        # Compute IC on test fold
        mean_ic, ics = compute_ic(model, xs_test, ys_test)
        melody_ics = compute_per_melody_ic(
            model, test_melodies, window_size, pitch_to_idx,
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
    window_size=10,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    dropout_rate=0.1,
    batch_size=32,
    epochs=75,
    patience=5,
    test_subset=None,
    random_state=42,
    export_test_csv=None,
    save_models_dir=None,
):
    """
    Cross-corpus evaluation with sliding window: train on one corpus,
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
    print(f"CROSS-CORPUS (sliding, window={window_size}): "
          f"Train={len(train_melodies)}  Test={len(test_melodies)}")
    print(f"{'='*60}")

    # Prepare training data (vocabulary from training corpus only)
    xs_train, ys_train, vocab_size, pitch_to_idx, idx_to_pitch = \
        prepare_melodies(train_melodies, window_size=window_size)

    # Prepare test data (reuses training vocabulary)
    xs_test, ys_test, _, _, _ = \
        prepare_melodies(test_melodies, window_size=window_size,
                                pitch_to_idx=pitch_to_idx)

    print(f"Vocab size (from train): {vocab_size}")
    print(f"Train examples: {len(ys_train)}  Test examples: {len(ys_test)}")

    # Build model
    model = build_transformer(
        vocab_size=vocab_size, window_size=window_size, embed_dim=embed_dim,
        num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
        dropout_rate=dropout_rate,
    )
    print(f"Total parameters: {model.count_params():,}")

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    )

    # Manual validation split
    n = len(ys_train)
    n_val = int(n * 0.1)
    indices = np.random.RandomState(random_state).permutation(n)
    val_idx = indices[:n_val]
    train_idx_inner = indices[n_val:]

    checkpoint_path = tempfile.mktemp(suffix=".weights.h5")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=False,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss",
            save_best_only=True, save_weights_only=True,
        ),
    ]

    model.fit(
        xs_train[train_idx_inner], ys_train[train_idx_inner],
        validation_data=(xs_train[val_idx], ys_train[val_idx]),
        batch_size=batch_size, epochs=epochs,
        callbacks=callbacks, verbose=1,
    )

    model.load_weights(checkpoint_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

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
                "window_size": window_size,
            }, f, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

    # Evaluate on test corpus
    mean_ic, all_ics = compute_ic(model, xs_test, ys_test)
    melody_ics = compute_per_melody_ic(model, test_melodies, window_size, pitch_to_idx)

    # Map melody_idx back to actual melody IDs
    for mic in melody_ics:
        mic["melody_id"] = test_ids[mic["melody_idx"]]

    # Per-note IC
    id_to_melody = dict(zip(test_ids, test_melodies))
    per_note_df = compute_per_note_ic(
        model, id_to_melody, test_ids, window_size, pitch_to_idx, idx_to_pitch
    )

    # Save CSVs
    base = export_test_csv.replace("_test.csv", "") if export_test_csv else "cross_corpus_sliding"

    melody_ic_df = pd.DataFrame(melody_ics)
    melody_ic_path = f"{base}_per_melody_ic.csv"
    melody_ic_df.to_csv(melody_ic_path, index=False)
    print(f"Per-melody IC saved to: {melody_ic_path} ({len(melody_ic_df)} melodies)")

    per_note_path = f"{base}_per_note_ic.csv"
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
    hymn_lisp_path=None,
):
    """
    Compute per-note and per-melody IC for hymn melodies
    using a pre-trained model.
    """
    if hymn_lisp_path is None:
        hymn_lisp_path = ROOT / "data" / "hymns.lisp"
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
    window_size = vocab_data["window_size"]
    vocab_size = vocab_data["vocab_size"]

    model = build_transformer(vocab_size=vocab_size, window_size=window_size)
    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Vocab size: {vocab_size}, Window size: {window_size}")

    # Load hymn melodies
    hymn_melodies = parse_lisp_melodies(hymn_lisp_path)
    melody_names = sorted(hymn_melodies.keys())

    # Out of vocab check
    oov = set(p for mel in hymn_melodies.values() for p in mel) - set(pitch_to_idx.keys())
    if oov:
        print(f"Warning: {len(oov)} OOV pitches in hymns: {sorted(oov)}")

    # Per-note IC
    per_note_df = compute_per_note_ic(
        model, hymn_melodies, melody_names, window_size, pitch_to_idx, vocab_data["idx_to_pitch"]
    )
    per_note_df.to_csv("hymn_sliding_transformer_per_note_ic.csv", index=False)
    print(f"Per-note IC saved ({len(per_note_df)} notes)")

    # Per-melody IC
    hymn_melodies_list = [hymn_melodies[name] for name in melody_names]
    melody_ics = compute_per_melody_ic(
        model, hymn_melodies_list, window_size, pitch_to_idx,
    )
    for i, mic in enumerate(melody_ics):
        mic["melody_id"] = melody_names[i]

    melody_ic_df = pd.DataFrame(melody_ics)
    melody_ic_df.to_csv("hymn_sliding_transformer_per_melody_ic.csv", index=False)
    print(f"Per-melody IC saved ({len(melody_ic_df)} melodies)")

    print(f"\nMean IC: {per_note_df['ic'].mean():.3f} bits")
    print("\nPer-melody summary:")
    print(per_note_df.groupby("melody")["ic"].agg(["mean", "std", "count"]))

    return {"per_note_ic": per_note_df, "melody_ics": melody_ic_df}


# ============================================================
# DATA LOADING HELPERS
# ============================================================
def load_corpus(name):
    """Load melodies and IDs from the pre-exported CSVs."""
    df = pd.read_csv(ROOT / "data" / f"{name}_unique_melodies.csv")
    melodies = df["pitch"].apply(ast.literal_eval).tolist()
    melody_ids = df["melody_id"].tolist()
    return melodies, melody_ids


def load_meta(name):
    """Load melody metadata (filename, path) for IDyOM export."""
    return pd.read_csv(ROOT / "data" / f"{name}_unique_meta_melodies.csv")


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sliding-window transformer experiments"
    )
    parser.add_argument("experiment", choices=[
        "full_essen",
        "full_meertens",
        "kfold_essen",
        "kfold_meertens",
        "cross_essen2meertens",
        "cross_meertens2essen",
        "hymn_ic",
    ])
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--test-subset", type=int, default=None,
                        help="Limit test set size for cross-corpus (debugging)")
    parser.add_argument("--save-models-dir", type=str, default=None,
                        help="Directory to save trained models")
    parser.add_argument("--hymn-lisp", type=str, default=None,
                        help="Path to hymn melodies lisp export from IDyOM")
    parser.add_argument("--trained-model-dir", type=str, default=None,
                        help="Directory with a trained model to load (for hymn_ic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    if args.experiment == "full_essen":
        melodies, ids = load_corpus("essen")
        model, history, data_info = train_model(
            melodies, window_size=args.window_size,
            epochs=args.epochs, patience=args.patience,
            save_models_dir=args.save_models_dir or "../pretrained_models/sliding_window/sliding_essen",
        )

    # ------------------------------------------------------------------
    elif args.experiment == "full_meertens":
        melodies, ids = load_corpus("meertens")
        model, history, data_info = train_model(
            melodies, window_size=args.window_size,
            epochs=args.epochs, patience=args.patience,
            save_models_dir=args.save_models_dir or "../pretrained_models/sliding_window/sliding_meertens",
        )

    # ------------------------------------------------------------------
    elif args.experiment == "kfold_essen":
        melodies, ids = load_corpus("essen")
        results = run_kfold(
            melodies, melody_ids=ids, k=10,
            window_size=args.window_size, epochs=args.epochs,
            patience=args.patience,
            export_folds_path=(
                f"essen_{args.window_size}_folds_sliding.csv"
            ),
            save_models_dir=(
                args.save_models_dir or "../pretrained_models/sliding_window/kfold_essen"
            ),
        )

    # ------------------------------------------------------------------
    elif args.experiment == "kfold_meertens":
        melodies, ids = load_corpus("meertens")
        results = run_kfold(
            melodies, melody_ids=ids, k=10,
            window_size=args.window_size, epochs=args.epochs,
            patience=args.patience,
            export_folds_path=(
                f"meertens_{args.window_size}_folds_sliding.csv"
            ),
            save_models_dir=(
                args.save_models_dir
                or "../pretrained_models/sliding_window/kfold_meertens"
            ),
        )

    # ------------------------------------------------------------------
    elif args.experiment == "cross_essen2meertens":
        train_mel, train_ids = load_corpus("essen")
        test_mel, test_ids = load_corpus("meertens")
        test_meta = load_meta("meertens")

        run_cross_corpus(
            train_melodies=train_mel, test_melodies=test_mel,
            train_ids=train_ids, test_ids=test_ids,
            test_meta=test_meta,
            window_size=args.window_size, epochs=args.epochs,
            patience=args.patience,
            test_subset=args.test_subset,
            export_test_csv="cross_corpus_sliding_essen2meertens_test.csv",
            save_models_dir=(
                args.save_models_dir
                or "../pretrained_models/sliding_window/cross_corpus_sliding_essen2meertens"
            )
        )

    # ------------------------------------------------------------------
    elif args.experiment == "cross_meertens2essen":
        train_mel, train_ids = load_corpus("meertens")
        test_mel, test_ids = load_corpus("essen")
        test_meta = load_meta("essen")

        run_cross_corpus(
            train_melodies=train_mel, test_melodies=test_mel,
            train_ids=train_ids, test_ids=test_ids,
            test_meta=test_meta,
            window_size=args.window_size, epochs=args.epochs,
            patience=args.patience,
            export_test_csv="cross_corpus_sliding_meertens2essen_test.csv",
            save_models_dir=(
                args.save_models_dir
                or "../pretrained_models/sliding_window/cross_corpus_sliding_meertens2essen"
            )
        )

    # ------------------------------------------------------------------
    elif args.experiment == "hymn_ic":
        run_hymn_ic(
            model_dir=args.trained_model_dir or "../pretrained_models/sliding_window/sliding_essen",
            hymn_lisp_path=args.hymn_lisp or "../data/hymns.lisp",
        )
