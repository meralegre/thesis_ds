# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import keras
from keras import layers, ops
from sklearn.model_selection import KFold
import tempfile
import os
import ast

def get_fixed_interval_params():
    """
    Fixed interval vocabulary covering [-48, +48] semitones (4 octaves).

    This avoids any data leakage between folds and matches IDyOM's
    approach where the viewpoint alphabet is defined by the domain,
    not by the data.
    """
    min_interval = -48
    max_interval = 48
    # = 49, so 0 stays PAD
    interval_offset = -min_interval + 1
    interval_vocab_size = max_interval - min_interval + 2

    return interval_offset, interval_vocab_size

def tonic_pc_to_midi(tonic_pc, first_pitch):
    """
    Convert tonic pitch class to the nearest MIDI pitch to the melody's first note
    Ensure cpintfref values are small intervals rather than spanning multiple octaves
    """
    octave = first_pitch // 12
    candidate = octave * 12 + tonic_pc
    options = [candidate - 12, candidate, candidate + 12]
    return min(options, key=lambda x: abs(x - first_pitch))


def prepare_melodies_sliding(melodies, window_size=16, pitch_to_idx=None,
                            interval_offset=None, interval_vocab_size=None,
                            tonic_map=None, melody_ids=None):
    """
    Create sliding-window training examples with five input features
    matching IDyOM's best viewpoint: (cpint (cpcint cpintfip) (cpitch cpintfref))
    """
    if pitch_to_idx is None:
        all_pitches = sorted(set(p for mel in melodies for p in mel))
        pitch_to_idx = {p: i + 1 for i, p in enumerate(all_pitches)}

    idx_to_pitch = {i: p for p, i in pitch_to_idx.items()}
    vocab_size = max(pitch_to_idx.values()) + 1
    PAD = 0

    if interval_offset is None or interval_vocab_size is None:
        interval_offset, interval_vocab_size = get_fixed_interval_params()

    # cpcint
    CPCINT_VOCAB_SIZE = 13

    xs_pitch = []
    xs_cpint = []
    xs_cpcint = []
    xs_cpintfip = []
    xs_cpintfref = []
    ys = []
    skipped = 0
    n_with_tonic = 0
    n_without_tonic = 0

    for mel_idx, mel in enumerate(melodies):
        indexed = [pitch_to_idx[p] for p in mel if p in pitch_to_idx]
        raw = [p for p in mel if p in pitch_to_idx]

        if len(indexed) < 2:
            skipped += 1
            continue

        first_pitch = raw[0]

        # Get tonic for cpintfref (fall back to first pitch if no key data)
        tonic_midi = first_pitch
        if tonic_map is not None and melody_ids is not None:
            tonic_pc = tonic_map.get(melody_ids[mel_idx])
            if tonic_pc is not None:
                tonic_midi = tonic_pc_to_midi(tonic_pc, first_pitch)
                n_with_tonic += 1
            else:
                n_without_tonic += 1
        else:
            n_without_tonic += 1

        for t in range(1, len(indexed)):
            start = max(0, t - window_size)
            context = indexed[start:t]
            raw_context = raw[start:t]

            # cpint: interval from previous note
            raw_cpint = [0]
            for i in range(1, len(raw_context)):
                raw_cpint.append(raw_context[i] - raw_context[i - 1])

            # cpcint: pitch-class interval (cpint mod 12)
            raw_cpcint = [0]
            for i in range(1, len(raw_context)):
                raw_cpcint.append((raw_context[i] - raw_context[i - 1]) % 12)

            # cpintfip: interval from first pitch of piece
            raw_cpintfip = [p - first_pitch for p in raw_context]

            # cpintfref: interval from tonal referent (tonic)
            raw_cpintfref = [p - tonic_midi for p in raw_context]

            # Left-pad
            pad_len = window_size - len(context)

            pitch_seq = [PAD] * pad_len + context

            cpint_seq = [PAD] * pad_len
            for iv in raw_cpint:
                shifted = iv + interval_offset
                shifted = max(1, min(shifted, interval_vocab_size - 1))
                cpint_seq.append(shifted)

            cpcint_seq = [PAD] * pad_len
            for iv in raw_cpcint:
                cpcint_seq.append(iv + 1)

            cpintfip_seq = [PAD] * pad_len
            for iv in raw_cpintfip:
                shifted = iv + interval_offset
                shifted = max(1, min(shifted, interval_vocab_size - 1))
                cpintfip_seq.append(shifted)

            cpintfref_seq = [PAD] * pad_len
            for iv in raw_cpintfref:
                shifted = iv + interval_offset
                shifted = max(1, min(shifted, interval_vocab_size - 1))
                cpintfref_seq.append(shifted)

            xs_pitch.append(pitch_seq)
            xs_cpint.append(cpint_seq)
            xs_cpcint.append(cpcint_seq)
            xs_cpintfip.append(cpintfip_seq)
            xs_cpintfref.append(cpintfref_seq)
            ys.append(indexed[t])

    if skipped > 0:
        print(f"  Warning: skipped {skipped} melodies (too short or unknown pitches)")

    print(f"  Total training examples: {len(xs_pitch)} (from {len(melodies)} melodies)")
    if tonic_map is not None:
        print(f"  Tonic info: {n_with_tonic} melodies with key, {n_without_tonic} using first-pitch fallback")

    return (np.array(xs_pitch), np.array(xs_cpint), np.array(xs_cpcint),
            np.array(xs_cpintfip), np.array(xs_cpintfref), np.array(ys),
            vocab_size, pitch_to_idx, idx_to_pitch)


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
    """

    def __init__(self, embed_dim, num_heads, ff_dim, window_size, dropout_rate=0.1):
        super().__init__()
        self.window_size = window_size
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
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
    interval_vocab_size,
    cpcint_vocab_size=13,
    window_size=16,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=2,
    dropout_rate=0.1,
):
    """
    Transformer with five input streams matching IDyOM's best viewpoint
    system: (cpint (cpcint cpintfip) (cpitch cpintfref))
    Each viewpoint gets its own embedding.
    """
    pitch_input = keras.Input(shape=(window_size,), name="pitch")
    cpint_input = keras.Input(shape=(window_size,), name="cpint")
    cpcint_input = keras.Input(shape=(window_size,), name="cpcint")
    cpintfip_input = keras.Input(shape=(window_size,), name="cpintfip")
    cpintfref_input = keras.Input(shape=(window_size,), name="cpintfref")

    # Pitch embedding with positional encoding
    pitch_emb = TokenAndPositionEmbedding(window_size, vocab_size, embed_dim)(pitch_input)

    # Viewpoint embeddings (no positional encoding)
    cpint_emb = layers.Embedding(interval_vocab_size, embed_dim)(cpint_input)
    cpcint_emb = layers.Embedding(cpcint_vocab_size, embed_dim)(cpcint_input)
    cpintfip_emb = layers.Embedding(interval_vocab_size, embed_dim)(cpintfip_input)
    cpintfref_emb = layers.Embedding(interval_vocab_size, embed_dim)(cpintfref_input)

    # Combine all streams
    x = layers.Add()([pitch_emb, cpint_emb, cpcint_emb, cpintfip_emb, cpintfref_emb])

    for _ in range(num_layers):
        x = SlidingWindowTransformerBlock(
            embed_dim, num_heads, ff_dim, window_size, dropout_rate
        )(x)

    x = x[:, -1, :]
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(
        inputs=[pitch_input, cpint_input, cpcint_input, cpintfip_input, cpintfref_input],
        outputs=outputs,
    )
    return model


def compute_ic(model, xs_pitch, xs_cpint, xs_cpcint, xs_cpintfip, xs_cpintfref, ys):
    """
    Compute Information Content (IC) in bits for each note prediction.
    Uses batched prediction to avoid GPU memory issues.
    """
    inputs = [xs_pitch, xs_cpint, xs_cpcint, xs_cpintfip, xs_cpintfref]

    # Predict in batches to avoid GPU compilation errors
    batch_size = 1024
    n = len(xs_pitch)
    all_probs = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [x[start:end] for x in inputs]
        batch_probs = model(batch, training=False)
        all_probs.append(batch_probs.numpy())

    probs = np.concatenate(all_probs, axis=0)

    all_ics = []
    for i in range(n):
        p = max(probs[i][ys[i]], 1e-10)
        all_ics.append(-np.log2(p))

    return np.mean(all_ics), all_ics


def compute_ic_per_melody(model, melodies, window_size, pitch_to_idx,
                          interval_offset, interval_vocab_size,
                          tonic_map=None, melody_ids=None):
    """
    Compute mean IC per melody using sliding windows.
    """
    PAD = 0
    melody_ics = []

    for mel_idx, mel in enumerate(melodies):
        indexed = [pitch_to_idx[p] for p in mel if p in pitch_to_idx]
        raw = [p for p in mel if p in pitch_to_idx]
        if len(indexed) < 2:
            continue

        first_pitch = raw[0]

        # Get tonic for cpintfref
        tonic_midi = first_pitch
        if tonic_map is not None and melody_ids is not None:
            tonic_pc = tonic_map.get(melody_ids[mel_idx])
            if tonic_pc is not None:
                tonic_midi = tonic_pc_to_midi(tonic_pc, first_pitch)

        mel_xs_pitch = []
        mel_xs_cpint = []
        mel_xs_cpcint = []
        mel_xs_cpintfip = []
        mel_xs_cpintfref = []
        mel_ys = []

        for t in range(1, len(indexed)):
            start = max(0, t - window_size)
            context = indexed[start:t]
            raw_context = raw[start:t]

            # cpint
            raw_cpint = [0]
            for i in range(1, len(raw_context)):
                raw_cpint.append(raw_context[i] - raw_context[i - 1])

            # cpcint
            raw_cpcint = [0]
            for i in range(1, len(raw_context)):
                raw_cpcint.append((raw_context[i] - raw_context[i - 1]) % 12)

            # cpintfip
            raw_cpintfip = [p - first_pitch for p in raw_context]

            # cpintfref (from tonic)
            raw_cpintfref = [p - tonic_midi for p in raw_context]

            # Left-pad
            pad_len = window_size - len(context)

            pitch_seq = [PAD] * pad_len + context

            cpint_seq = [PAD] * pad_len
            for iv in raw_cpint:
                shifted = iv + interval_offset
                shifted = max(1, min(shifted, interval_vocab_size - 1))
                cpint_seq.append(shifted)

            cpcint_seq = [PAD] * pad_len
            for iv in raw_cpcint:
                cpcint_seq.append(iv + 1)

            cpintfip_seq = [PAD] * pad_len
            for iv in raw_cpintfip:
                shifted = iv + interval_offset
                shifted = max(1, min(shifted, interval_vocab_size - 1))
                cpintfip_seq.append(shifted)

            cpintfref_seq = [PAD] * pad_len
            for iv in raw_cpintfref:
                shifted = iv + interval_offset
                shifted = max(1, min(shifted, interval_vocab_size - 1))
                cpintfref_seq.append(shifted)

            mel_xs_pitch.append(pitch_seq)
            mel_xs_cpint.append(cpint_seq)
            mel_xs_cpcint.append(cpcint_seq)
            mel_xs_cpintfip.append(cpintfip_seq)
            mel_xs_cpintfref.append(cpintfref_seq)
            mel_ys.append(indexed[t])

        mel_xs_pitch = np.array(mel_xs_pitch)
        mel_xs_cpint = np.array(mel_xs_cpint)
        mel_xs_cpcint = np.array(mel_xs_cpcint)
        mel_xs_cpintfip = np.array(mel_xs_cpintfip)
        mel_xs_cpintfref = np.array(mel_xs_cpintfref)
        mel_ys = np.array(mel_ys)

        probs = model(
            [mel_xs_pitch, mel_xs_cpint, mel_xs_cpcint,
             mel_xs_cpintfip, mel_xs_cpintfref], training=False
        ).numpy()

        ics = []
        for i in range(len(mel_xs_pitch)):
            p = max(probs[i][mel_ys[i]], 1e-10)
            ics.append(-np.log2(p))

        melody_ics.append({
            "melody_idx": mel_idx,
            "mean_ic": np.mean(ics),
            "n_notes": len(ics),
        })

    return melody_ics


#     IDyOM command per fold:
# (idyom:idyom <test-dataset-id> '(cpitch) '(cpitch) :models :ltm :pretraining-ids '(<train-dataset-id>))

def run_kfold(
    melodies,
    melody_ids=None,
    tonic_map=None,
    k=10,
    window_size=16,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=2,
    dropout_rate=0.1,
    batch_size=32,
    epochs=50,
    patience=5,
    random_state=42,
    export_folds_path=None,
):
    """
    K-fold cross-validation for fair comparison with IDyOM.
    """
    if melody_ids is None:
        melody_ids = list(range(len(melodies)))

    # Fixed interval params - no data leakage
    interval_offset, interval_vocab_size = get_fixed_interval_params()

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

        # Prepare training data
        xs_pitch_train, xs_cpint_train, xs_cpcint_train, xs_cpintfip_train, xs_cpintfref_train, \
            ys_train, vocab_size, pitch_to_idx, _ = \
            prepare_melodies_sliding(
                train_melodies, window_size=window_size,
                interval_offset=interval_offset, interval_vocab_size=interval_vocab_size,
                tonic_map=tonic_map, melody_ids=train_ids,
            )

        # Test data: reuse training pitch vocab
        xs_pitch_test, xs_cpint_test, xs_cpcint_test, xs_cpintfip_test, xs_cpintfref_test, \
            ys_test, _, _, _ = \
            prepare_melodies_sliding(
                test_melodies, window_size=window_size,
                pitch_to_idx=pitch_to_idx,
                interval_offset=interval_offset, interval_vocab_size=interval_vocab_size,
                tonic_map=tonic_map, melody_ids=test_ids,
            )

        print(f"Vocab size: {vocab_size}")
        print(f"Interval vocab size: {interval_vocab_size}")

        # Build and train
        model = build_transformer(
            vocab_size=vocab_size, interval_vocab_size=interval_vocab_size,
            window_size=window_size, embed_dim=embed_dim,
            num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        if fold == 0:
            print(f"Total parameters: {model.count_params():,}")

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        )

        # Manual validation split
        n = len(ys_train)
        n_val = int(n * 0.1)
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx_inner = indices[n_val:]

        all_train_xs = [xs_pitch_train, xs_cpint_train, xs_cpcint_train,
                        xs_cpintfip_train, xs_cpintfref_train]
        train_inputs = [x[train_idx_inner] for x in all_train_xs]
        val_inputs = [x[val_idx] for x in all_train_xs]

        # ModelCheckpoint instead of restore_best_weights
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
            train_inputs, ys_train[train_idx_inner],
            validation_data=(val_inputs, ys_train[val_idx]),
            batch_size=batch_size, epochs=epochs,
            callbacks=callbacks, verbose=1,
        )

        # Load best weights
        model.load_weights(checkpoint_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Evaluate on test fold
        mean_ic, ics = compute_ic(
            model, xs_pitch_test, xs_cpint_test, xs_cpcint_test,
            xs_cpintfip_test, xs_cpintfref_test, ys_test,
        )
        melody_ics = compute_ic_per_melody(
            model, test_melodies, window_size, pitch_to_idx,
            interval_offset, interval_vocab_size,
            tonic_map=tonic_map, melody_ids=test_ids,
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

    # Export fold assignments
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
        print(f"  Fold {r['fold']}: {r['mean_ic']:.3f} bits ({r['test_size']} test melodies)")

    return {
        "overall_mean_ic": overall_mean_ic,
        "overall_std_ic": overall_std_ic,
        "all_ics": all_ics,
        "fold_results": fold_results,
        "fold_assignments": df_folds,
    }

"""## Essen"""

essen_melodies = pd.read_csv("essen_unique_melodies.csv")["pitch"].apply(ast.literal_eval).tolist()
esseb_melody_ids = pd.read_csv("essen_unique_melodies.csv")["melody_id"].tolist()

# essen_melodies = pd.read_csv("essen_unique_melodies.csv")["pitch"].apply(ast.literal_eval).tolist()

# model, history, data_info = train_model(essen_melodies, window_size=10, epochs=10)

# interval_offset = data_info["interval_offset"]
# interval_vocab_size = data_info["interval_vocab_size"]

# xs_pitch, xs_cpint, xs_cpcint, xs_cpintfip, xs_cpintfref, ys, _, _, _ = \
#     prepare_melodies_sliding(
#         essen_melodies, window_size=10,
#         interval_offset=interval_offset, interval_vocab_size=interval_vocab_size,
#     )
# mean_ic, all_ics = compute_ic(model, xs_pitch, xs_cpint, xs_cpcint, xs_cpintfip, xs_cpintfref, ys)
# print(f"\nMean IC: {mean_ic:.3f} bits")

"""## Meertens"""

meertens_melodies = pd.read_csv("meertens_unique_melodies.csv")["pitch"].apply(ast.literal_eval).tolist()
meertens_melody_ids = pd.read_csv("meertens_unique_melodies.csv")["melody_id"].tolist()

# meertens_model, meertens_history, meertens_data_info = train_model(
#     meertens_melodies,
#     window_size=10,
#     epochs=50,
# )

# Compute IC
# interval_offset = meertens_data_info["interval_offset"]
# interval_vocab_size = meertens_data_info["interval_vocab_size"]

# xs_pitch, xs_cpint, xs_cpcint, xs_cpintfip, xs_cpintfref, ys, _, _, _ = \
#     prepare_melodies_sliding(
#         meertens_melodies, window_size=10,
#         interval_offset=interval_offset, interval_vocab_size=interval_vocab_size,
#     )
# mean_ic, all_ics = compute_ic(meertens_model, xs_pitch, xs_cpint, xs_cpcint, xs_cpintfip, xs_cpintfref, ys)
# print(f"\nMean IC: {mean_ic:.3f} bits")

# Meertens
results = run_kfold(
    melodies=meertens_melodies,
    melody_ids=meertens_melody_ids,
    k=5,
    window_size=10,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    dropout_rate=0.1,
    batch_size=32,
    epochs=75,
    patience=10,
    export_folds_path="meertens_fold_assignments_viewpoints.csv",
)

