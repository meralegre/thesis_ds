import pandas as pd
import numpy as np
import keras
from keras import layers, ops
from sklearn.model_selection import KFold
import tempfile

import os
import re
import ast
import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# ====================================================
# REPRODUCIBILITY
# ====================================================
def set_seed(seed=42):
    """Set random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ====================================================
# INTERVAL PARAMS
# ====================================================
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


# ====================================================
# DATA PREPARATION
# ====================================================
def prepare_melodies_sliding(
    melodies,
    window_size=10,
    pitch_to_idx=None,
    interval_offset=None,
    interval_vocab_size=None,
    tonic_map=None,
    melody_ids=None
):
    """
    Create sliding-window training examples with five input features
    matching IDyOM's best viewpoint: ((cpintfip cpintfref) (cpcint cpintfip) (cpitch cpintfref))
    """
    if pitch_to_idx is None:
        all_pitches = sorted(set(p for mel in melodies for p in mel))
        pitch_to_idx = {p: i + 1 for i, p in enumerate(all_pitches)}

    idx_to_pitch = {i: p for p, i in pitch_to_idx.items()}
    vocab_size = max(pitch_to_idx.values()) + 1
    PAD = 0

    if interval_offset is None or interval_vocab_size is None:
        interval_offset, interval_vocab_size = get_fixed_interval_params()

    xs_pitch = []
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

            # cpcint: pitch-class interval (intervals mod 12)
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
            xs_cpcint.append(cpcint_seq)
            xs_cpintfip.append(cpintfip_seq)
            xs_cpintfref.append(cpintfref_seq)
            ys.append(indexed[t])

    if skipped > 0:
        print(f"Warning: skipped {skipped} melodies (too short or unknown pitches)")

    print(f"Total training examples: {len(xs_pitch)} (from {len(melodies)} melodies)")
    if tonic_map is not None:
        print(f"Tonic info: {n_with_tonic} melodies with key, {n_without_tonic} using first-pitch fallback")

    return (
        np.array(xs_pitch),
        np.array(xs_cpcint),
        np.array(xs_cpintfip),
        np.array(xs_cpintfref),
        np.array(ys),
        vocab_size,
        pitch_to_idx,
        idx_to_pitch
    )


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


# ====================================================
# MODEL COMPONENTS
# ====================================================
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
        """Create a causal sliding window attention mask"""
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
    window_size=10,
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
    cpcint_input = keras.Input(shape=(window_size,), name="cpcint")
    cpintfip_input = keras.Input(shape=(window_size,), name="cpintfip")
    cpintfref_input = keras.Input(shape=(window_size,), name="cpintfref")

    # Pitch embedding with positional encoding
    pitch_emb = TokenAndPositionEmbedding(window_size, vocab_size, embed_dim)(pitch_input)

    # Viewpoint embeddings (no positional encoding)
    cpcint_emb = layers.Embedding(cpcint_vocab_size, embed_dim)(cpcint_input)
    cpintfip_emb = layers.Embedding(interval_vocab_size, embed_dim)(cpintfip_input)
    cpintfref_emb = layers.Embedding(interval_vocab_size, embed_dim)(cpintfref_input)

    # Combine all streams
    x = layers.Add()([pitch_emb, cpcint_emb, cpintfip_emb, cpintfref_emb])

    for _ in range(num_layers):
        x = SlidingWindowTransformerBlock(
            embed_dim, num_heads, ff_dim, window_size, dropout_rate
        )(x)

    x = x[:, -1, :]
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(
        inputs=[pitch_input, cpcint_input, cpintfip_input, cpintfref_input],
        outputs=outputs,
    )
    return model


# ====================================================
# IC COMPUTATION
# ====================================================
def compute_ic(
        model,
        xs_pitch,
        xs_cpcint,
        xs_cpintfip,
        xs_cpintfref,
        ys
):
    """
    Compute Information Content (IC) in bits for each note prediction.
    Uses batched prediction to avoid GPU memory issues.
    """
    inputs = [xs_pitch, xs_cpcint, xs_cpintfip, xs_cpintfref]

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


def compute_per_melody_ic(
        model,
        melodies,
        window_size,
        pitch_to_idx,
        interval_offset,
        interval_vocab_size,
        tonic_map=None,
        melody_ids=None
):
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
        mel_xs_cpcint = []
        mel_xs_cpintfip = []
        mel_xs_cpintfref = []
        mel_ys = []

        for t in range(1, len(indexed)):
            start = max(0, t - window_size)
            context = indexed[start:t]
            raw_context = raw[start:t]

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
            mel_xs_cpcint.append(cpcint_seq)
            mel_xs_cpintfip.append(cpintfip_seq)
            mel_xs_cpintfref.append(cpintfref_seq)
            mel_ys.append(indexed[t])

        mel_xs_pitch = np.array(mel_xs_pitch)
        mel_xs_cpcint = np.array(mel_xs_cpcint)
        mel_xs_cpintfip = np.array(mel_xs_cpintfip)
        mel_xs_cpintfref = np.array(mel_xs_cpintfref)
        mel_ys = np.array(mel_ys)

        probs = model(
            [mel_xs_pitch, mel_xs_cpcint,
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


def compute_per_note_ic(
        model,
        melodies,
        melody_names,
        window_size,
        pitch_to_idx,
        idx_to_pitch,
        interval_offset,
        interval_vocab_size,
        tonic_map=None,
        melody_ids=None
):
    """
    Compute IC at every note position for each melody.
    Returns DataFrame with columns: melody, note, pitch, ic
    """
    PAD = 0
    rows = []

    for name in melody_names:
        mel = melodies[name]

        indexed, raw, valid_pos = [], [], []
        for pos, p in enumerate(mel):
            if int(p) in pitch_to_idx:
                indexed.append(pitch_to_idx[int(p)])
                raw.append(p)
                valid_pos.append(pos)
            else:
                print(f"Warning: pitch {p} in {name} not in vocabulary, skipping")

        if len(indexed) < 2:
            print(f"Warning: {name} has fewer than 2 valid pitches, skipping")
            continue

        first_pitch = raw[0]

        # Get tonic
        tonic_midi = first_pitch
        if tonic_map is not None and melody_ids is not None:
            tonic_pc = tonic_map.get(name)
            if tonic_pc is not None:
                tonic_midi = tonic_pc_to_midi(tonic_pc, first_pitch)

        mel_xs_pitch, mel_xs_cpcint = [], []
        mel_xs_cpintfip, mel_xs_cpintfref = [], []
        mel_ys = []
        note_positions = []

        for t in range(1, len(indexed)):
            start = max(0, t - window_size)
            context = indexed[start:t]
            raw_context = raw[start:t]

            raw_cpcint = [0]
            for i in range(1, len(raw_context)):
                raw_cpcint.append((raw_context[i] - raw_context[i - 1]) % 12)

            raw_cpintfip = [p - first_pitch for p in raw_context]
            raw_cpintfref = [p - tonic_midi for p in raw_context]

            pad_len = window_size - len(context)
            pitch_seq = [PAD] * pad_len + context

            cpcint_seq = [PAD] * pad_len
            for iv in raw_cpcint:
                cpcint_seq.append(iv + 1)

            cpintfip_seq = [PAD] * pad_len
            for iv in raw_cpintfip:
                s = max(1, min(iv + interval_offset, interval_vocab_size - 1))
                cpintfip_seq.append(s)

            cpintfref_seq = [PAD] * pad_len
            for iv in raw_cpintfref:
                s = max(1, min(iv + interval_offset, interval_vocab_size - 1))
                cpintfref_seq.append(s)

            mel_xs_pitch.append(pitch_seq)
            mel_xs_cpcint.append(cpcint_seq)
            mel_xs_cpintfip.append(cpintfip_seq)
            mel_xs_cpintfref.append(cpintfref_seq)
            mel_ys.append(indexed[t])
            note_positions.append(valid_pos[t])

        inputs = [np.array(x) for x in [
            mel_xs_pitch,
            mel_xs_cpcint,
            mel_xs_cpintfip,
            mel_xs_cpintfref
        ]]
        mel_ys = np.array(mel_ys)

        probs = model(inputs, training=False).numpy()

        for i in range(len(mel_xs_pitch)):
            p = max(probs[i][mel_ys[i]], 1e-10)

            predicted_token = int(np.argmax(probs[i]))
            predicted_pitch = idx_to_pitch[predicted_token]
            
            rows.append({
                "melody": name,
                "note": note_positions[i],
                "pitch": mel[note_positions[i]],
                "ic": -np.log2(p),
                "predicted_pitch": predicted_pitch,
                "correct_pitch": predicted_pitch == mel[note_positions[i]], 
            })

    return pd.DataFrame(rows)


# ====================================================
# SAVING HELPER
# ====================================================
def save_model_and_vocab(
    model,
    save_dir,
    pitch_to_idx,
    idx_to_pitch,
    vocab_size,
    window_size,
    interval_offset,
    interval_vocab_size,
    filename="model.keras",
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
):
    """
    Save model and vocabulary to a directory.
    New function, we need to save extra information
    (e.g. interval_offset, interval_vocab_size)
    """
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, filename)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    vocab_name = filename.replace(".keras", "_vocab.json")
    vocab_path = os.path.join(save_dir, vocab_name)
    with open(vocab_path, "w") as f:
        json.dump({
            "pitch_to_idx": pitch_to_idx,
            "idx_to_pitch": {str(k): v for k, v in idx_to_pitch.items()},
            "vocab_size": vocab_size,
            "window_size": window_size,
            "interval_offset": interval_offset,
            "interval_vocab_size": interval_vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
        }, f, indent=2)
    print(f"Vocabulary saved to {vocab_path}")

# ====================================================
# TRAINING
# ====================================================
def train_model(
    melodies,
    melody_ids=None,
    tonic_map=None,
    window_size=10,
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
    Full-corpus training for the viewpoints transformer.
    For proper evaluation, use run_kfold.
    """
    interval_offset, interval_vocab_size = get_fixed_interval_params()

    xs_pitch, xs_cpcint, xs_cpintfip, xs_cpintfref, ys, \
        vocab_size, pitch_to_idx, idx_to_pitch = \
        prepare_melodies_sliding(
            melodies, window_size=window_size,
            interval_offset=interval_offset, interval_vocab_size=interval_vocab_size,
            tonic_map=tonic_map, melody_ids=melody_ids,
        )

    print("Data prepared:")
    print(f"Melodies: {len(melodies)}")
    print(f"Vocab size: {vocab_size} ({vocab_size - 1} pitches + PAD)")
    print(f"Interval vocab size: {interval_vocab_size}")
    print(f"Window size: {window_size}")
    print(f"Training examples: {xs_pitch.shape[0]}")

    model = build_transformer(
        vocab_size=vocab_size, interval_vocab_size=interval_vocab_size,
        window_size=window_size, embed_dim=embed_dim,
        num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers,
        dropout_rate=dropout_rate,
    )

    print(f"\n  Total parameters: {model.count_params():,}")
    model.summary()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    )

    n = len(ys)
    n_val = int(n * validation_split)
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    all_xs = [xs_pitch, xs_cpcint, xs_cpintfip, xs_cpintfref]
    train_inputs = [x[train_idx] for x in all_xs]
    val_inputs = [x[val_idx] for x in all_xs]

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

    history = model.fit(
        train_inputs, ys[train_idx],
        validation_data=(val_inputs, ys[val_idx]),
        batch_size=batch_size, epochs=epochs,
        callbacks=callbacks,
    )

    model.load_weights(checkpoint_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if save_models_dir is not None:
        save_model_and_vocab(
            model, save_models_dir, pitch_to_idx, idx_to_pitch,
            vocab_size, window_size, interval_offset, interval_vocab_size,
            embed_dim=embed_dim, num_heads=num_heads,
            ff_dim=ff_dim, num_layers=num_layers,
        )

    return model, history, {
        "vocab_size": vocab_size,
        "interval_vocab_size": interval_vocab_size,
        "interval_offset": interval_offset,
        "pitch_to_idx": pitch_to_idx,
        "idx_to_pitch": idx_to_pitch,
        "window_size": window_size,
    }


def run_kfold(
    melodies,
    melody_ids=None,
    tonic_map=None,
    k=10,
    window_size=10,
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
        xs_pitch_train, xs_cpcint_train, xs_cpintfip_train, xs_cpintfref_train, \
            ys_train, vocab_size, pitch_to_idx, idx_to_pitch = \
            prepare_melodies_sliding(
                train_melodies, window_size=window_size,
                interval_offset=interval_offset, interval_vocab_size=interval_vocab_size,
                tonic_map=tonic_map, melody_ids=train_ids,
            )

        # Test data: reuse training pitch vocab
        xs_pitch_test, xs_cpcint_test, xs_cpintfip_test, xs_cpintfref_test, \
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
            vocab_size=vocab_size,
            interval_vocab_size=interval_vocab_size,
            window_size=window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
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

        all_train_xs = [xs_pitch_train, xs_cpcint_train,
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

        # Save fold model
        if save_models_dir is not None:
            save_model_and_vocab(
                model, save_models_dir, pitch_to_idx, idx_to_pitch,
                vocab_size, window_size, interval_offset, interval_vocab_size,
                filename=f"fold_{fold + 1}.keras",
                embed_dim=embed_dim, num_heads=num_heads,
                ff_dim=ff_dim, num_layers=num_layers,
            )

        # Evaluate on test fold
        mean_ic, ics = compute_ic(
            model, xs_pitch_test, xs_cpcint_test,
            xs_cpintfip_test, xs_cpintfref_test, ys_test,
        )
        melody_ics = compute_per_melody_ic(
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


def run_hymn_ic(
    model_dir,
    hymn_lisp_path=None,
):
    """
    Compute per-note and per-melody IC for hymn melodies
    using a pre-trained viewpoints model.
    """
    if hymn_lisp_path is None:
        hymn_lisp_path = ROOT / "data" / "hymns.lisp"
    model_path = os.path.join(model_dir, "model.keras")
    vocab_path = os.path.join(model_dir, "model_vocab.json")

    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        print("Run 'full_essen' experiment first.")
        exit(1)

    if not os.path.exists(vocab_path):
        print(f"Error: vocab not found at {vocab_path}")
        exit(1)

    if not os.path.exists(hymn_lisp_path):
        print(f"Error: hymn lisp file not found at {hymn_lisp_path}")
        exit(1)

    # Load vocab and config
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
    pitch_to_idx = {int(k): v for k, v in vocab_data["pitch_to_idx"].items()}
    idx_to_pitch = {int(k): v for k, v in vocab_data["idx_to_pitch"].items()}
    vocab_size = vocab_data["vocab_size"]
    interval_vocab_size = vocab_data["interval_vocab_size"]
    interval_offset = vocab_data["interval_offset"]
    window_size = vocab_data["window_size"]

    # Rebuild and load model (use saved hyperparams, or training defaults)
    embed_dim = vocab_data.get("embed_dim", 32)
    num_heads = vocab_data.get("num_heads", 4)
    ff_dim = vocab_data.get("ff_dim", 64)
    num_layers = vocab_data.get("num_layers", 2)

    model = build_transformer(
        vocab_size=vocab_size,
        interval_vocab_size=interval_vocab_size,
        window_size=window_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
    )
    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Vocab size: {vocab_size}, Window size: {window_size}")

    # Load hymn melodies
    hymn_melodies = parse_lisp_melodies(hymn_lisp_path)
    melody_names = sorted(hymn_melodies.keys())
    print(f"Loaded {len(hymn_melodies)} hymn melodies")

    # out of vocab check
    oov = set(p for mel in hymn_melodies.values() for p in mel) - set(pitch_to_idx.keys())
    if oov:
        print(f"Warning: {len(oov)} OOV pitches in hymns: {sorted(oov)}")

    # Per-note IC (pass raw melodies dict)
    per_note_df = compute_per_note_ic(
        model, hymn_melodies, melody_names, window_size,
        pitch_to_idx, idx_to_pitch, interval_offset, interval_vocab_size,
    )
    per_note_path = "hymn_viewpoints_per_note_ic.csv"
    per_note_df.to_csv(per_note_path, index=False)
    print(f"Per-note IC saved to: {per_note_path} ({len(per_note_df)} notes)")

    # Per-melody IC
    hymn_melodies_list = [hymn_melodies[name] for name in melody_names]
    melody_ics = compute_per_melody_ic(
        model, hymn_melodies_list, window_size, pitch_to_idx,
        interval_offset, interval_vocab_size,
    )
    for i, mic in enumerate(melody_ics):
        mic["melody_id"] = melody_names[i]

    melody_ic_df = pd.DataFrame(melody_ics)
    melody_ic_path = "hymn_viewpoints_per_melody_ic.csv"
    melody_ic_df.to_csv(melody_ic_path, index=False)
    print(f"Per-melody IC saved to: {melody_ic_path} ({len(melody_ic_df)} melodies)")

    print(f"\nMean IC: {per_note_df['ic'].mean():.3f} bits")
    print("\nPer-melody summary:")
    print(per_note_df.groupby("melody")["ic"].agg(["mean", "std", "count"]))

    return {"per_note_ic": per_note_df, "melody_ics": melody_ic_df}


# ====================================================
# DATA LOADING
# ====================================================
def load_corpus(name):
    df = pd.read_csv(ROOT / "data" / f"{name}_unique_melodies.csv")
    melodies = df["pitch"].apply(ast.literal_eval).tolist()
    melody_ids = df["melody_id"].tolist()
    return melodies, melody_ids


def load_meta(name):
    return pd.read_csv(ROOT / "data" / f"{name}_unique_meta_melodies.csv")


# ====================================================
# CLI
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Viewpoints transformer experiments"
    )
    parser.add_argument(
        "experiment",
        choices=[
            "kfold_essen",
            "kfold_meertens",
            "full_essen",
            "full_meertens",
            "hymn_ic",
        ]
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--save-models-dir", type=str, default=None)
    parser.add_argument("--hymn-lisp", type=str, default=None)
    parser.add_argument("--trained-model-dir", type=str, default=None,
                        help="Directory with a trained model to load (for hymn_ic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    if args.experiment == "kfold_essen":
        melodies, ids = load_corpus("essen")
        run_kfold(
            melodies,
            melody_ids=ids,
            k=10,
            window_size=args.window_size,
            epochs=args.epochs,
            export_folds_path=f"essen_{args.window_size}_folds_viewpoint.csv",
            save_models_dir=(
                args.save_models_dir or "../pretrained_models/viewpoints/kfold_viewpoints_essen"
            ),
        )

    # ------------------------------------------------------------------
    elif args.experiment == "kfold_meertens":
        melodies, ids = load_corpus("meertens")
        run_kfold(
            melodies,
            melody_ids=ids,
            k=10,
            window_size=args.window_size,
            epochs=args.epochs,
            patience=args.patience,
            export_folds_path=f"meertens_{args.window_size}_folds_viewpoints.csv",
            save_models_dir=(
                args.save_models_dir or "../pretrained_models/viewpoints/kfold_viewpoints_meertens"
            ),
        )

    # ------------------------------------------------------------------
    elif args.experiment == "full_essen":
        melodies, ids = load_corpus("essen")
        model, history, data_info = train_model(
            melodies,
            melody_ids=ids,
            window_size=args.window_size,
            epochs=args.epochs,
            patience=args.patience,
            save_models_dir=args.save_models_dir or "../pretrained_models/viewpoints/full_viewpoints_essen",
        )

    # ------------------------------------------------------------------
    elif args.experiment == "hymn_ic":
        run_hymn_ic(
            model_dir=args.trained_model_dir or "../pretrained_models/viewpoints/full_viewpoints_essen",
            hymn_lisp_path=args.hymn_lisp or "../data/hymns.lisp",
        )
