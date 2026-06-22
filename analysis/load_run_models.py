import pandas as pd
import numpy as np

import argparse
import json
import ast
import os
import re

import keras
from keras import layers, ops

# ======================================================= 
# MODEL COMPONENTS (full-window & sliding-window)
# ======================================================= 

class TokenAndPositionEmbedding(layers.Layer):
  """Embeds tokens and adds learned positional encoding."""

  def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.max_len = max_len
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
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

  def get_config(self):
    config = super().get_config()
    config.update({
        "max_len": self.max_len,
        "vocab_size": self.vocab_size,
        "embed_dim": self.embed_dim
      })
    return config


class CausalTransformerBlock(layers.Layer):
  """
  Transformer block with CAUSAL attention mask.

  The causal mask ensures position i can only attend to positions <= i,
  matching IDyOM's left-to-right prediction constraint.
  """
  def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.ff_dim = ff_dim
    self.dropout_rate = dropout_rate
    
    self.att = layers.MultiHeadAttention(
      num_heads=num_heads, key_dim=embed_dim // num_heads,
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
    attn_output = self.att(
      inputs, inputs, use_causal_mask=True
    )
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output)

    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    return self.layernorm2(out1 + ffn_output)

  def get_config(self):
    config = super().get_config()
    config.update({
      "embed_dim": self.embed_dim,
      "num_heads": self.num_heads,
      "ff_dim": self.ff_dim,
      "dropout_rate": self.dropout_rate,
    })
    return config


class SlidingWindowTransformerBlock(layers.Layer):
  """
  Transformer block where position i can only attend to
  positions [i-window_size+1, ..., i].

  This matches IDyOM's order bound: at each prediction step,
  only the previous `window_size` notes are visible.
  """

  def __init__(self, embed_dim, num_heads, ff_dim, window_size, dropout_rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.ff_dim = ff_dim
    self.dropout_rate = dropout_rate
    
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
    """
    Create a causal sliding window attention mask.
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

  def get_config(self):
    config = super().get_config()
    config.update({
      "embed_dim": self.embed_dim,
      "num_heads": self.num_heads,
      "ff_dim": self.ff_dim,
      "window_size": self.window_size,
      "dropout_rate": self.dropout_rate,
    })
    return config


# =======================================================
# VIEWPOINTS MODEL COMPONENTS
# =======================================================

class ViewpointsTokenAndPositionEmbedding(layers.Layer):
    """Embeds tokens and adds learned positional encoding (viewpoints version)."""

    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
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


class ViewpointsSlidingWindowTransformerBlock(layers.Layer):
    """
    Transformer block with sliding window mask (viewpoints version).
    """

    def __init__(self, embed_dim, num_heads, ff_dim, window_size, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.window_size = window_size
        self.dropout_rate = dropout_rate
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


def build_viewpoints_transformer(
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
    Rebuild the viewpoints transformer architecture from hyperparameters.
    (cpint (cpcint cpintfip) (cpitch cpintfref))
    """
    pitch_input = keras.Input(shape=(window_size,), name="pitch")
    cpcint_input = keras.Input(shape=(window_size,), name="cpcint")
    cpintfip_input = keras.Input(shape=(window_size,), name="cpintfip")
    cpintfref_input = keras.Input(shape=(window_size,), name="cpintfref")

    # Pitch embedding with positional encoding
    pitch_emb = ViewpointsTokenAndPositionEmbedding(
        window_size, vocab_size, embed_dim
    )(pitch_input)

    # Viewpoint embeddings (no positional encoding)
    cpcint_emb = layers.Embedding(cpcint_vocab_size, embed_dim)(cpcint_input)
    cpintfip_emb = layers.Embedding(interval_vocab_size, embed_dim)(cpintfip_input)
    cpintfref_emb = layers.Embedding(interval_vocab_size, embed_dim)(cpintfref_input)

    # Combine all streams via element-wise addition
    x = layers.Add()([pitch_emb, cpcint_emb, cpintfip_emb, cpintfref_emb])

    for _ in range(num_layers):
        x = ViewpointsSlidingWindowTransformerBlock(
            embed_dim, num_heads, ff_dim, window_size, dropout_rate
        )(x)

    x = x[:, -1, :]
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(
        inputs=[pitch_input, cpcint_input, cpintfip_input, cpintfref_input],
        outputs=outputs,
    )
    return model


def tonic_pc_to_midi(tonic_pc, first_pitch):
    """
    Convert tonic pitch class to the nearest MIDI pitch to the melody's
    first note. Ensures cpintfref values are small intervals rather than
    spanning multiple octaves.
    """
    octave = first_pitch // 12
    candidate = octave * 12 + tonic_pc
    options = [candidate - 12, candidate, candidate + 12]
    return min(options, key=lambda x: abs(x - first_pitch))


# ======================================================= 
# LOADING 
# =======================================================

def load_saved_model(model_dir, model_type, fold=None):
    """
    Load a saved model and its vocabulary.
    
    Args:
        model_dir:   Path to directory containing .keras and vocab.json files.
        model_type:  One of 'full', 'sliding', 'viewpoints'.
        fold:        Optional fold number for k-fold models.
    """
    if fold is not None:
        model_path = os.path.join(model_dir, f"fold_{fold}.keras")
        vocab_path = os.path.join(model_dir, f"fold_{fold}_vocab.json")
    else:
        model_path = os.path.join(model_dir, "model.keras")
        vocab_path = os.path.join(model_dir, "vocab.json")
 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
 
    # Load vocab
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
      
    # Convert keys back to ints
    vocab_data["pitch_to_idx"] = {
        int(k): v for k, v in vocab_data["pitch_to_idx"].items()
    }
    vocab_data["idx_to_pitch"] = {
        int(k): int(v) for k, v in vocab_data["idx_to_pitch"].items()
    }
 
    vocab_data["model_type"] = model_type
 
    if model_type == "viewpoints":
        # Viewpoints model
        model = build_viewpoints_transformer(
            vocab_size=vocab_data["vocab_size"],
            interval_vocab_size=vocab_data["interval_vocab_size"],
            window_size=vocab_data["window_size"],
            embed_dim=vocab_data.get("embed_dim", 32),
            num_heads=vocab_data.get("num_heads", 4),
            ff_dim=vocab_data.get("ff_dim", 64),
            num_layers=vocab_data.get("num_layers", 2),
        )
        model.load_weights(model_path)
    else:
        # Full-window and sliding-window: use keras deserialization
        model = keras.models.load_model(model_path, custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "SlidingWindowTransformerBlock": SlidingWindowTransformerBlock,
            "CausalTransformerBlock": CausalTransformerBlock,
        }, compile=False)
 
    print(f"Loaded {model_type} model from {model_path}")
    print(f"Vocab size: {vocab_data['vocab_size']}")
    if "window_size" in vocab_data:
        print(f"Window size: {vocab_data['window_size']}")
    if "max_len" in vocab_data:
        print(f"Max length: {vocab_data['max_len']}")
    if "interval_vocab_size" in vocab_data:
        print(f"Interval vocab size: {vocab_data['interval_vocab_size']}")
        print(f"Interval offset: {vocab_data['interval_offset']}")
    print(f"Parameters: {model.count_params():,}")
 
    return model, vocab_data
 
 
# ======================================================= 
# DATA PARSER AND LOADING
# ======================================================= 
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
 
def parse_csv_melodies(filepath):
  """Load melodies from CSV format"""
  df = pd.read_csv(filepath)
  melodies = {}
  for _, row in df.iterrows():
    name = str(row["melody_id"])
    pitches = ast.literal_eval(row["pitch"])
    melodies[name] = pitches
  return melodies
 
 
def load_melodies(filepath):
  """Auto-detect format and load melodies from either CSV or lisp file."""
  if filepath.endswith(".csv"):
    return parse_csv_melodies(filepath)
  elif filepath.endswith(".lisp"):
    return parse_lisp_melodies(filepath)
  else:
    raise ValueError(f"Unknown file format: {filepath}. Expected .csv or .lisp")
 
 
def load_tonic_map(filepath):
    """
    Load a tonic map CSV with columns: melody_id, tonic_pc
    where tonic_pc is the pitch class (0-11) of the tonic.
    Returns a dict mapping melody_id (str) -> tonic_pc (int).
    """
    df = pd.read_csv(filepath)
    tonic_map = {}
    for _, row in df.iterrows():
        tonic_map[str(row["melody_id"])] = int(row["tonic_pc"])
    return tonic_map
 
 
# =======================================================
# IC PER NOTE COMPUTATION
# =======================================================
def compute_per_note_ic_full_window(
    model,
    melodies,
    melody_names,
    max_len,
    pitch_to_idx,
    idx_to_pitch
):
  """Per-note IC for the full-window transformer."""
  PAD = 0
  rows = []
 
  for name in melody_names:
    mel = melodies[name]
    indexed, valid_pos = [], []
    for pos, p in enumerate(mel):
      if p in pitch_to_idx:
        indexed.append(pitch_to_idx[p])
        valid_pos.append(pos)
 
    if len(indexed) < 2:
      continue
 
    inp = indexed[:-1]
    tgt = indexed[1:]
 
    if len(inp) >= max_len:
      inp = inp[-max_len:]
      tgt = tgt[-max_len:]
      offset = len(indexed) - 1 - max_len
      note_positions = valid_pos[offset + 1:]
      pad_len = 0
    else:
      pad_len = max_len - len(inp)
      inp = [PAD] * pad_len + inp
      tgt = [PAD] * pad_len + tgt
      note_positions = valid_pos[1:]
 
    probs = model.predict(np.array([inp]), verbose=0)
 
    for i in range(pad_len, max_len):
      p = max(probs[0][i][tgt[i]], 1e-10)

      # Predicted pitch
      predicted_token = int(np.argmax(probs[0][i]))
      predicted_pitch = idx_to_pitch[predicted_token]
      
      rows.append({
        "melody": name,
        "note": note_positions[i - pad_len],
        "pitch": mel[note_positions[i - pad_len]],
        "ic": -np.log2(p),
        "predicted_pitch": predicted_pitch,
        "correct_pitch": predicted_pitch == mel[note_positions[i - pad_len]],
      })
 
  return pd.DataFrame(rows)
 
 
def compute_per_note_ic_sliding(
    model,
    melodies,
    melody_names,
    window_size,
    pitch_to_idx,
    idx_to_pitch
):
  """Per-note IC for the sliding-window transformer."""
  PAD = 0
  rows = []
 
  for name in melody_names:
    mel = melodies[name]
    indexed, valid_pos = [], []
    for pos, p in enumerate(mel):
      if p in pitch_to_idx:
        indexed.append(pitch_to_idx[p])
        valid_pos.append(pos)
 
    if len(indexed) < 2:
      continue
 
    mel_xs, mel_ys, note_positions = [], [], []
    for t in range(1, len(indexed)):
      start = max(0, t - window_size)
      context = indexed[start:t]
      pad_len = window_size - len(context)
      mel_xs.append([PAD] * pad_len + context)
      mel_ys.append(indexed[t])
      note_positions.append(valid_pos[t])
 
    probs = model.predict(np.array(mel_xs), verbose=0)
 
    for i in range(len(mel_xs)):
      p = max(probs[i][mel_ys[i]], 1e-10)
      # Predicted pitch
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
 
 
def compute_per_note_ic_viewpoints(
    model,
    melodies,
    melody_names,
    window_size,
    pitch_to_idx,
    idx_to_pitch,
    interval_offset,
    interval_vocab_size,
    tonic_map=None,
):
    """
    Per-note IC for the viewpoints transformer.
    Derives cpcint, cpintfip, and cpintfref from raw pitch sequences
    and feeds all four input streams to the multi-input model.
    """
    PAD = 0
    rows = []
 
    for name in melody_names:
        mel = melodies[name]
 
        indexed, raw, valid_pos = [], [], []
        for pos, p in enumerate(mel):
            if int(p) in pitch_to_idx:
                indexed.append(pitch_to_idx[int(p)])
                raw.append(int(p))
                valid_pos.append(pos)
 
        if len(indexed) < 2:
            continue
 
        first_pitch = raw[0]
 
        # Get tonic for cpintfref (fall back to first pitch if unavailable)
        tonic_midi = first_pitch
        if tonic_map is not None:
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
 
            # cpcint: pitch-class interval (successive intervals mod 12)
            raw_cpcint = [0]
            for i in range(1, len(raw_context)):
                raw_cpcint.append((raw_context[i] - raw_context[i - 1]) % 12)
 
            # cpintfip: interval from first pitch of piece
            raw_cpintfip = [p - first_pitch for p in raw_context]
 
            # cpintfref: interval from tonal referent (tonic)
            raw_cpintfref = [p - tonic_midi for p in raw_context]
 
            # Left-pad to window_size
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
            note_positions.append(valid_pos[t])
 
        # Build input arrays
        inputs = [
            np.array(mel_xs_pitch),
            np.array(mel_xs_cpcint),
            np.array(mel_xs_cpintfip),
            np.array(mel_xs_cpintfref),
        ]
        mel_ys = np.array(mel_ys)
 
        # Predict (use model() to avoid retracing overhead per melody)
        probs = model(inputs, training=False).numpy()
 
        for i in range(len(mel_xs_pitch)):
            p = max(probs[i][mel_ys[i]], 1e-10)

            # Predicted pitch
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
 
 
# =========================================================================
# CLI
# =========================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load saved models and compute IC on new melodies"
    )
    parser.add_argument("model_dir",
                        help="Path to saved model directory")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["full", "sliding", "viewpoints"],
                        help="Type of transformer model to load")
    parser.add_argument("melody_file", nargs="?", default=None,
                        help="Path to melodies (.csv or .lisp)")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to meta CSV with melody_id and filename columns")
    parser.add_argument("--tonic-map", type=str, default=None,
                        help="Path to CSV with melody_id and tonic_pc columns "
                             "(for viewpoints model cpintfref computation)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Which fold to load (for k-fold models)")
    parser.add_argument("--output", type=str, default="per_note_ic.csv",
                        help="Output CSV path")
    parser.add_argument("--info", action="store_true",
                        help="Just print model info, don't run inference")
    args = parser.parse_args()
 
    # Load model
    model, vocab_data = load_saved_model(
        args.model_dir, args.model_type, fold=args.fold
    )
    model_type = args.model_type
 
    if args.info:
        print(f"\nModel type: {model_type}")
        print(f"\nVocab mapping:")
        pitch_to_idx = vocab_data["pitch_to_idx"]
        print(f"{len(pitch_to_idx)} total pitches")
        for pitch, idx in sorted(pitch_to_idx.items()):
            print(f"MIDI {pitch} -> index {idx}")
        if model_type == "viewpoints":
            print(f"\nInterval offset: {vocab_data['interval_offset']}")
            print(f"Interval vocab size: {vocab_data['interval_vocab_size']}")
        exit(0)
 
    if args.melody_file is None:
        print("No melody file provided. Use --info to inspect the model,")
        print("or provide a .csv or .lisp file to compute IC.")
        exit(1)
 
    # Load melodies
    melodies = load_melodies(args.melody_file)
    print(f"\nLoaded {len(melodies)} melodies from {args.melody_file}")
 
    # Load metadata for melody names
    name_map = None
    if args.meta:
        meta_df = pd.read_csv(args.meta)
        name_map = dict(zip(
            meta_df["melody_id"].astype(str),
            meta_df["filename"].apply(
                lambda x: x.split("/")[-1].replace(".krn", "") if isinstance(x, str) else x
            )
        ))
        print(f"Loaded metadata for {len(name_map)} melodies")
 
    # Load tonic map (for viewpoints model)
    tonic_map = None
    if args.tonic_map:
        tonic_map = load_tonic_map(args.tonic_map)
        print(f"Loaded tonic map for {len(tonic_map)} melodies")
    elif model_type == "viewpoints":
        print("Note: no --tonic-map provided; using first pitch as tonic fallback")
 
    # Check out-of-vocabulary pitches
    pitch_to_idx = vocab_data["pitch_to_idx"]
    idx_to_pitch = vocab_data["idx_to_pitch"]
    
    train_vocab = set(pitch_to_idx.keys())
    melody_vocab = set(p for mel in melodies.values() for p in mel)
    oov = melody_vocab - train_vocab
    if oov:
        print(f"Warning: {len(oov)} out-of-vocab pitches: {sorted(oov)}")
 
    # Compute IC based on model type
    melody_names = sorted(melodies.keys())
 
    if model_type == "viewpoints":
        print(f"Running viewpoints model (window={vocab_data['window_size']})")
        ic_df = compute_per_note_ic_viewpoints(
            model, melodies, melody_names,
            vocab_data["window_size"],
            pitch_to_idx, idx_to_pitch,
            vocab_data["interval_offset"],
            vocab_data["interval_vocab_size"],
            tonic_map=tonic_map,
        )
    elif model_type == "sliding":
        print(f"Running sliding-window model (window={vocab_data['window_size']})")
        ic_df = compute_per_note_ic_sliding(
            model, melodies, melody_names,
            vocab_data["window_size"], pitch_to_idx, idx_to_pitch,
        )
    elif model_type == "full":
        print(f"Running full-window model (max_len={vocab_data['max_len']})")
        ic_df = compute_per_note_ic_full_window(
            model, melodies, melody_names,
            vocab_data["max_len"], pitch_to_idx, idx_to_pitch,
        )
 
    # Add melody names from metadata
    if name_map is not None:
        ic_df["melody_name"] = ic_df["melody"].astype(str).map(name_map)
        unnamed = ic_df["melody_name"].isna().sum()
        if unnamed > 0:
            print(f"Warning: {unnamed} notes without melody name mapping")
 
    # Save and summarise
    ic_df.to_csv(args.output, index=False)
    print(f"\nPer-note IC saved to {args.output}")
    print(f"Total notes: {len(ic_df)}")
    print(f"Mean IC: {ic_df['ic'].mean():.3f} bits")
    print(f"\nPer-melody summary:")
    group_col = "melody_name" if name_map is not None else "melody"
    print(ic_df.groupby(group_col)["ic"].agg(["mean", "std", "count"]).to_string())
