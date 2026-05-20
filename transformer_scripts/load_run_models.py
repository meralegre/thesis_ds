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
# MODEL COMPONENTS
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
      # dropout=dropout_rate
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
# LOADING 
# =======================================================
def load_saved_model(model_dir, fold=None):
  """
  Load a saved model and its vocabulary.
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

  # Load model with custom objects
  model = keras.models.load_model(model_path, custom_objects={
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    "SlidingWindowTransformerBlock": SlidingWindowTransformerBlock,
    "CausalTransformerBlock": CausalTransformerBlock,
  }, compile=False)

  print(f"Loaded model from {model_path}")
  print(f"Vocab size: {vocab_data['vocab_size']}")
  if "window_size" in vocab_data:
    print(f"Window size: {vocab_data['window_size']}")
  if "max_len" in vocab_data:
    print(f"Max length: {vocab_data['max_len']}")
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


# =======================================================
# IC PER NOTE COMPUTATION
# =======================================================
def compute_per_note_ic_full_window(
    model,
    melodies,
    melody_names,
    max_len,
    pitch_to_idx
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
      rows.append({
        "melody": name,
        "note": note_positions[i - pad_len],
        "pitch": mel[note_positions[i - pad_len]],
        "ic": -np.log2(p),
      })

  return pd.DataFrame(rows)


def compute_per_note_ic_sliding(
    model,
    melodies,
    melody_names,
    window_size,
    pitch_to_idx
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
      rows.append({
        "melody": name,
        "note": note_positions[i],
        "pitch": mel[note_positions[i]],
        "ic": -np.log2(p),
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
    parser.add_argument("melody_file", nargs="?", default=None,
                        help="Path to melodies (.csv or .lisp)")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to meta CSV with melody_id and filename columns")
    parser.add_argument("--fold", type=int, default=None,
                        help="Which fold to load (for k-fold models)")
    parser.add_argument("--output", type=str, default="per_note_ic.csv",
                        help="Output CSV path")
    parser.add_argument("--info", action="store_true",
                        help="Just print model info, don't run inference")
    args = parser.parse_args()

    # Load model
    model, vocab_data = load_saved_model(args.model_dir, fold=args.fold)

    if args.info:
        print("\nVocab mapping:")
        pitch_to_idx = vocab_data["pitch_to_idx"]
        print(f"{len(pitch_to_idx)} total pitches")
        for pitch, idx in sorted(pitch_to_idx.items()):
            print(f"MIDI {pitch} -> index {idx}")
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

    # Check out-of-vocabulary pitches
    pitch_to_idx = vocab_data["pitch_to_idx"]
    train_vocab = set(pitch_to_idx.keys())
    melody_vocab = set(p for mel in melodies.values() for p in mel)
    oov = melody_vocab - train_vocab
    if oov:
        print(f"Warning: {len(oov)} out-of-vocab pitches: {sorted(oov)}")

    # Detect model type and compute IC
    melody_names = sorted(melodies.keys())

    if "window_size" in vocab_data:
        print(f"Detected sliding-window model (window={vocab_data['window_size']})")
        ic_df = compute_per_note_ic_sliding(
            model, melodies, melody_names,
            vocab_data["window_size"], pitch_to_idx,
        )
    elif "max_len" in vocab_data:
        print(f"Detected full-window model (max_len={vocab_data['max_len']})")
        ic_df = compute_per_note_ic_full_window(
            model, melodies, melody_names,
            vocab_data["max_len"], pitch_to_idx,
        )
    else:
        print("Error: vocab.json missing both window_size and max_len")
        exit(1)

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
