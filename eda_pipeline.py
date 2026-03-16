#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

# ============================================================================
# KERN SYNTAX / CHARACTER SETS
# ============================================================================

DURATION_CHARS   = set("0123456789.")
PITCH_CHARS      = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
ACCIDENTAL_CHARS = set("_-#n")
REST_CHAR        = "r"
PAUSE_CHAR       = ";"
GROUPING_CHARS   = set("{}()[]")

ORNAMENT_CHARS      = set("Mm$STtwWRO")
APPOGGIATURA_CHARS  = set("pP")
GRACE_CHAR          = "q"
GROUPETTO_CHAR      = "Q"
ARTICULATION_CHARS  = set("z'\"`^~:I")
BOWING_CHARS        = set("uv")
STEM_DIR_CHARS      = set("/\\")
BEAM_CHARS          = set("LJ")
PARTIAL_BEAM_CHARS  = set("kK")
EDITORIAL_CHARS     = set("xXyY?")
USER_MARK_CHARS     = set("iltNUVZ@%+")

NON_CORE_CHARS = (
    ORNAMENT_CHARS
    | APPOGGIATURA_CHARS
    | ARTICULATION_CHARS
    | BOWING_CHARS
    | STEM_DIR_CHARS
    | BEAM_CHARS
    | PARTIAL_BEAM_CHARS
    | EDITORIAL_CHARS
    | USER_MARK_CHARS
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_null_token(line: str) -> bool:
    return line.startswith(".")


def has_grace_or_groupetto(tok: str) -> bool:
    """Durationless ornaments – should not be counted as events."""
    return (GRACE_CHAR in tok) or (GROUPETTO_CHAR in tok)


def strip_non_core_signifiers(tok: str) -> str:
    """Remove ornaments, articulation, beams, editorial marks, etc."""
    return "".join(ch for ch in tok if ch not in NON_CORE_CHARS)


# ============================================================================
# TOKEN PARSER
# ============================================================================

@dataclass
class KernEvent:
    duration: str
    pitch: str
    is_rest: bool


def parse_kern_token(tok: str) -> Optional[KernEvent]:
    """Parse a single **kern token into a KernEvent (or None if it should be
    skipped, e.g. null tokens, grace notes, groupettos)."""
    tok = tok.strip()
    if not tok or is_null_token(tok):
        return None
    if has_grace_or_groupetto(tok):
        return None

    tok = strip_non_core_signifiers(tok)

    # strip grouping / phrasing markers that can wrap tokens
    while tok and tok[0] in GROUPING_CHARS:
        tok = tok[1:]
    while tok and tok[-1] in GROUPING_CHARS:
        tok = tok[:-1]
    if not tok or is_null_token(tok):
        return None

    # extract duration
    i = 0
    dur_chars: list[str] = []
    while i < len(tok) and tok[i].isdigit():
        dur_chars.append(tok[i])
        i += 1
    while i < len(tok) and tok[i] == ".":
        dur_chars.append(tok[i])
        i += 1
    duration = "".join(dur_chars)
    if not duration:
        return None

    pitch_part = tok[i:]
    if not pitch_part:
        return None

    is_rest = (REST_CHAR in pitch_part) or (PAUSE_CHAR in pitch_part)
    if is_rest:
        pitch_norm = "r"
    else:
        pitch_chars = [ch for ch in pitch_part if ch in PITCH_CHARS or ch in ACCIDENTAL_CHARS]
        if not pitch_chars:
            return None
        pitch_norm = "".join(pitch_chars)

    return KernEvent(duration=duration, pitch=pitch_norm, is_rest=is_rest)


# ============================================================================
# FILE LOADING
# ============================================================================

def load_kern_melody(path: pathlib.Path) -> List[KernEvent]:
    """Load a single **kern file and return its list of cleaned events.
    Assumes monophonic input (takes first spine only)."""
    out: list[KernEvent] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # skip comments, interpretations, barlines, null tokens
            if line.startswith(("*", "!", "=", ".")):
                continue

            tok = line.split("\t")[0].strip()
            ev = parse_kern_token(tok)
            if ev is not None:
                out.append(ev)
    return out


# ============================================================================
# CORPUS COLLECTION
# ============================================================================

def derive_labels(path: pathlib.Path, corpus_root: pathlib.Path,
                  label_depth: int = 0,
                  corpus_name: str = "corpus") -> dict:
    """Derive hierarchical labels from the relative path."""
    if label_depth == 0:
        return {"label_0": corpus_name}

    rel_parts = path.relative_to(corpus_root).parts
    labels: dict = {}
    for i in range(label_depth):
        labels[f"label_{i}"] = rel_parts[i] if i < len(rel_parts) else "unknown"
    return labels


def collect_corpus(corpus_root: pathlib.Path,
                   label_depth: int = 2,
                   min_events: int = 0,
                   file_voices: Optional[Dict[str, int]] = None,
                   corpus_name: str = "corpus",
                   ) -> Tuple[List[List[KernEvent]], List[dict]]:
    """Walk corpus_root, parse every krn file and return (melodies, meta)"""
    melodies: list[list[KernEvent]] = []
    meta_list: list[dict] = []

    for path in sorted(corpus_root.rglob("*.krn")):
        events = load_kern_melody(path)
        if len(events) < min_events:
            continue

        n_voices = (file_voices or {}).get(str(path), 1)
        labels = derive_labels(path, corpus_root, label_depth,
                               corpus_name=corpus_name)
        melodies.append(events)
        meta_list.append({
            "filename": path.name,
            "path": str(path),
            **labels,
            "length": len(events),
            "n_voices": n_voices,
            "is_monophonic": n_voices == 1,
        })

    return melodies, meta_list


# ============================================================================
# EVENTS DATAFRAME
# ============================================================================

def build_events_df(melodies: List[List[KernEvent]],
                    meta: List[dict]) -> pd.DataFrame:
    rows = []
    for i, (events, m) in enumerate(zip(melodies, meta)):
        for j, ev in enumerate(events):
            row = m.copy()
            row.update({
                "melody_id": i,
                "position": j,
                "pitch": ev.pitch,
                "duration": ev.duration,
            })
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# DUPLICATE REMOVAL
# ============================================================================

def find_exact_duplicates(df: pd.DataFrame) -> Tuple[List[int], List[dict]]:
    """Return (keep_ids, duplicate_records) based on exact (pitch, duration)
    signature matching."""
    prep = (
        df.sort_values(["melody_id", "position"])
        .groupby("melody_id")
        .apply(lambda g: tuple(zip(g["pitch"], g["duration"])))
    )
    melody_sigs = prep.drop_duplicates()

    seen: dict = {}
    keep_ids: list[int] = []
    exact_dupes: list[dict] = []

    for mel_id, sig in prep.items():
        sig = tuple(sig)
        if sig in seen:
            exact_dupes.append({"removed_id": mel_id, "duplicate_of": seen[sig]})
        else:
            seen[sig] = mel_id
            keep_ids.append(mel_id)

    return keep_ids, exact_dupes


# ============================================================================
# PITCH CONVERSION  (kern to MIDI)
# ============================================================================

NOTE_MAP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def kern_to_midi(k_pitch: str) -> int:
    letter = k_pitch[0].lower()

    if k_pitch[0].islower():
        count = 0
        for ch in k_pitch:
            if ch.lower() == letter:
                count += 1
            else:
                break
        # c is C4, cc is C5 
        octave = 3 + count
    else:
        count = 0
        for ch in k_pitch:
            if ch.upper() == k_pitch[0]:
                count += 1
            else:
                break
        # C is C3, CC is C2
        octave = 4 - count

    midi = 12 * (octave + 1) + NOTE_MAP[letter]
    remainder = k_pitch[count:]
    midi += remainder.count("#")
    midi -= remainder.count("-")
    return midi


# ============================================================================
# INTERVAL COMPUTATION
# ============================================================================

def get_intervals(df_notes: pd.DataFrame,
                  group_col: str = "label_0") -> pd.DataFrame:
    """Compute successive pitch intervals (in semitones) per melody."""
    rows = []
    for mel_id, group in (
        df_notes.sort_values(["melody_id", "position"]).groupby("melody_id")
    ):
        pitches = group["midi_pitch"].values
        label = group[group_col].iloc[0]
        for i in range(1, len(pitches)):
            rows.append({
                "melody_id": mel_id,
                "interval": int(pitches[i] - pitches[i - 1]),
                group_col: label,
            })
    return pd.DataFrame(rows)


# ============================================================================
# KEY EXTRACTION FROM KERN TANDEM INTERPRETATIONS
# ============================================================================

# _KEY_TANDEM_RE = re.compile(r"^\*[A-Ga-g](?:#|-)?\s*:\s*$", re.M)
# _KSIG_RE       = re.compile(r"^\*k\[[^\]]*\]\s*$", re.M)


# def extract_keys_from_files(corpus_root: pathlib.Path) -> pd.DataFrame:
#     """Read tonic and key-signature tandem interpretations from all .krn files."""
#     data = []
#     for f in sorted(corpus_root.rglob("*.krn")):
#         txt = f.read_text(encoding="utf-8", errors="ignore")
#         key_m = _KEY_TANDEM_RE.search(txt)
#         ksig_m = _KSIG_RE.search(txt)
#         data.append({
#             "path": str(f),
#             "tonic": key_m.group(0).strip() if key_m else None,
#             "key_signature": ksig_m.group(0).strip() if ksig_m else None,
#         })
#     return pd.DataFrame(data)


# ============================================================================
# POLYPHONY CHECK
# ============================================================================

def get_voice_count(path: pathlib.Path) -> int:
    """Return the number of spines (voices) declared in a single krn file.
    Determined by counting tab-separated columns on the first **kern
    interpretation line."""
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("**"):
                return line.count("\t") + 1
    return 1


def check_polyphony(corpus_root: pathlib.Path) -> Dict[str, int]:
    """Return a dict  {file_path_str: n_voices}  for every .krn file."""
    return {
        str(f): get_voice_count(f)
        for f in sorted(corpus_root.rglob("*.krn"))
    }


def summarise_polyphony(file_voices: Dict[str, int]) -> None:
    """Pretty-print an aggregate summary and list any non-monophonic files."""
    from collections import Counter as _Counter
    agg = _Counter(file_voices.values())
    for n in sorted(agg):
        print(f"{n} voice(s): {agg[n]} files")

    poly_files = {p: v for p, v in file_voices.items() if v > 1}
    if poly_files:
        print(f"\n {len(poly_files)} non-monophonic file(s) detected:")
        for p, v in sorted(poly_files.items()):
            print(f"{v} voices -> {p}")
    else:
        print(" All files are monophonic.")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_melody_length_distribution(df_notes: pd.DataFrame,
                                    keep_ids: List[int],
                                    out_dir: Path,
                                    corpus_name: str,
                                    max_len: int = 128):
    ids = df_notes[df_notes.melody_id.isin(keep_ids)]
    lengths = ids.groupby("melody_id")["position"].count()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lengths, bins=50, color="#2d5a7b", edgecolor="white", alpha=0.85)
    axes[0].axvline(x=max_len, color="red", linestyle="--", linewidth=2,
                    label=f"max_len = {max_len}")
    axes[0].set_xlabel("Melody length (number of notes)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Melody length distribution ({corpus_name})")
    axes[0].legend()

    n_truncated = (lengths > max_len).sum()
    n_total = len(lengths)
    stats_text = (
        f"Total: {n_total}\nMean: {lengths.mean():.1f}\n"
        f"Median: {lengths.median():.1f}\n"
        f"Truncated (>{max_len}): {n_truncated} ({n_truncated / n_total * 100:.1f}%)"
    )
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                 va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    axes[1].plot(sorted_lengths, cumulative, color="#2d5a7b", linewidth=2)
    axes[1].axvline(x=max_len, color="red", linestyle="--", linewidth=2,
                    label=f"max_len = {max_len}")
    axes[1].set_xlabel("Melody length (number of notes)")
    axes[1].set_ylabel("Cumulative proportion")
    axes[1].set_title("Cumulative melody length distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "melody_length_distribution.png", dpi=100,
                bbox_inches="tight")
    plt.close()


def plot_pitch_frequency(melodies: List[List[KernEvent]],
                         out_dir: Path, corpus_name: str):
    pitch_counter: Counter = Counter()
    for events in melodies:
        for ev in events:
            if not ev.is_rest:
                pitch_counter[ev.pitch] += 1

    df = (
        pd.DataFrame(pitch_counter.items(), columns=["pitch", "count"])
        .sort_values("count", ascending=False)
        .head(30)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="pitch", y="count", palette="viridis", hue="pitch", legend=False)
    plt.xticks(rotation=90)
    plt.title(f"Most frequent pitches – top 30 ({corpus_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "pitch_freq.png", dpi=100, bbox_inches="tight")
    plt.close()

    print(f"  Unique pitches: {len(pitch_counter)}")


def plot_rhythm_frequency(melodies: List[List[KernEvent]],
                          out_dir: Path, corpus_name: str):
    dur_counter: Counter = Counter()
    for events in melodies:
        for ev in events:
            dur_counter[ev.duration] += 1

    df = (
        pd.DataFrame(dur_counter.items(), columns=["duration", "count"])
        .sort_values("count", ascending=False)
        .head(20)
    )
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x="duration", y="count", palette="viridis", hue="duration", legend=False)
    plt.title(f"Most frequent rhythmic durations ({corpus_name})")
    plt.xlabel("Duration (kern reciprocal notation)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "rhythm_freq.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_rhythm_by_group(df_events: pd.DataFrame, group_col: str,
                         out_dir: Path, corpus_name: str):
    plt.figure(figsize=(12, 5))
    sns.countplot(
        data=df_events,
        x="duration",
        hue=group_col,
        order=df_events["duration"].value_counts().index[:10],
        palette="viridis",
    )
    plt.xticks(rotation=45)
    plt.title(f"Rhythm by {group_col} ({corpus_name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"rhythm_by_{group_col}.png", dpi=120,
                bbox_inches="tight")
    plt.close()


def plot_interval_distribution(df_intervals: pd.DataFrame,
                               group_col: str,
                               out_dir: Path,
                               corpus_name: str):
    n_groups = df_intervals[group_col].nunique()
    n_panels = 1 if n_groups <= 1 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Overall
    clipped = df_intervals["interval"].clip(-15, 15)
    axes[0].hist(clipped, bins=np.arange(-15.5, 16.5, 1),
                 color="#2d5a7b", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Pitch interval (semitones)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Pitch interval distribution ({corpus_name})")
    axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Unison")
    axes[0].legend()

    # By top-2 groups (only when multiple groups exist)
    if n_panels == 2:
        groups = df_intervals[group_col].value_counts().index[:2]
        for grp in groups:
            subset = df_intervals[df_intervals[group_col] == grp]["interval"].clip(-15, 15)
            axes[1].hist(subset, bins=np.arange(-15.5, 16.5, 1),
                         alpha=0.5, label=grp, density=True)
        axes[1].set_xlabel("Pitch interval (semitones)")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"Interval distribution by {group_col}")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "interval_distribution.png", dpi=100,
                bbox_inches="tight")
    plt.close()


def plot_interval_by_group(df_intervals: pd.DataFrame, group_col: str,
                           out_dir: Path, corpus_name: str):
    df_top = (
        df_intervals
        .groupby(group_col)["interval"]
        .value_counts()
        .groupby(group_col)
        .head(10)
        .rename("count")
        .reset_index()
    )
    df_top["proportion"] = df_top.groupby(group_col)["count"].transform(
        lambda x: x / x.sum()
    )

    plt.figure(figsize=(14, 5))
    g = sns.barplot(
        data=df_top,
        x="interval",
        y="proportion",
        hue=group_col,
        order=df_intervals["interval"].value_counts().head(10).index,
        palette="viridis",
    )
    sns.move_legend(g, "upper right", title=group_col.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.title(f"Top intervals by {group_col} – normalised ({corpus_name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"interval_by_{group_col}.png", dpi=100,
                bbox_inches="tight")
    plt.close()


def plot_bigram_heatmap(df_notes: pd.DataFrame, keep_ids: List[int],
                        out_dir: Path, corpus_name: str):
    ids = df_notes[df_notes.melody_id.isin(keep_ids)]
    pitch_class_names = ["C", "C#", "D", "D#", "E", "F",
                         "F#", "G", "G#", "A", "A#", "B"]

    transitions = np.zeros((12, 12))
    for _, group in ids.sort_values(["melody_id", "position"]).groupby("melody_id"):
        pitches = group["midi_pitch"].values
        for i in range(len(pitches) - 1):
            transitions[pitches[i] % 12][pitches[i + 1] % 12] += 1

    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = transitions / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(np.log1p(transitions), xticklabels=pitch_class_names,
                yticklabels=pitch_class_names, cmap="YlOrRd", ax=axes[0],
                cbar_kws={"label": "Count (log scale)"})
    axes[0].set_xlabel("Next pitch class")
    axes[0].set_ylabel("Current pitch class")
    axes[0].set_title(f"Bigram counts – log ({corpus_name})")

    sns.heatmap(trans_prob, xticklabels=pitch_class_names,
                yticklabels=pitch_class_names, cmap="YlOrRd", ax=axes[1],
                cbar_kws={"label": "P(next | current)"})
    axes[1].set_xlabel("Next pitch class")
    axes[1].set_ylabel("Current pitch class")
    axes[1].set_title("Bigram transition probabilities")

    plt.tight_layout()
    plt.savefig(out_dir / "bigram_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_key_mode_proportions(key_df: pd.DataFrame, group_col: str,
                              out_dir: Path, corpus_name: str):
    """Plot major/minor proportions when estimated-key data is available."""
    if "mode" not in key_df.columns or group_col not in key_df.columns:
        print("  [skip] key mode plot – missing columns")
        return

    tab = key_df.groupby([group_col, "mode"]).size().unstack(fill_value=0)
    tab_prop = tab.div(tab.sum(axis=1), axis=0)

    ax = tab_prop.plot(kind="bar", stacked=True, figsize=(7, 4),
                       colormap="viridis")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_title(f"Major vs minor modes ({corpus_name})")
    ax.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "major_minor_proportion.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def plot_polyphony_overview(df_events: pd.DataFrame, group_col: str,
                            out_dir: Path, corpus_name: str):
    """Bar charts showing voice-count distribution and mono/poly split per group."""
    melody_meta = (
        df_events[["melody_id", "n_voices", "is_monophonic", group_col]]
        .drop_duplicates(subset="melody_id")
    )

    has_groups = melody_meta[group_col].nunique() > 1
    n_panels = 2 if has_groups else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Left / only: overall voice-count distribution
    vc = melody_meta["n_voices"].value_counts().sort_index()
    axes[0].bar(vc.index.astype(str), vc.values, color="#2d5a7b",
                edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Number of voices (spines)")
    axes[0].set_ylabel("Number of melodies")
    axes[0].set_title(f"Voice-count distribution ({corpus_name})")
    for i, (x, y) in enumerate(zip(vc.index.astype(str), vc.values)):
        axes[0].text(i, y + 0.5, str(y), ha="center", va="bottom", fontsize=9)

    # Right: monophonic vs polyphonic per group (only with multiple groups)
    if has_groups:
        ct = (
            melody_meta
            .assign(texture=melody_meta["is_monophonic"].map(
                {True: "monophonic", False: "polyphonic"}))
            .groupby([group_col, "texture"])
            .size()
            .unstack(fill_value=0)
        )
        ct_prop = ct.div(ct.sum(axis=1), axis=0)
        ct_prop.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
        axes[1].set_ylabel("Proportion")
        axes[1].set_xlabel(group_col.replace("_", " ").title())
        axes[1].set_title(f"Monophonic vs polyphonic by {group_col}")
        axes[1].legend(title="Texture", bbox_to_anchor=(1.02, 1),
                       loc="upper left")

    plt.tight_layout()
    plt.savefig(out_dir / "polyphony_overview.png", dpi=120,
                bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_eda(corpus_root: Path,
            corpus_name: str = "corpus",
            label_depth: int = 2,
            out_dir: Optional[Path] = None,
            remove_duplicates: bool = False,
            max_len: int = 128):

    if out_dir is None:
        out_dir = Path(f"eda_output_{corpus_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    group_col = "label_0"

    # Polyphony check
    print("=" * 60)
    print(f"EDA for: {corpus_name}  ({corpus_root})")
    print("=" * 60)

    print("\n[1] Polyphony check")
    file_voices = check_polyphony(corpus_root)
    summarise_polyphony(file_voices)

    # Collect & parse
    print("\n[2] Collecting & parsing melodies …")
    melodies, meta = collect_corpus(corpus_root, label_depth=label_depth,
                                    file_voices=file_voices,
                                    corpus_name=corpus_name)
    print(f"    Melodies found: {len(melodies)}")
    n_poly = sum(1 for m in meta if not m["is_monophonic"])
    if n_poly:
        print(f" Of which {n_poly} are polyphonic "
              f"(first spine kept, others discarded)")
    if not melodies:
        print(" No krn files found – aborting.")
        return

    # Build events DataFrame
    print("\n[3] Building events DataFrame …")
    df_events = build_events_df(melodies, meta)
    print(f"    Shape: {df_events.shape}")
    print(f"    Columns: {list(df_events.columns)}")

    n_groups = df_events[group_col].nunique() if group_col in df_events.columns else 0
    has_groups = n_groups > 1

    if has_groups:
        grp_counts = df_events.groupby(group_col)["melody_id"].nunique()
        print(f"\n    Melodies per {group_col}:")
        for g, c in grp_counts.items():
            print(f"      {g}: {c}")
    else:
        print("\n Flat corpus (no sub-groups) – group-faceted plots will be skipped.")

    # Duplicate removal
    print("\n[4] Duplicate detection …")
    keep_ids, exact_dupes = find_exact_duplicates(df_events)
    print(f"    Total melodies: {len(melodies)}")
    print(f"    Exact duplicates: {len(exact_dupes)}")
    print(f"    Unique melodies: {len(keep_ids)}")

    if remove_duplicates and exact_dupes:
        df_paths = df_events[["filename", "path", "melody_id"]].drop_duplicates()
        df_dupes = pd.DataFrame(exact_dupes)
        dup_paths = df_dupes.merge(df_paths, left_on="removed_id",
                                   right_on="melody_id", how="inner")
        for p in dup_paths["path"]:
            os.remove(p)
        print(f"Removed {len(exact_dupes)} duplicate files from disk.")

    # MIDI pitch conversion
    print("\n[5] Converting pitches to MIDI …")
    df_notes = df_events[~df_events["is_rest"]].copy()
    df_notes["midi_pitch"] = df_notes["pitch"].apply(kern_to_midi)
    print(f"    Note events (excl. rests): {len(df_notes)}")

    # Intervals
    print("\n[6] Computing intervals …")
    ids_notes = df_notes[df_notes.melody_id.isin(keep_ids)]
    df_intervals = get_intervals(ids_notes, group_col=group_col)
    print(f"    Interval rows: {len(df_intervals)}")

    # ── 7. Key extraction from tandem interpretations ──────────────────
    # print("\n[7] Extracting keys from tandem interpretations …")
    # key_df = extract_keys_from_files(corpus_root)
    # n_with_key = key_df["tonic"].notna().sum()
    # print(f"    Files with tonic info: {n_with_key}/{len(key_df)}")

    # Merge group label into key_df for plotting
    # path_labels = (
    #     df_events[["path", group_col]]
    #     .drop_duplicates()
    #     .set_index("path")[group_col]
    # )
    # key_df[group_col] = key_df["path"].map(path_labels)

    # Save intermediate CSVs
    print("\n[8] Saving CSVs …")
    df_events.to_csv(out_dir / "events.csv", index=False)
    df_notes.to_csv(out_dir / "notes.csv", index=False)
    df_intervals.to_csv(out_dir / "intervals.csv", index=False)
    # key_df.to_csv(out_dir / "keys.csv", index=False)

    # Polyphony report (per-file)
    poly_df = (
        df_events[["melody_id", "filename", "path", "n_voices", "is_monophonic"]]
        .drop_duplicates(subset="melody_id")
        .sort_values("n_voices", ascending=False)
    )
    poly_df.to_csv(out_dir / "polyphony_report.csv", index=False)
    print(f"    Polyphony report: {(~poly_df['is_monophonic']).sum()} "
          f"polyphonic / {len(poly_df)} total")

    # Unique melodies in compressed format
    unique_mel = (
        df_notes[df_notes.melody_id.isin(keep_ids)]
        .sort_values(["melody_id", "position"])
        .groupby("melody_id")["midi_pitch"]
        .apply(list)
    )
    pd.DataFrame({"melody_id": unique_mel.index, "pitch": unique_mel.values})\
        .to_csv(out_dir / "unique_melodies.csv", index=False)
    print(f"Saved to {out_dir}/")

    # Plots
    print("\n[9] Generating plots …")

    print(" Melody length distribution")
    plot_melody_length_distribution(df_notes, keep_ids, out_dir,
                                   corpus_name, max_len)

    print(" Pitch frequency")
    plot_pitch_frequency(melodies, out_dir, corpus_name)

    print(" Rhythm frequency")
    plot_rhythm_frequency(melodies, out_dir, corpus_name)

    if has_groups:
        print(" Rhythm by group")
        plot_rhythm_by_group(df_events, group_col, out_dir, corpus_name)
    else:
        print(" Rhythm by group  [skip – single group]")

    print(" Interval distribution")
    plot_interval_distribution(df_intervals, group_col, out_dir, corpus_name)

    if has_groups:
        print(" Interval by group")
        plot_interval_by_group(df_intervals, group_col, out_dir, corpus_name)
    else:
        print(" Interval by group  [skip – single group]")

    print(" Bigram heatmap")
    plot_bigram_heatmap(df_notes, keep_ids, out_dir, corpus_name)

    # if has_groups:
    #     print(" Key/mode proportions")
    #     plot_key_mode_proportions(key_df, group_col, out_dir, corpus_name)
    # else:
    #     print(" Key/mode proportions  [skip – single group]")

    print(" Polyphony overview")
    plot_polyphony_overview(df_events, group_col, out_dir, corpus_name)

    print("\n Done.  All outputs in:", out_dir)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EDA & Preprocessing for **kern musical corpora"
    )
    parser.add_argument(
        "--corpus", default="meertens",
        help="Human-readable corpus name (used in titles & filenames)"
    )
    parser.add_argument(
        "--root", required=True, type=Path,
        help="Root directory of the corpus (e.g. data/meertens)"
    )
    parser.add_argument(
        "--label-depth", type=int, default=2,
        help="How many path levels to use as hierarchical labels "
             "(Essen=2 for continent/country, Meertens=0 for flat folder)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for CSVs and plots (default: eda_output_<corpus>)"
    )
    parser.add_argument(
        "--remove-duplicates", action="store_true",
        help="Actually delete duplicate .krn files from disk (destructive!)"
    )
    parser.add_argument(
        "--max-len", type=int, default=128,
        help="Melody length cutoff shown on plots"
    )
    args = parser.parse_args()

    run_eda(
        corpus_root=args.root,
        corpus_name=args.corpus,
        label_depth=args.label_depth,
        out_dir=args.out_dir,
        remove_duplicates=args.remove_duplicates,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()