import pandas as pd
import numpy as np
import os
import re
import pathlib
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")

# ============================================================================
# KERN SYNTAX / CHARACTER SETS
# ============================================================================

DURATION_CHARS = set("0123456789.")
PITCH_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
ACCIDENTAL_CHARS = set("_-#n")
REST_CHAR = "r"
PAUSE_CHAR = ";"
GROUPING_CHARS = set("{}()[]")

ORNAMENT_CHARS = set("Mm$STtwWRO")
APPOGGIATURA_CHARS = set("pP")
GRACE_CHAR = "q"
GROUPETTO_CHAR = "Q"
ARTICULATION_CHARS = set("z'\"`^~:I")
BOWING_CHARS = set("uv")
STEM_DIR_CHARS = set("/\\")
BEAM_CHARS = set("LJ")
PARTIAL_BEAM_CHARS = set("kK")
EDITORIAL_CHARS = set("xXyY?")
USER_MARK_CHARS = set("iltNUVZ@%+")

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

NOTE_MAP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

FRACTION_RE = re.compile(r'(?<!\*)\b\d+/\d+')
DUR_RE = re.compile(r'^(\d+\.*)')

KEY_TANDEM_RE = re.compile(r'^\*[A-Ga-g](?:#|-)?:\s*$', re.M)
KSIG_RE = re.compile(r'^\*k\[[^\]]*\]\s*$', re.M)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_null_token(line: str) -> bool:
    return line.startswith(".")


def has_grace_or_groupetto(tok: str) -> bool:
    """Durationless ornaments, should not be counted as events."""
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
    tok = tok.strip()
    if not tok or is_null_token(tok):
        return None
    if has_grace_or_groupetto(tok):
        return None
 
    tok = strip_non_core_signifiers(tok)
 
    while tok and tok[0] in GROUPING_CHARS:
        tok = tok[1:]
    while tok and tok[-1] in GROUPING_CHARS:
        tok = tok[:-1]
 
    if not tok:
        return None
 
    is_rest = REST_CHAR in tok
 
    dur_chars = []
    pitch_chars = []
    for ch in tok:
        if ch in DURATION_CHARS:
            dur_chars.append(ch)
        elif ch in PITCH_CHARS or ch in ACCIDENTAL_CHARS:
            pitch_chars.append(ch)
 
    duration = "".join(dur_chars) if dur_chars else ""
    pitch = "".join(pitch_chars) if pitch_chars else ""
 
    if is_rest:
        pitch = "r"
 
    return KernEvent(duration=duration, pitch=pitch, is_rest=is_rest)


# ============================================================================
# FILE LOADING
# ============================================================================

def load_kern_melody(path: pathlib.Path, spine=0):
    """
    Parse a kern file and extract the melodic line.
    
    For multi-spine (polyphonic) files, extracts the specified spine.
    Default spine=0 takes the first (leftmost) voice, which is
    conventionally the main melody in folk song collections.
    """
    out = []
    n_spines = 1
    
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            
            # Detect number of spines from header
            if line.startswith("**"):
                n_spines = line.count("\t") + 1
                continue
            
            # Skip comments, interpretations, barlines
            if line.startswith(("!", "=")):
                continue
            
            # Skip spine manipulation lines (splits, joins, terminators)
            if line.startswith("*"):
                continue
            
            # Extract the target spine
            fields = line.split("\t")
            if spine < len(fields):
                tok = fields[spine].strip()
            else:
                continue
            
            # Skip null tokens (no event in this spine)
            if not tok or tok == ".":
                continue
            
            ev = parse_kern_token(tok)
            if ev is not None:
                out.append(ev)
    return out

# ============================================================================
# PITCH CONVERSION  (kern to MIDI)
# ============================================================================

def kern_to_midi(k_pitch: str) -> int:
    if not k_pitch or k_pitch == "r":
        return None
    
    letter = k_pitch[0].lower()
    if letter not in NOTE_MAP:
        return None

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
            if ch.upper() == letter.upper():
                count += 1
            else:
                break
        # C is C3, CC is C2
        octave = 4 - count

    midi = 12 * (octave + 1) + NOTE_MAP[letter]
    remainder = k_pitch[count:]
    for ch in remainder:
        if ch == "#":
            midi += 1
        elif ch == "-":
            midi -= 1
        elif ch == "n":
            pass

    return midi

# ============================================================================
# KEY EXTRACTION
# ============================================================================

def extract_keys(root_path: Path):
    """Extract key and key signature from kern file tandem interpretations."""
    
    files = sorted(root_path.rglob("*.krn"))
    data = []
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        key_m = KEY_TANDEM_RE.search(txt)
        ksig_m = KSIG_RE.search(txt)
        
        key_line = key_m.group(0).strip() if key_m else None
        ksig_line = ksig_m.group(0).strip() if ksig_m else None
        
        rel = f.relative_to(root_path)
        data.append({
            "path": str(f),
            "key_tandem": key_line,
            "key_signature": ksig_line,
        })
        
    return pd.DataFrame(data)

# =============================================================================
# CORPUS PROCESSOR
# =============================================================================

class CorpusProcessor:
    """
    Full preprocessing and EDA pipeline for a kern corpus.
    """

    def __init__(self, root_path: str, corpus_name: str = "Corpus", label_depth: int = 2):
        """
        Args:
            root_path: path to the corpus root folder
            corpus_name: name for display in plots and prints
            label_depth: how many path levels to use as hierarchical labels.
                         Essen=2 (continent/country), Meertens=0 (flat folder)
                         1 = just first subfolder level
        """
        self.root = Path(root_path)
        self.name = corpus_name
        self.label_depth = label_depth
        self.melodies = None
        self.meta = None
        self.df_events = None
        self.df_notes = None
        self.df_duplicates = None
        self.df_intervals = None
        self.df_validation = None
        self.unique_melodies = None

    # ── Loading and cleaning ──

    def load_corpus(self):
        """Parse all kern files and build event dataframe."""
        melodies = []
        meta_list = []

        for path in self.root.rglob("*.krn"):
            events = load_kern_melody(path)
            rel_parts = path.relative_to(self.root).parts

            # Extract hierarchical labels based on label_depth
            if self.label_depth >= 2:
                continent = rel_parts[0] if len(rel_parts) > 1 else "unknown"
                country = rel_parts[1] if len(rel_parts) > 2 else "unknown"
            elif self.label_depth == 1:
                continent = rel_parts[0] if len(rel_parts) > 1 else self.name
                country = "all"
            else:
                continent = self.name
                country = "all"

            melodies.append(events)
            meta_list.append({
                "filename": path.name,
                "path": str(path.relative_to(self.root)),
                "continent": continent,
                "country": country,
                "length": len(events),
            })

        self.melodies = melodies
        self.meta = meta_list
        print(f"[{self.name}] Loaded {len(melodies)} melodies")

        # Build events dataframe
        events_df = []
        for i, (events, m) in enumerate(zip(melodies, meta_list)):
            for j, ev in enumerate(events):
                row = m.copy()
                row.update({
                    "melody_id": i,
                    "position": j,
                    "pitch": ev.pitch,
                    "duration": ev.duration,
                    "is_rest": ev.is_rest,
                })
                events_df.append(row)

        self.df_events = pd.DataFrame(events_df)
        print(f"[{self.name}] Total events: {len(self.df_events)}")

        # Notes only (no rests), with MIDI pitch
        self.df_notes = self.df_events[~self.df_events["is_rest"]].copy()
        self.df_notes["midi_pitch"] = self.df_notes["pitch"].apply(kern_to_midi)
        self.df_notes = self.df_notes.dropna(subset=["midi_pitch"])
        self.df_notes["midi_pitch"] = self.df_notes["midi_pitch"].astype(int)
        print(f"[{self.name}] Notes (no rests): {len(self.df_notes)}")

        return self

    # ── Validation ──

    def validate_corpus(self, remove_invalid=True, idyom_timebase=96):
        """
        Validate each melody for:
        1. Has pitched notes (not just rests/comments)
        2. IDyOM-compatible durations — kern durations must produce integer
           ticks at IDyOM's timebase
        3. No invalid duration strings
        4. Minimum length (at least 2 notes for prediction)

        Also logs polyphonic files (multi-voice) as informational —
        these are NOT removed since load_kern_melody extracts the
        first voice automatically.
        """
        if self.df_events is None:
            raise ValueError("Call load_corpus() first")

        invalid_files = []
        reasons = []
        polyphonic_count = 0

        for mel_id in self.df_events["melody_id"].unique():
            mel_events = self.df_events[self.df_events["melody_id"] == mel_id]
            mel_meta = self.meta[mel_id]
            filepath = mel_meta["path"]
            full_path = self.root / filepath

            # 1. Has pitched notes
            notes = mel_events[~mel_events["is_rest"]]
            if len(notes) == 0:
                invalid_files.append(mel_id)
                reasons.append({"melody_id": mel_id, "path": filepath, "reason": "no_pitched_notes"})
                continue

            # 2. IDyOM-compatible durations — scan raw kern file
            bad_tick_durs = []
            try:
                raw_text = full_path.read_text(encoding="utf-8", errors="ignore")
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line or line.startswith(("*", "!", "=", ".")):
                        continue
                    for tok in line.split("\t"):
                        tok = tok.strip()
                        if not tok:
                            continue
                        m = DUR_RE.match(tok)
                        if m:
                            dur_str = m.group(1)
                            try:
                                base = int(dur_str.rstrip("."))
                                dots = dur_str.count(".")
                                t = idyom_timebase / base
                                for d in range(dots):
                                    t += idyom_timebase / base / (2 ** (d + 1))
                                ticks = t  # assigned outside the dots loop
                                if ticks != int(ticks):
                                    bad_tick_durs.append(dur_str)
                            except (ValueError, ZeroDivisionError):
                                pass
            except Exception:
                pass

            if bad_tick_durs:
                unique_bad = list(set(bad_tick_durs))
                invalid_files.append(mel_id)
                reasons.append({
                    "melody_id": mel_id, "path": filepath,
                    "reason": f"non_integer_idyom_ticks: {unique_bad}"
                })
                continue

            # 3. Check parsed durations for unexpected characters
            bad_durations = []
            for dur in mel_events["duration"].unique():
                dur_str = str(dur).strip()
                if not dur_str:
                    continue
                if not all(c in "0123456789." for c in dur_str):
                    bad_durations.append(dur_str)

            if bad_durations:
                invalid_files.append(mel_id)
                reasons.append({
                    "melody_id": mel_id, "path": filepath,
                    "reason": f"invalid_duration: {bad_durations}"
                })
                continue

            # 4. Minimum length
            if len(notes) < 2:
                invalid_files.append(mel_id)
                reasons.append({"melody_id": mel_id, "path": filepath, "reason": "too_short"})
                continue

            # Count polyphonic files (informational only — not removed)
            if full_path.exists():
                try:
                    with full_path.open("r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.startswith("**"):
                                n_voices = line.count("\t") + 1
                                if n_voices > 1:
                                    polyphonic_count += 1
                                break
                except Exception:
                    pass

        self.df_validation = pd.DataFrame(reasons)

        print(f"[{self.name}] Validation results:")
        print(f"  Total melodies: {self.df_events['melody_id'].nunique()}")
        print(f"  Invalid (to remove): {len(invalid_files)}")

        if len(reasons) > 0:
            reason_counts = self.df_validation["reason"].apply(
                lambda r: r.split(":")[0] if ":" in r else r
            ).value_counts()
            for reason, count in reason_counts.items():
                print(f"    {reason}: {count}")

        print(f"  Polyphonic (first voice extracted, kept): {polyphonic_count}")

        if remove_invalid and invalid_files:
            self.df_events = self.df_events[~self.df_events["melody_id"].isin(invalid_files)].copy()
            self.df_notes = self.df_notes[~self.df_notes["melody_id"].isin(invalid_files)].copy()
            print(f"  Remaining after removal: {self.df_events['melody_id'].nunique()}")

        return self

    # ── Duplicate removal ──

    def remove_duplicates(self):
        """Remove exact duplicates (same pitch + duration sequence)."""
        df = self.df_notes

        # Build signatures: tuple of (pitch, duration) pairs per melody
        sigs = (df.sort_values(["melody_id", "position"])
                .groupby("melody_id")
                .apply(lambda g: tuple(zip(g["pitch"], g["duration"]))))

        seen = {}
        keep_ids = []
        exact_dupes = []

        for mel_id, sig in sigs.items():
            sig = tuple(sig)
            if sig in seen:
                exact_dupes.append({"removed_id": mel_id, "duplicate_of": seen[sig], "type": "exact"})
            else:
                seen[sig] = mel_id
                keep_ids.append(mel_id)

        self.df_duplicates = pd.DataFrame(exact_dupes)
        self.df_notes = self.df_notes[self.df_notes["melody_id"].isin(keep_ids)].copy()

        # Build unique melodies list
        self.unique_melodies = (
            self.df_notes.sort_values(["melody_id", "position"])
            .groupby("melody_id")["midi_pitch"]
            .apply(list)
        )

        print(f"[{self.name}] Duplicates removed:")
        print(f"  Exact: {len(exact_dupes)}")
        print(f"  Remaining melodies: {len(keep_ids)}")

        return self

    def fix_fractional_durations(self, mapping: dict = None) -> "CorpusProcessor":
        """
        Build (or accept) a mapping of fractional → standard kern durations,
        stored for use during export_clean_corpus.

        Args:
            mapping: optional dict like {"3/2": "2.", "4/3": "6"}.
                     If None, auto-converts each fraction to nearest kern duration.
        """
        counts = self.scan_fractional_durations()
        if not counts:
            print(f"[{self.name}] No fractional durations found.")
            self._fraction_mapping = {}
            return self

        if mapping is None:
            mapping = {frac: _kern_fraction_to_standard(frac) for frac in counts}

        print(f"[{self.name}] Fractional duration mapping:")
        for frac, kern in sorted(mapping.items()):
            print(f"  {frac:>8}  →  {kern:<6}  (occurs {counts.get(frac, 0):,} times)")

        self._fraction_mapping = mapping
        return self

    # ── Intervals utility ──

    def get_intervals(self):
        """
        Compute pitch intervals for all melodies.
        Returns a dataframe with columns: melody_id, interval, continent.
        """
        df = self.df_notes
        intervals_data = []
        for mel_id, group in df.sort_values(["melody_id", "position"]).groupby("melody_id"):
            pitches = group["midi_pitch"].values
            continent = group["continent"].iloc[0]
            for i in range(1, len(pitches)):
                intervals_data.append({
                    "melody_id": mel_id,
                    "interval": pitches[i] - pitches[i - 1],
                    "continent": continent,
                })
        # FIX: build DataFrame once, outside both loops
        self.df_intervals = pd.DataFrame(intervals_data)
        return self.df_intervals

    # ── EDA Plots ──

    def plot_melody_lengths(self, output_dir, max_len=128):
        """Melody length distribution with max_len cutoff."""
        lengths = self.df_notes.groupby("melody_id")["position"].count()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(lengths, bins=50, color="#2d5a7b", edgecolor="white", alpha=0.85)
        axes[0].axvline(x=max_len, color="red", linestyle="--", linewidth=2, label=f"max_len = {max_len}")
        axes[0].set_xlabel("Melody length (number of notes)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Melody length distribution ({self.name})")
        axes[0].legend()

        n_total = len(lengths)
        n_trunc = (lengths > max_len).sum()
        stats = (f"Total: {n_total}\nMean: {lengths.mean():.1f}\nMedian: {lengths.median():.1f}\n"
                 f"Truncated (>{max_len}): {n_trunc} ({n_trunc / n_total * 100:.1f}%)")
        axes[0].text(0.95, 0.95, stats, transform=axes[0].transAxes, va="top", ha="right",
                     fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        sorted_l = np.sort(lengths)
        cumulative = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
        axes[1].plot(sorted_l, cumulative, color="#2d5a7b", linewidth=2)
        axes[1].axvline(x=max_len, color="red", linestyle="--", linewidth=2, label=f"max_len = {max_len}")
        axes[1].set_xlabel("Melody length (number of notes)")
        axes[1].set_ylabel("Cumulative proportion")
        axes[1].set_title(f"Cumulative melody length ({self.name})")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/melody_lengths.png", dpi=150, bbox_inches="tight")
        plt.close()

        for ml in [32, 64, 96, 128, 256]:
            pct = (lengths > ml).sum() / n_total * 100
            print(f"  max_len={ml}: {pct:.1f}% truncated")

    def plot_pitch_frequency(self, output_dir, top_n=30):
        """Most frequent pitches."""
        pitch_counts = self.df_notes["pitch"].value_counts().head(top_n).reset_index()
        pitch_counts.columns = ["pitch", "count"]

        plt.figure(figsize=(10, 6))
        sns.barplot(data=pitch_counts, x="pitch", y="count", palette="viridis", hue="pitch")
        plt.xticks(rotation=90)
        plt.title(f"Most frequent pitches (top {top_n}) ({self.name})")
        plt.xlabel("Pitch")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pitch_freq.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_duration_frequency(self, output_dir, top_n=20, log_scale=False):
        """Most frequent rhythmic durations."""
        dur_counts = self.df_events["duration"].value_counts().head(top_n).reset_index()
        dur_counts.columns = ["duration", "count"]

        plt.figure(figsize=(12, 5))
        sns.barplot(data=dur_counts, x="duration", y="count", palette="viridis", hue="duration")
        # FIX: only yscale is conditional; labels and layout always apply
        if log_scale:
            plt.yscale("log")
        plt.title(f"Most frequent durations ({self.name})")
        plt.xlabel("Duration (kern)")
        plt.ylabel("Count" + (" (log)" if log_scale else ""))
        plt.tight_layout()

        fname = "duration_freq_log.png" if log_scale else "duration_freq.png"
        plt.savefig(f"{output_dir}/{fname}", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_duration_by_continent(self, output_dir, top_n=10):
        """Duration distribution by continent."""
        order = self.df_events["duration"].value_counts().index[:top_n]

        plt.figure(figsize=(12, 5))
        sns.countplot(data=self.df_events, x="duration", hue="continent",
                      order=order, palette="viridis")
        plt.title(f"Rhythmic duration by continent ({self.name})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/duration_by_continent.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_interval_distribution(self, output_dir):
        """Pitch interval distribution overall and by continent."""
        if self.df_intervals is None:
            self.get_intervals()
        # FIX: always assigned, whether intervals were just computed or already existed
        df_int = self.df_intervals

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        clipped = df_int["interval"].clip(-15, 15)
        axes[0].hist(clipped, bins=np.arange(-15.5, 16.5, 1), color="#2d5a7b", edgecolor="white", alpha=0.85)
        axes[0].set_xlabel("Pitch interval (semitones)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Pitch interval distribution ({self.name})")
        axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Unison")
        axes[0].legend()

        continents = [c for c in df_int["continent"].unique()
                      if df_int[df_int["continent"] == c].shape[0] > 100]
        for continent in sorted(continents):
            subset = df_int[df_int["continent"] == continent]["interval"].clip(-15, 15)
            axes[1].hist(subset, bins=np.arange(-15.5, 16.5, 1), alpha=0.5,
                         label=continent, density=True)
        axes[1].set_xlabel("Pitch interval (semitones)")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"Interval distribution by continent ({self.name})")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/interval_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

        intervals = df_int["interval"]
        print(f"  Mean interval: {intervals.mean():.2f} semitones")
        print(f"  Steps (±1-2): {((intervals.abs() >= 1) & (intervals.abs() <= 2)).mean() * 100:.1f}%")
        print(f"  Leaps (>±5): {(intervals.abs() > 5).mean() * 100:.1f}%")

    def plot_bigram_heatmap(self, output_dir):
        """Pitch class bigram transition heatmap."""
        df = self.df_notes
        pitch_class_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        transitions = np.zeros((12, 12))
        for mel_id, group in df.sort_values(["melody_id", "position"]).groupby("melody_id"):
            pitches = group["midi_pitch"].values
            for i in range(len(pitches) - 1):
                transitions[pitches[i] % 12][pitches[i + 1] % 12] += 1

        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = transitions / row_sums

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        sns.heatmap(np.log1p(transitions), xticklabels=pitch_class_names,
                    yticklabels=pitch_class_names, cmap="YlOrRd", ax=axes[0],
                    cbar_kws={"label": "log(count + 1)"})
        axes[0].set_xlabel("Next pitch class")
        axes[0].set_ylabel("Current pitch class")
        axes[0].set_title(f"Bigram counts ({self.name})")

        sns.heatmap(trans_prob, xticklabels=pitch_class_names,
                    yticklabels=pitch_class_names, cmap="YlOrRd", ax=axes[1],
                    cbar_kws={"label": "P(next | current)"})
        axes[1].set_xlabel("Next pitch class")
        axes[1].set_ylabel("Current pitch class")
        axes[1].set_title(f"Transition probabilities ({self.name})")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/bigram_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_ngram_counts(self, output_dir, max_order=10):
        """N-gram counts by order."""
        melodies_list = (
            self.df_notes.sort_values(["melody_id", "position"])
            .groupby("melody_id")["midi_pitch"]
            .apply(list)
            .tolist()
        )

        counts_per_order = {}
        total_instances = {}

        for n in range(1, max_order + 1):
            ngrams = set()
            total = 0
            for mel in melodies_list:
                for i in range(len(mel) - n + 1):
                    ngrams.add(tuple(mel[i:i + n]))
                    total += 1
            # FIX: update counts after all melodies for this order, not inside loops
            counts_per_order[n] = len(ngrams)
            total_instances[n] = total

        orders = list(counts_per_order.keys())
        unique_counts = list(counts_per_order.values())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        bars = axes[0].bar(orders, unique_counts, color="#2d5a7b", edgecolor="white", alpha=0.85)
        axes[0].set_xlabel("N-gram order")
        axes[0].set_ylabel("Unique n-grams")
        axes[0].set_title(f"N-gram counts by order ({self.name})")
        axes[0].set_xticks(orders)
        for bar, count in zip(bars, unique_counts):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{count:,}", ha="center", va="bottom", fontsize=8)

        coverage = [u / t * 100 for u, t in zip(unique_counts, list(total_instances.values()))]
        axes[1].bar(orders, coverage, color="#5a8a5e", edgecolor="white", alpha=0.85)
        axes[1].set_xlabel("N-gram order")
        axes[1].set_ylabel("Unique / Total (%)")
        axes[1].set_title(f"N-gram coverage ({self.name})")
        axes[1].set_xticks(orders)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/ngram_counts.png", dpi=150, bbox_inches="tight")
        plt.close()

        cumulative = 0
        print(f"\n  {'Order':<8}{'Unique':<12}{'Instances':<12}{'Coverage %':<12}{'Cumulative'}")
        for n in orders:
            cumulative += counts_per_order[n]
            cov = counts_per_order[n] / total_instances[n] * 100
            print(f"  {n:<8}{counts_per_order[n]:<12,}{total_instances[n]:<12,}{cov:<12.1f}{cumulative:,}")

    def plot_mode_proportion(self, output_dir):
        """Major vs minor mode proportion by continent."""
        key_df = extract_keys(self.root)

        if key_df["key_tandem"].isna().all():
            print(f"  [{self.name}] No key tandem interpretations found, skipping mode plot")
            return

        key_df["mode"] = key_df["key_tandem"].apply(
            lambda k: "minor" if isinstance(k, str) and len(k) > 1 and k[1].islower()
            else "major" if isinstance(k, str)
            else None
        )
        key_df = key_df.dropna(subset=["mode"])

        def get_label(p):
            try:
                parts = Path(p).relative_to(self.root).parts
                if self.label_depth >= 1 and len(parts) > 1:
                    return parts[0]
                else:
                    return self.name
            except ValueError:
                return self.name

        key_df["continent"] = key_df["path"].apply(get_label)

        tab = key_df.groupby(["continent", "mode"]).size().unstack(fill_value=0)
        tab_prop = tab.div(tab.sum(axis=1), axis=0)

        ax = tab_prop.plot(kind="bar", stacked=True, figsize=(7, 4), colormap="viridis")
        ax.set_ylabel("Proportion")
        ax.set_xlabel("Continent")
        ax.set_title(f"Major vs minor modes ({self.name})")
        ax.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mode_proportion.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Summary statistics ──

    def print_summary(self):
        """Print corpus summary statistics."""
        df = self.df_notes
        n_melodies = df["melody_id"].nunique()
        n_events = len(df)
        lengths = df.groupby("melody_id")["position"].count()
        n_unique_pitches = df["midi_pitch"].nunique()
        pitch_range = (df["midi_pitch"].min(), df["midi_pitch"].max())
        continents = df["continent"].unique()

        print(f"\n{'=' * 60}")
        print(f"CORPUS SUMMARY: {self.name}")
        print(f"{'=' * 60}")
        print(f"  Melodies: {n_melodies}")
        print(f"  Total notes: {n_events}")
        print(f"  Unique pitches: {n_unique_pitches}")
        print(f"  Pitch range: {pitch_range[0]} - {pitch_range[1]} (MIDI)")
        print(f"  Mean melody length: {lengths.mean():.1f}")
        print(f"  Median melody length: {lengths.median():.1f}")
        print(f"  Min/Max length: {lengths.min()} / {lengths.max()}")
        print(f"  Continents/regions: {list(continents)}")
        if self.df_duplicates is not None and len(self.df_duplicates) > 0:
            print(f"  Duplicates removed: {len(self.df_duplicates)}")
        print(f"{'=' * 60}\n")

    # ── Export clean corpus ──

    def export_clean_corpus(self, output_root=None):
        """
        Copy only valid, non-duplicate kern files to a new folder,
        preserving the original directory structure.
        """
        if output_root is None:
            output_root = str(self.root) + "_unique"

        output_root = Path(output_root)
        surviving_ids = self.df_notes["melody_id"].unique()

        copied = 0
        skipped = 0

        for mel_id in surviving_ids:
            if mel_id >= len(self.meta):
                continue
            mel_meta = self.meta[mel_id]
            rel_path = mel_meta["path"]

            src = self.root / rel_path
            dst = output_root / rel_path

            if not src.exists():
                skipped += 1
                continue

            # FIX: actually copy the file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

        print(f"[{self.name}] Exported clean corpus:")
        print(f"  Destination: {output_root}")
        print(f"  Files copied: {copied}")
        if skipped > 0:
            print(f"  Files skipped (not found): {skipped}")

        return output_root

    # ── Run all ──

    def run_all(self, output_dir="eda_output", max_len=128, export_clean=True):
        """Run full pipeline: load, validate, clean, deduplicate, plot, export."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- Loading {self.name} ---")
        self.load_corpus()

        print(f"\n--- Validating ---")
        self.validate_corpus(remove_invalid=True)

        print(f"\n--- Removing duplicates ---")
        self.remove_duplicates()

        self.print_summary()

        print(f"\n--- Computing intervals ---")
        self.get_intervals()

        print(f"\n--- Generating plots ---")
        print("Melody lengths:")
        self.plot_melody_lengths(output_dir, max_len=max_len)

        print("\nPitch frequency:")
        self.plot_pitch_frequency(output_dir)

        print("\nDuration frequency:")
        self.plot_duration_frequency(output_dir)
        self.plot_duration_frequency(output_dir, log_scale=True)

        print("\nDuration by continent:")
        self.plot_duration_by_continent(output_dir)

        print("\nInterval distribution:")
        self.plot_interval_distribution(output_dir)

        print("\nBigram heatmap:")
        self.plot_bigram_heatmap(output_dir)

        print("\nN-gram counts:")
        self.plot_ngram_counts(output_dir)

        print("\nMode proportion:")
        self.plot_mode_proportion(output_dir)

        print(f"\n--- All plots saved to {output_dir}/ ---")

        if export_clean:
            print(f"\n--- Exporting clean corpus ---")
            self.export_clean_corpus()

        return self

    def get_melodies_for_transformer(self):
        """Return clean melodies as list of lists (MIDI pitch integers)."""
        if self.unique_melodies is None:
            self.unique_melodies = (
                self.df_notes.sort_values(["melody_id", "position"])
                .groupby("melody_id")["midi_pitch"]
                .apply(list)
            )
        return self.unique_melodies.tolist()

    def get_melody_ids(self):
        """Return melody IDs matching the order of get_melodies_for_transformer."""
        if self.unique_melodies is None:
            self.get_melodies_for_transformer()
        return self.unique_melodies.index.tolist()
    
# =============================================================================
# MAIN
# =============================================================================
 
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python corpus_eda.py <path_to_corpus> [corpus_name] [label_depth]")
        print("  label_depth: 2 for Essen (continent/country), 0 for flat (Meertens)")
        print("Example: python corpus_eda.py data/essen Essen 2")
        print("         python corpus_eda.py data/meertens Meertens 0")
        sys.exit(1)
        
    root = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else Path(root).name
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    output = f"eda_{name.lower()}"
    
    cp = CorpusProcessor(root, corpus_name=name, label_depth=depth)
    cp.run_all(output_dir=output)
