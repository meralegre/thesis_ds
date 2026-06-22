import pandas as pd
import numpy as np
from scipy.stats import pearsonr, t as t_dist
import argparse
import os

# =============================================================
# DATA LOADING
# =============================================================

def load_ratings(filepath):
    """Load and clean human expectation ratings (.dat file)"""
    ratings = pd.read_csv(filepath)
    ratings = ratings.dropna(subset=["response"])
    ratings["note"] = ratings["note"].astype(int)
    ratings["response"] = ratings["response"].astype(float)
    ratings["pitch"] = ratings["pitch"].astype(int)

    mean_ratings = (
        ratings.groupby(["melody", "note"])["response"]
        .mean()
        .reset_index()
        .rename(columns={"response": "mean_response"})
    )

    pitches = (
        ratings.groupby(["melody", "note"])["pitch"]
        .first()
        .reset_index()
    )
    mean_ratings = mean_ratings.merge(pitches, on=["melody", "note"])

    n_subjects = ratings["subject"].nunique()
    n_probes = len(mean_ratings)
    n_melodies = mean_ratings["melody"].nunique()

    print(f"  Ratings: {n_probes} probes, {n_melodies} melodies, "
          f"{n_subjects} subjects, "
          f"range {mean_ratings['mean_response'].min():.2f}"
          f"--{mean_ratings['mean_response'].max():.2f}")

    return mean_ratings, ratings


def load_transformer_ic(filepath):
    """Load transformer per-note IC output"""
    df = pd.read_csv(filepath)
    mel_col = "melody_name" if "melody_name" in df.columns else "melody"
    df = df.rename(columns={mel_col: "melody"})
    print(f"  Transformer: {len(df)} notes, {df['melody'].nunique()} melodies")
    return df


def load_idyom_ic(filepath):
    """Load IDyOM .dat output file."""
    df = pd.read_csv(filepath, sep=r"\s+")
    print(f"  IDyOM: {len(df)} notes, {df['melody.name'].nunique()} melodies")
    return df


# =============================================================
# ALIGNMENT AND MERGING
# =============================================================

def verify_pitch_alignment(merged, transformer_df, idyom_df, n_check=10):
    """Verify that pitches match across ratings, transformer, and IDyOM."""
    misaligned = 0
    checked = min(n_check, len(merged))

    for _, row in merged.head(checked).iterrows():
        name = row["melody"]
        note = int(row["note"])
        r_pitch = int(row["pitch"])

        i_row = idyom_df[
            (idyom_df["melody.name"] == name) & (idyom_df["note.id"] == note)
        ]
        i_pitch = int(i_row["cpitch"].iloc[0]) if len(i_row) > 0 else None

        if r_pitch != i_pitch:
            print(f"    WARNING: {name} note {note}: "
                  f"rating={r_pitch}, idyom={i_pitch}")
            misaligned += 1

    if misaligned == 0:
        print(f"verified ({checked} notes)")
    else:
        print(f"WARNING: {misaligned}/{checked} misaligned!")

    return misaligned == 0


def merge_data(mean_ratings, transformer_df, idyom_df):
    """
    Merge ratings with transformer and IDyOM IC values.

    Note alignment:
      - Ratings note N = IDyOM note.id N (both 1-indexed)
      - Ratings note N = Transformer note N-1 (transformer skips first note)
    """
    mean_ratings = mean_ratings.copy()
    mean_ratings["t_note"] = mean_ratings["note"] - 1

    merged = mean_ratings.merge(
        transformer_df[["melody", "note", "ic"]].rename(
            columns={"ic": "ic_transformer", "note": "t_note_ref"}
        ),
        left_on=["melody", "t_note"],
        right_on=["melody", "t_note_ref"],
        how="inner",
    )

    idyom_ic = idyom_df[["melody.name", "note.id", "ic"]].copy()
    idyom_ic.columns = ["melody", "note_id", "ic_idyom"]
    idyom_ic["note_id"] = idyom_ic["note_id"].astype(int)
    merged = merged.merge(
        idyom_ic,
        left_on=["melody", "note"],
        right_on=["melody", "note_id"],
        how="inner",
    )

    drop = [c for c in ["t_note", "t_note_ref", "note_id"] if c in merged.columns]
    merged = merged.drop(columns=drop)
    merged = merged[["melody", "note", "pitch", "mean_response",
                      "ic_transformer", "ic_idyom"]]

    return merged


# =============================================================
# STATISTICAL ANALYSIS
# =============================================================

def pearson_with_ci(x, y):
    """Pearson r with p-value and 95% confidence interval."""
    r, p = pearsonr(x, y)
    n = len(x)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    ci_low = np.tanh(z - 1.96 * se)
    ci_high = np.tanh(z + 1.96 * se)
    return {"r": r, "r2": r ** 2, "p": p,
            "ci_low": ci_low, "ci_high": ci_high, "n": n}


def williams_test(r12, r13, r23, n):
    """
    Williams' t-test for comparing two dependent correlations.

    r12: model 1 vs criterion
    r13: model 2 vs criterion
    r23: model 1 vs model 2
    """
    r_bar = (r12 + r13) / 2.0
    R = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    t_stat = (r12 - r13) * np.sqrt(
        (n - 1) * (1 + r23)
        / (2 * (n - 1) / (n - 3) * R + r_bar**2 * (1 - r23) ** 3)
    )
    p_val = 2 * t_dist.sf(abs(t_stat), df=n - 3)
    return {"t": t_stat, "p": p_val, "df": n - 3}


def compute_reliability(ratings_raw):
    """Compute split-half reliability and Cronbach's alpha."""
    subjects = sorted(ratings_raw["subject"].unique())
    half = len(subjects) // 2
    s1 = (
        ratings_raw[ratings_raw["subject"].isin(subjects[:half])]
        .groupby(["melody", "note"])["response"].mean()
    )
    s2 = (
        ratings_raw[ratings_raw["subject"].isin(subjects[half:])]
        .groupby(["melody", "note"])["response"].mean()
    )
    common = s1.index.intersection(s2.index)
    split_r, _ = pearsonr(s1[common], s2[common])
    alpha = 2 * split_r / (1 + split_r)
    return {"split_half_r": split_r, "cronbach_alpha": alpha}


# =============================================================
# REPORTING
# =============================================================

def print_report(t_stats, i_stats, williams, reliability,
                 transformer_label, idyom_label):
    """Print a formatted summary."""
    col_w = max(len(transformer_label), len(idyom_label)) + 2
    col_w = max(col_w, 25)
    line_w = col_w + 48

    print("\n" + "=" * line_w)
    print("  HUMAN RATINGS COMPARISON REPORT")
    print("=" * line_w)

    print(f"\n  Inter-rater reliability")
    print(f"    Split-half r  = {reliability['split_half_r']:.4f}")
    print(f"    Cronbach's α  = {reliability['cronbach_alpha']:.4f}")

    n = t_stats["n"]
    print(f"\n  Correlations with mean expectedness ratings (n = {n})")
    print(f"  {'-' * (line_w)}")
    print(f"  {'Model':<{col_w}} {'r':>7} {'r²':>7} {'p':>9} {'95% CI':>18}")
    print(f"  {'-' * (line_w)}")

    for label, s in [(transformer_label, t_stats), (idyom_label, i_stats)]:
        sig = " *" if s["p"] < 0.05 else "  "
        sig = "**" if s["p"] < 0.01 else sig
        ci = f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
        print(f"  {label:<{col_w}} {s['r']:>7.4f} {s['r2']:>7.4f} "
              f"{s['p']:>8.4f}{sig} {ci}")

    print(f"\n  Williams' t-test")
    print(f"  {'-' * (line_w - 4)}")
    sig = " *" if williams["p"] < 0.05 else "  "
    sig = "**" if williams["p"] < 0.01 else sig
    print(f"  t({williams['df']}) = {williams['t']:.4f}, "
          f"p = {williams['p']:.4f}{sig}")

    print(f"\n  * p < .05  ** p < .01")

# =============================================================
# MAIN PIPELINE
# =============================================================

def run_pipeline(
    ratings_path,
    transformer_path,
    idyom_path,
    transformer_label=None,
    idyom_label=None,
    output_path=None,
):
    """Run the full comparison pipeline"""

    if transformer_label is None:
        transformer_label = "Transformer"
    if idyom_label is None:
        idyom_label = "IDyOM"

    # Load
    print("Loading data...")
    mean_ratings, ratings_raw = load_ratings(ratings_path)
    transformer_df = load_transformer_ic(transformer_path)
    idyom_df = load_idyom_ic(idyom_path)

    reliability = compute_reliability(ratings_raw)
    print(f"  Reliability: α = {reliability['cronbach_alpha']:.4f}")

    print("\nAligning notes...")
    merged = merge_data(mean_ratings, transformer_df, idyom_df)
    print(f"  Merged: {len(merged)} probed notes")

    print("  Pitch check: ", end="")
    verify_pitch_alignment(merged, transformer_df, idyom_df)

    # Correlations
    t_stats = pearson_with_ci(merged["ic_transformer"], merged["mean_response"])
    i_stats = pearson_with_ci(merged["ic_idyom"], merged["mean_response"])

    # Williams' test
    r23 = merged["ic_transformer"].corr(merged["ic_idyom"])
    w = williams_test(t_stats["r"], i_stats["r"], r23, t_stats["n"])

    print_report(t_stats, i_stats, w, reliability, transformer_label, idyom_label)
    
    if output_path:
        merged.to_csv(output_path, index=False)
        print(f"\nMerged data saved to: {output_path}")

    return merged, {"transformer": t_stats, "idyom": i_stats, "williams": w}


# =============================================================
# CLI
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Transformer and IDyOM IC against human expectation ratings"
    )
    parser.add_argument("--ratings", required=True,
                        help="Path to human ratings .dat file")
    parser.add_argument("--transformer", required=True,
                        help="Path to transformer per-note IC CSV")
    parser.add_argument("--transformer-label", default=None,
                        help="Display label (default: 'Transformer')")
    parser.add_argument("--idyom", required=True,
                        help="Path to IDyOM .dat output file")
    parser.add_argument("--idyom-label", default=None,
                        help="Display label (default: auto-parsed from filename)")
    parser.add_argument("--output", default=None,
                        help="Path to save merged results CSV")
    args = parser.parse_args()

    run_pipeline(
        ratings_path=args.ratings,
        transformer_path=args.transformer,
        idyom_path=args.idyom,
        transformer_label=args.transformer_label,
        idyom_label=args.idyom_label,
        output_path=args.output,
    )
