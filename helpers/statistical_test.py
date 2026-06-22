import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import t as t_dist


def compare_models(tf_folds, idyom_folds, experiment_name, corpus_name):
    """Compare Transformer and IDyOM IC values across folds."""
    tf = np.array(tf_folds)
    idyom = np.array(idyom_folds)
    differences = tf - idyom
    n = len(differences)

    # Means
    tf_mean = np.mean(tf)
    idyom_mean = np.mean(idyom)
    delta = np.mean(differences)

    # Paired t-test
    t_stat, p_ttest = ttest_rel(tf, idyom)

    # Wilcoxon signed-rank test
    w_stat, p_wilcoxon = wilcoxon(tf, idyom)

    # Cohen's d
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)

    # 95% confidence interval
    se = np.std(differences, ddof=1) / np.sqrt(n)
    ci_low, ci_high = t_dist.interval(0.95, df=n-1, loc=delta, scale=se)

    # Per-fold wins
    tf_wins = np.sum(differences < 0)

    # Print results
    print(f"{'='*60}")
    print(f"{experiment_name} — {corpus_name}")
    print(f"{'='*60}")

    print(f"Transformer mean IC: {tf_mean:.3f} bits (std: {np.std(tf, ddof=1):.4f})")
    print(f"IDyOM mean IC: {idyom_mean:.3f} bits (std: {np.std(idyom, ddof=1):.4f})")

    print(f"Delta (TF - IDyOM): {delta:.3f} bits")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Cohen's d: {cohens_d:.2f}")

    print(f"Paired t-test: t = {t_stat:.2f}, p = {p_ttest:.6f}")
    print(f"Wilcoxon: W = {w_stat:.0f}, p = {p_wilcoxon:.6f}")
    print(f"TF wins: {tf_wins}/{n} folds")
    print()

    return {
        "experiment": experiment_name,
        "corpus": corpus_name,
        "tf_mean": tf_mean,
        "idyom_mean": idyom_mean,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "w_stat": w_stat,
        "p_wilcoxon": p_wilcoxon,
        "tf_wins": tf_wins,
    }


# ============================================================
# EXPERIMENT 1: Full-Window Comparison
# ============================================================

# Essen
tf_essen_exp1 = [2.462, 2.476, 2.508, 2.503, 2.461, 2.485, 2.494, 2.497, 2.500, 2.518]
idyom_essen_exp1 = [2.535226, 2.5390768, 2.5471635, 2.5566814, 2.5386233, 2.534133, 2.575533, 2.521926, 2.5589767, 2.573886]
# Meertens
tf_meertens_exp1 = [2.360, 2.398, 2.413, 2.412, 2.367, 2.384, 2.390, 2.425, 2.381, 2.407]
idyom_meertens_exp1 = [2.4172227, 2.4273055, 2.4213908, 2.4252439, 2.4228864, 2.4178965, 2.4315414, 2.431613, 2.428834, 2.4472625]

results = []
results.append(compare_models(tf_essen_exp1, idyom_essen_exp1, "1: Full-window", "Essen"))
results.append(compare_models(tf_meertens_exp1, idyom_meertens_exp1, "1: Full-window", "Meertens"))

# ============================================================
# EXPERIMENT 2: Sliding-Window Comparison
# ============================================================

# Essen
tf_essen_exp2    = [2.508, 2.525, 2.541, 2.555, 2.549, 2.516, 2.529, 2.526, 2.550, 2.552]
idyom_essen_exp2 = [2.6582701, 2.6666737, 2.6787922, 2.6879437, 2.638519, 2.6812217, 2.699566, 2.6540735, 2.6721344, 2.7025795]

# Meertens
tf_meertens_exp2    = [2.422, 2.449, 2.446, 2.453, 2.433, 2.438, 2.445, 2.476, 2.441, 2.476]
idyom_meertens_exp2 = [2.471999, 2.4768965, 2.4813156, 2.481897, 2.471321, 2.476943, 2.4784322, 2.4876518, 2.4770033, 2.5023453]

results.append(compare_models(tf_essen_exp2, idyom_essen_exp2, "2: Sliding-window", "Essen"))
results.append(compare_models(tf_meertens_exp2, idyom_meertens_exp2, "2: Sliding-window", "Meertens"))


# ============================================================
# EXPERIMENT 3: Cross-Corpus — per-melody IC comparison
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
IDYOM_CC = ROOT / 'results' / 'ic_values' / 'idyom' / 'cross_corpus'
TF_CC    = ROOT / 'results' / 'ic_values' / 'transformer' / 'cross_corpus'


def load_idyom_per_melody(dat_path):
    df = pd.read_csv(dat_path, sep=' ')
    # preserve melody order as it appears in the file
    return (df.groupby('melody.id', sort=False)['ic']
              .mean()
              .reset_index(drop=True)
              .rename('idyom_ic'))


def load_tf_per_melody(csv_path):
    df = pd.read_csv(csv_path)
    return df['mean_ic'].rename('tf_ic').reset_index(drop=True)


def compare_models_per_melody(tf_csv, idyom_dat, experiment_name, direction):
    tf_s    = load_tf_per_melody(tf_csv)
    idyom_s = load_idyom_per_melody(idyom_dat)

    # align by position — both files contain the same melodies in the same order
    assert len(tf_s) == len(idyom_s), (
        f"Row count mismatch: transformer={len(tf_s)}, idyom={len(idyom_s)}"
    )
    merged = pd.DataFrame({'tf_ic': tf_s, 'idyom_ic': idyom_s})

    tf    = merged['tf_ic'].values
    idyom = merged['idyom_ic'].values
    diff  = tf - idyom
    n     = len(diff)

    tf_mean    = np.mean(tf)
    idyom_mean = np.mean(idyom)
    delta      = np.mean(diff)

    t_stat, p_ttest     = ttest_rel(tf, idyom)
    w_stat, p_wilcoxon  = wilcoxon(tf, idyom)
    cohens_d            = delta / np.std(diff, ddof=1)
    se                  = np.std(diff, ddof=1) / np.sqrt(n)
    ci_low, ci_high     = t_dist.interval(0.95, df=n - 1, loc=delta, scale=se)
    tf_wins             = np.sum(diff < 0)

    print(f"{'='*60}")
    print(f"{experiment_name} — {direction}  (n={n} melodies)")
    print(f"{'='*60}")
    
    print(f"Transformer mean IC: {tf_mean:.4f} bits (std: {np.std(tf, ddof=1):.4f})")
    print(f"IDyOM mean IC: {idyom_mean:.4f} bits (std: {np.std(idyom, ddof=1):.4f})")
    
    print(f"Delta (TF - IDyOM): {delta:.4f} bits")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Cohen's d: {cohens_d:.3f}")
    
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_ttest:.6f}")
    print(f"Wilcoxon: W = {w_stat:.0f}, p = {p_wilcoxon:.6f}")
    print(f"TF wins: {tf_wins}/{n} melodies")
    
    print()

    return {
        "experiment": experiment_name,
        "direction": direction,
        "n": n,
        "tf_mean": tf_mean,
        "idyom_mean": idyom_mean,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "w_stat": w_stat,
        "p_wilcoxon": p_wilcoxon,
        "tf_wins": int(tf_wins),
    }


# Essen -> Meertens  (trained on Essen, evaluated on Meertens)
results.append(compare_models_per_melody(
    tf_csv   = TF_CC / 'full_window/cross_corpus_full_essen2meertens_per_melody_ic.csv',
    idyom_dat= IDYOM_CC / 'idyom_essen2meertens_both' / '8-cpitch-cpitch-7-nil-melody-nil-1-both-nil-t-nil-c-nil-t-t-x-3.dat',
    experiment_name='3: Cross-corpus Full-window',
    direction='Essen -> Meertens',
))

results.append(compare_models_per_melody(
    tf_csv   = TF_CC / 'sliding_window/cross_corpus_sliding_essen2meertens_per_melody_ic.csv',
    idyom_dat= IDYOM_CC / 'idyom_essen2meertens_ltm' / '8-cpitch-cpitch-7-nil-melody-nil-1-ltm-10-t-nil-c-nil-t-t-x-3.dat',
    experiment_name='3: Cross-corpus Sliding-window',
    direction='Essen -> Meertens',
))

# Meertens -> Essen  (trained on Meertens, evaluated on Essen)
results.append(compare_models_per_melody(
    tf_csv   = TF_CC / 'full_window/cross_corpus_full_meertens2essen_per_melody_ic.csv',
    idyom_dat= IDYOM_CC / 'idyom_meertens2essen_both' / '7-cpitch-cpitch-8-nil-melody-nil-1-both-nil-t-nil-c-nil-t-t-x-3.dat',
    experiment_name='3: Cross-corpus Full-window',
    direction='Meertens -> Essen',
))

results.append(compare_models_per_melody(
    tf_csv   = TF_CC / 'sliding_window/cross_corpus_sliding_meertens2essen_per_melody_ic.csv',
    idyom_dat= IDYOM_CC / 'idyom_meertens2essen_ltm' / '7-cpitch-cpitch-8-nil-melody-nil-1-ltm-10-t-nil-c-nil-t-t-x-3.dat',
    experiment_name='3: Cross-corpus Sliding-window',
    direction='Meertens -> Essen',
))


# ============================================================
# EXPERIMENT 4: Viewpoints Comparison
# ============================================================

# Essen
tf_essen_exp4    = [2.458, 2.490, 2.511, 2.496, 2.464, 2.478, 2.481, 2.476, 2.505, 2.492]
idyom_essen_exp4 = [2.334313, 2.355234, 2.3830264, 2.3754687, 2.3431766, 2.3740604, 2.3804896, 2.3442097, 2.3594937, 2.377519]

# Meertens
tf_meertens_exp4    = [2.385, 2.417, 2.410, 2.40, 2.406, 2.401, 2.403, 2.420, 2.406, 2.424]
idyom_meertens_exp4 = [2.2162623, 2.2283545, 2.220926, 2.2294137, 2.219569, 2.220671, 2.223932, 2.2306828, 2.2194073, 2.2388785]

results.append(compare_models(tf_essen_exp4, idyom_essen_exp4, "4: Viewpoints", "Essen"))
results.append(compare_models(tf_meertens_exp4, idyom_meertens_exp4, "4: Viewpoints", "Meertens"))
