import numpy as np
from scipy.stats import ttest_rel, wilcoxon,
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

    print(f"Δ (TF - IDyOM): {delta:.3f} bits")
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
tf_essen_exp1 = [2.479, 2.479, 2.495, 2.502, 2.465, 2.483, 2.502, 2.487, 2.517, 2.514]
idyom_essen_exp1 = [2.535226, 2.5390768, 2.5471635, 2.5566814, 2.5386233, 2.534133, 2.575533, 2.521926, 2.5589767, 2.573886]

# Meertens
tf_meertens_exp1 = [2.358, 2.408, 2.398, 2.400, 2.377, 2.393, 2.397, 2.432, 2.377, 2.414]
idyom_meertens_exp1 = [2.4172227, 2.4273055, 2.4213908, 2.4252439, 2.4228864, 2.4178965, 2.4315414, 2.431613, 2.428834, 2.4472625]

results = []

results.append(compare_models(tf_essen_exp1, idyom_essen_exp1, "1: Full-window", "Essen"))
results.append(compare_models(tf_meertens_exp1, idyom_meertens_exp1, "1: Full-window", "Meertens"))
