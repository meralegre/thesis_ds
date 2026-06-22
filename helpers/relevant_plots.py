import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ==========================================
# Style configuration for double-column ACM 
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# colorblind-friendly palette
C_TRANSFORMER = '#2274A5'
C_IDYOM = '#D64933'
C_TRANSFORMER_LIGHT = '#94C5E0'
C_IDYOM_LIGHT = '#F0A899'

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

folds = np.arange(1, 11)

# ==========================================
# DATA — Per-fold IC values
# ==========================================
data = {
    'exp1': {  # Full-window
        'essen': {
            'transformer': [2.479, 2.479, 2.495, 2.502, 2.465, 2.483, 2.502, 2.487, 2.517, 2.514],
            'idyom':       [2.535, 2.539, 2.547, 2.557, 2.539, 2.534, 2.576, 2.522, 2.559, 2.574],
        },
        'meertens': {
            'transformer': [2.358, 2.408, 2.398, 2.400, 2.377, 2.393, 2.397, 2.432, 2.377, 2.414],
            'idyom':       [2.417, 2.427, 2.421, 2.425, 2.423, 2.418, 2.432, 2.432, 2.429, 2.447],
        },
    },
    'exp2': {  # Sliding-window
        'essen': {
            'transformer': [2.514, 2.540, 2.539, 2.534, 2.512, 2.548, 2.530, 2.514, 2.541, 2.532],
            'idyom':       [2.658, 2.667, 2.679, 2.688, 2.639, 2.681, 2.700, 2.654, 2.672, 2.703],
        },
        'meertens': {
            'transformer': [2.432, 2.457, 2.448, 2.451, 2.445, 2.442, 2.441, 2.472, 2.452, 2.472],
            'idyom':       [2.472, 2.477, 2.481, 2.482, 2.471, 2.477, 2.478, 2.488, 2.477, 2.502],
        },
    },
    'exp4': {  # Viewpoints
        'essen': {
            'transformer': [2.458, 2.481, 2.497, 2.479, 2.453, 2.465, 2.481, 2.481, 2.487, 2.480],
            'idyom':       [2.334, 2.355, 2.383, 2.375, 2.343, 2.374, 2.380, 2.344, 2.359, 2.378],
        },
        'meertens': {
            'transformer': [2.381, 2.405, 2.406, 2.408, 2.400, 2.400, 2.399, 2.416, 2.404, 2.413],
            'idyom':       [2.216, 2.228, 2.221, 2.229, 2.220, 2.221, 2.224, 2.231, 2.219, 2.239],
        },
    },
}


# ==========================================
# Per-fold IC line plots
# ==========================================
def plot_perfold(exp_key, exp_title, filename):
    """Line plot showing per-fold IC for both models on both corpora."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

    for ax, corpus, corpus_label in [(ax1, 'essen', 'Essen'), (ax2, 'meertens', 'Meertens')]:
        t_vals = data[exp_key][corpus]['transformer']
        i_vals = data[exp_key][corpus]['idyom']
        t_mean = np.mean(t_vals)
        i_mean = np.mean(i_vals)

        ax.plot(folds, t_vals, 'o-', color=C_TRANSFORMER, markersize=4,
                linewidth=1.5, label='Transformer', zorder=3)
        ax.plot(folds, i_vals, 's-', color=C_IDYOM, markersize=4,
                linewidth=1.5, label='IDyOM', zorder=3)

        # Mean lines
        ax.axhline(t_mean, color=C_TRANSFORMER, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(i_mean, color=C_IDYOM, linestyle='--', linewidth=0.8, alpha=0.5)

        ax.set_xlabel('Fold')
        ax.set_ylabel('Mean IC (bits)')
        ax.set_title(corpus_label)
        ax.set_xticks(folds)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Tight y-axis around the data range
        all_vals = t_vals + i_vals
        margin = (max(all_vals) - min(all_vals)) * 0.15
        ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

    ax1.legend(loc='best', framealpha=0.9)
    fig.suptitle(exp_title, fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


plot_perfold('exp1', 'Experiment 1: Full-Window vs Unbouded IDyOM', 'perfold_exp1.png')
plot_perfold('exp2', 'Experiment 2: Sliding-Window vs Order-Bounded IDyOM', 'perfold_exp2.png')
plot_perfold('exp4', 'Experiment 4: Viewpoint Selection vs Explicit Features', 'perfold_exp4.png')


# ==========================================
# Combined per-fold panel
# ==========================================
def plot_perfold_combined():
    """3x2 grid: rows = experiments, columns = corpora."""
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 7.5), sharex=True)

    experiments = [
        ('exp1', 'Exp. 1: Full-Window'),
        ('exp2', 'Exp. 2: Sliding-Window'),
        ('exp4', 'Exp. 4: Viewpoints'),
    ]
    corpora = [('essen', 'Essen'), ('meertens', 'Meertens')]

    for row, (exp_key, exp_label) in enumerate(experiments):
        for col, (corpus, corpus_label) in enumerate(corpora):
            ax = axes[row][col]
            t_vals = data[exp_key][corpus]['transformer']
            i_vals = data[exp_key][corpus]['idyom']
            t_mean = np.mean(t_vals)
            i_mean = np.mean(i_vals)

            ax.plot(folds, t_vals, 'o-', color=C_TRANSFORMER, markersize=3,
                    linewidth=1.2, label='Transformer')
            ax.plot(folds, i_vals, 's-', color=C_IDYOM, markersize=3,
                    linewidth=1.2, label='IDyOM')
            ax.axhline(t_mean, color=C_TRANSFORMER, linestyle='--', linewidth=0.7, alpha=0.5)
            ax.axhline(i_mean, color=C_IDYOM, linestyle='--', linewidth=0.7, alpha=0.5)

            all_vals = t_vals + i_vals
            margin = (max(all_vals) - min(all_vals)) * 0.15
            ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
            ax.set_xticks(folds)

            if row == 0:
                ax.set_title(corpus_label, fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{exp_label}\nIC (bits)', fontsize=8)
            if row == 2:
                ax.set_xlabel('Fold')

    # Single legend at bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f'{OUTPUT_DIR}/perfold_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_perfold_combined()


# ==========================================
# Summary IC bar chart (Results section)
# ==========================================
def plot_summary_bars():
    """Grouped bar chart comparing mean IC across all experiments."""
    configs = [
        ('Exp 1\nEssen',    2.492, 2.548),
        ('Exp 1\nMeertens', 2.395, 2.427),
        ('Exp 2\nEssen',    2.530, 2.674),
        ('Exp 2\nMeertens', 2.451, 2.481),
        ('Exp 3\nM→E Full', 2.648, 2.724),
        ('Exp 3\nM→E Slid', 2.795, 3.007),
        ('Exp 3\nE→M Full', 2.785, 2.682),
        ('Exp 3\nE→M Slid', 2.761, 2.895),
        ('Exp 4\nEssen',    2.476, 2.363),
        ('Exp 4\nMeertens', 2.403, 2.225),
    ]

    labels = [c[0] for c in configs]
    t_vals = [c[1] for c in configs]
    i_vals = [c[2] for c in configs]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bars_t = ax.bar(x - width/2, t_vals, width, label='Transformer',
                    color=C_TRANSFORMER, edgecolor='white', linewidth=0.5)
    bars_i = ax.bar(x + width/2, i_vals, width, label='IDyOM',
                    color=C_IDYOM, edgecolor='white', linewidth=0.5)

    # Highlight which model wins each comparison
    for idx in range(len(configs)):
        winner_bar = bars_t[idx] if t_vals[idx] < i_vals[idx] else bars_i[idx]
        winner_bar.set_edgecolor('black')
        winner_bar.set_linewidth(1.0)

    ax.set_ylabel('Mean IC (bits)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(2.0, 3.15)

    # Add vertical separators between experiment groups
    for sep in [1.5, 3.5, 7.5]:
        ax.axvline(sep, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/summary_ic_bars.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_summary_bars()


# ==========================================
# Cross-corpus degradation
# ==========================================
def plot_cross_corpus_degradation():
    """Bar chart showing IC degradation from within- to cross-corpus."""
    models = ['Transformer\n(Full)', 'IDyOM\n(LTM+STM)', 'Transformer\n(Sliding)', 'IDyOM\n(LTM b=10)']

    # Meertens -> Essen 
    within_essen = [2.492, 2.548, 2.530, 2.674]
    cross_me = [2.648, 2.724, 2.795, 3.007]

    # Essen -> Meertens 
    within_meertens = [2.395, 2.427, 2.451, 2.481]
    cross_em = [2.785, 2.682, 2.761, 2.895]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))
    x = np.arange(len(models))
    width = 0.3

    # Meertens -> Essen
    ax1.bar(x - width/2, within_essen, width, label='Within-corpus',
            color=C_TRANSFORMER_LIGHT, edgecolor=C_TRANSFORMER, linewidth=0.8)
    ax1.bar(x + width/2, cross_me, width, label='Cross-corpus',
            color=C_TRANSFORMER, edgecolor='white', linewidth=0.5)
    ax1.set_title('Meertens → Essen', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Mean IC (bits)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7)
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.set_ylim(2.2, 3.2)

    # Essen -> Meertens
    ax2.bar(x - width/2, within_meertens, width, label='Within-corpus',
            color=C_IDYOM_LIGHT, edgecolor=C_IDYOM, linewidth=0.8)
    ax2.bar(x + width/2, cross_em, width, label='Cross-corpus',
            color=C_IDYOM, edgecolor='white', linewidth=0.5)
    ax2.set_title('Essen → Meertens', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Mean IC (bits)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=7)
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.set_ylim(2.2, 3.2)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cross_corpus_degradation.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_cross_corpus_degradation()


# ==========================================
# IC difference across experiments (Main)
# ==========================================

def plot_ic_difference():
    """Dot plot showing IC difference (IDyOM - Transformer) per fold.
    Positive = Transformer wins. Negative = IDyOM wins."""
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6), sharex=True)

    experiments = [
        ('exp1', 'Exp. 1: Full-Window'),
        ('exp2', 'Exp. 2: Sliding-Window'),
        ('exp4', 'Exp. 4: Viewpoints'),
    ]
    corpora = [('essen', 'Essen'), ('meertens', 'Meertens')]

    for row, (exp_key, exp_label) in enumerate(experiments):
        for col, (corpus, corpus_label) in enumerate(corpora):
            ax = axes[row][col]
            t_vals = np.array(data[exp_key][corpus]['transformer'])
            i_vals = np.array(data[exp_key][corpus]['idyom'])
            diff = i_vals - t_vals  # positive = Transformer wins

            colors = [C_TRANSFORMER if d > 0 else C_IDYOM for d in diff]
            ax.bar(folds, diff, color=colors, edgecolor='white', linewidth=0.3, width=0.6)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axhline(np.mean(diff), color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

            ax.set_xticks(folds)
            if row == 0:
                ax.set_title(corpus_label, fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{exp_label}\nΔIC (bits)', fontsize=7.5)
            if row == 2:
                ax.set_xlabel('Fold')

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_TRANSFORMER, label='Transformer advantage'),
        Patch(facecolor=C_IDYOM, label='IDyOM advantage'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.02), framealpha=0.9, fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f'{OUTPUT_DIR}/ic_difference_perfold.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_ic_difference()

# Summary
print('\n=== All figures generated ===')
print(f'Output directory: {OUTPUT_DIR}/')
