import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt
from pathlib import Path
from music21 import pitch as m21pitch

ROOT = Path(__file__).resolve().parent.parent


def midi_to_name(midi_pitch):
    """Convert MIDI pitch number to note name using music21"""
    return m21pitch.Pitch(midi=int(midi_pitch)).nameWithOctave

# =============================================================
# DATA LOADING
# =============================================================

# Human Expectation Ratings
df = pd.read_csv(ROOT / 'data' / 'PearceEtAl2010.dat')

print(f"Total rows: {len(df)}")
print(f"Participants: {df['subject'].nunique()}")
print(f"Unique melodies: {df['melody'].nunique()}")
print(f"Probes per melody: {df.groupby('melody')['note'].nunique().values}")


# Load IDyOM IC values
IDYOM_DIR = ROOT / 'results' / 'ic_values' / 'idyom' / 'human_ratings'

idyom_models = {
    'idyom_both_unbounded': IDYOM_DIR / 'hymn_idyom_both_unbounded' /
        'full_window_1.dat',
    'idyom_ltm_ob10': IDYOM_DIR / 'hymn_idyom_ltm_ob10' /
        'sliding_window_1.dat',
    'idyom_ltm_ob10_viewpoints': IDYOM_DIR / 'hymn_idyom_ltm_ob10_viewpoints' /
        'viewpoints_1.dat',
}


def load_idyom(path):
    """Load IDyOM .dat file with predicted pitch extracted from distribution columns."""
    idyom = pd.read_csv(path, sep=' ')
    idyom['melody_name'] = idyom['melody.name'].str.strip('"')
 
    # Extract predicted pitch from cpitch.XX distribution columns
    pitch_cols = sorted(
        [c for c in idyom.columns if c.startswith('cpitch.') and c.split('.')[1].isdigit()],
        key=lambda c: int(c.split('.')[1])
    )
    pitch_values = np.array([int(c.split('.')[1]) for c in pitch_cols])
 
    if len(pitch_cols) > 0:
        probs = idyom[pitch_cols].values
        idyom['predicted_pitch'] = pitch_values[np.argmax(probs, axis=1)]
        idyom['correct_pitch'] = idyom['predicted_pitch'] == idyom['cpitch'].astype(int)
 
        cols = ['melody_name', 'note.id', 'information.gain', 'ic',
                'predicted_pitch', 'correct_pitch']
    else:
        cols = ['melody_name', 'note.id', 'information.gain', 'ic']
 
    return idyom[cols].rename(
        columns={'melody_name': 'melody', 'note.id': 'note', 'ic': 'idyom_ic'}
    )


# Load Transformer IC values
TRANSFORMER_DIR = ROOT / 'results' / 'ic_values' / 'transformer' / 'human_ratings'

transformer_models = {
    'transformer_full': TRANSFORMER_DIR / 'full_window/hymn_full_per_note_ic.csv',
    'transformer_sliding': TRANSFORMER_DIR / 'sliding_window/hymn_sliding_per_note_ic.csv',
    'transformer_viewpoints': TRANSFORMER_DIR / 'viewpoints/hymn_viewpoints_per_note_ic.csv',
}


# =============================================================
# MEAN EXPECTEDNESS PER PROBE (across participants)
# =============================================================
probes = (df
    .groupby(['melody', 'note', 'pitch'])
    .agg(
        mean_rating=('response', 'mean'),
        sd_rating=('response', 'std'),
        n_participants=('response', 'count'),
    )
    .reset_index()
    .sort_values(['melody', 'note'])
    .reset_index(drop=True)
)

print(f"\n{len(probes)} probes found")
print(probes[['melody', 'note', 'pitch', 'mean_rating', 'n_participants']].to_string(index=False))
print(f"\nRating range: {probes['mean_rating'].min():.2f} – {probes['mean_rating'].max():.2f}")
print(f"Rating mean: {probes['mean_rating'].mean():.2f}")


# =============================================================
# IDyOM and Transformer IC dataframes
# =============================================================
idyom_dfs = {name: load_idyom(path) for name, path in idyom_models.items()}

def load_transformer(path):
    """Load Transformer IC CSV with note offset to align with ratings indexing.
    The Transformer's output at position N predicts note N+1 in the ratings data,
    so we shift note indices by +1 before merging."""
    df = pd.read_csv(path).rename(columns={'ic': 'transformer_ic'})
    df['note'] = df['note'] + 1  # align to ratings indexing (N-1 offset)
    return df

transformer_dfs = {
    name: load_transformer(path)
    for name, path in transformer_models.items()
}


# =============================================================
# MERGE IC INTO PROBES AND COMPUTE CORRELATIONS
# =============================================================
results = []


def merge_ic(probes, ic_df, ic_col, model_name):
    # Pick which columns to merge
    merge_cols = ['melody', 'note', ic_col]
    for extra in ['predicted_pitch', 'correct_pitch']:
        if extra in ic_df.columns:
            merge_cols.append(extra)

    merged = probes.merge(
        ic_df[merge_cols],
        on=['melody', 'note'],
        how='left'
    )
    n_missing = merged[ic_col].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} probes missing IC for {model_name}")
    return merged


print("\n" + "="*60)
print("CORRELATIONS: model IC vs mean expectedness rating")
print("="*60)

# All IDyOM models
for model_name, ic_df in idyom_dfs.items():
    merged = merge_ic(probes, ic_df, 'idyom_ic', model_name)
    # print("columns", merged.columns)
    valid = merged.dropna(subset=['idyom_ic', 'mean_rating'])
    r_p, p_p = pearsonr(valid['idyom_ic'], valid['mean_rating'])
    r_s, p_s = spearmanr(valid['idyom_ic'], valid['mean_rating'])
    print(f"\n{model_name} (n={len(valid)}):")
    print(f"Pearson  r = {r_p:.3f}, p = {p_p:.4f}")
    print(f"Spearman r = {r_s:.3f}, p = {p_s:.4f}")
    results.append({'model': model_name,
                    'pearson_r': r_p,
                    'pearson_p': p_p,
                    'spearman_r': r_s,
                    'spearman_p': p_s,
                    'n': len(valid)})
    probes[model_name + '_ic'] = merged['idyom_ic'].values

    # Store predicted pitch and correctness per IDyOM config
    probes[model_name + '_predicted_pitch'] = merged['predicted_pitch'].values
    probes[model_name + '_correct_pitch'] = merged['correct_pitch'].values


# All transformer models
for model_name, ic_df in transformer_dfs.items():
    merged = merge_ic(probes, ic_df, 'transformer_ic', model_name)
    valid = merged.dropna(subset=['transformer_ic', 'mean_rating'])
    r_p, p_p = pearsonr(valid['transformer_ic'], valid['mean_rating'])
    r_s, p_s = spearmanr(valid['transformer_ic'], valid['mean_rating'])
    print(f"\n{model_name} (n={len(valid)}):")
    print(f"Pearson  r = {r_p:.3f}, p = {p_p:.4f}")
    print(f"Spearman r = {r_s:.3f}, p = {p_s:.4f}")
    results.append({'model': model_name,
                    'pearson_r': r_p,
                    'pearson_p': p_p,
                    'spearman_r': r_s,
                    'spearman_p': p_s,
                    'n': len(valid)})
    probes[model_name + '_ic'] = merged['transformer_ic'].values

    # Store predicted pitch and correctedness per Transformer config
    probes[model_name + '_predicted_pitch'] = merged['predicted_pitch'].values
    probes[model_name + '_correct_pitch'] = merged['correct_pitch'].values

results_df = pd.DataFrame(results).sort_values('pearson_r', ascending=False)
print("\n\nSUMMARY:")
print(results_df.to_string(index=False))


# =============================================================
# PREDICTION ACCURACY vs HUMAN EXPECTEDNESS
# =============================================================
print("\n" + "="*60)
print("PREDICTION ACCURACY vs HUMAN EXPECTEDNESS")
print("="*60)
 
all_models = list(idyom_models.keys()) + list(transformer_models.keys())
 
for model_name in all_models:
    correct_col = model_name + '_correct_pitch'
    pred_col = model_name + '_predicted_pitch'
 
    if correct_col not in probes.columns:
        continue
 
    valid = probes.dropna(subset=[correct_col])
    right = valid[valid[correct_col].astype(bool)]
    wrong = valid[~valid[correct_col].astype(bool)]
 
    print(f"\n{model_name}:")
    print(f"Correctly predicted probes: {len(right)}/{len(valid)} ({len(right)/len(valid):.0%})")
 
    if len(right) > 0 and len(wrong) > 0:
        print(f"Correct ({len(right)} probes): mean rating = {right['mean_rating'].mean():.2f} "
              f"(SD = {right['mean_rating'].std():.2f})")
        print(f"Wrong ({len(wrong)} probes): mean rating = {wrong['mean_rating'].mean():.2f} "
              f"(SD = {wrong['mean_rating'].std():.2f})")
 
        t, p = ttest_ind(right['mean_rating'], wrong['mean_rating'])
        print(f"t-test: t = {t:.3f}, p = {p:.4f}")
 
        if p < 0.05:
            direction = "more" if t > 0 else "less"
            print(f" - Correctly predicted notes were rated significantly {direction} expected")
        else:
            print(f" - No significant difference (p = {p:.3f})")
 
    # Detailed probe table for this model
    if pred_col in probes.columns:
        detail = valid[['melody', 'note', 'pitch', pred_col, correct_col, 'mean_rating']].copy()
        detail['actual_note'] = detail['pitch'].apply(midi_to_name)
        detail['predicted_note'] = detail[pred_col].apply(midi_to_name)
        detail = detail.sort_values('mean_rating')
        print(f"\n  Probe details (sorted by human rating):")
        print(detail[['melody', 'note', 'actual_note', 'predicted_note',
                       correct_col, 'mean_rating']].to_string(index=False))


# =============================================================
# SAVE ENRICHED PROBE TABLE
# =============================================================
probes.to_csv(ROOT / 'results' / 'probe_data_paired.csv', index=False)
print("\nSaved probe_data_paired.csv")


# =============================================================
# PLOTS: IC vs MEAN EXPECTEDNESS (per model)
# =============================================================
model_columns = (
    [(n + '_ic', n) for n in idyom_models] +
    [(n + '_ic', n) for n in transformer_models]
)

colors = {
    'idyom_both_unbounded': '#1f77b4',
    'idyom_ltm': '#aec7e8',
    'idyom_ltm_viewpoints': '#17becf',
    'transformer_full': '#d62728',
    'transformer_sliding': '#ff9896',
    'transformer_viewpoints': '#e377c2',
}

fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
axes = axes.flatten()

for ax, (ic_col, model_name) in zip(axes, model_columns):
    valid = probes.dropna(subset=[ic_col, 'mean_rating'])
    ax.scatter(valid[ic_col], valid['mean_rating'],
               color=colors.get(model_name, 'steelblue'), alpha=0.75, s=50)
    # regression line
    m, b = np.polyfit(valid[ic_col], valid['mean_rating'], 1)
    x_range = np.linspace(valid[ic_col].min(), valid[ic_col].max(), 100)

    ax.plot(x_range, m * x_range + b, color='black', linewidth=1, linestyle='--')
    r_row = results_df[results_df['model'] == model_name].iloc[0]

    ax.set_title(f"{model_name}\nr={r_row['pearson_r']:.3f}, p={r_row['pearson_p']:.3f}",
                 fontsize=9)
    ax.set_xlabel('IC (bits)', fontsize=8)
    ax.set_ylabel('Mean Expectedness Rating', fontsize=8)

fig.suptitle('Model IC vs Human Expectedness Ratings (Pearce et al. 2010)', fontsize=11)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'human_ratings_ic_comparison.png', dpi=150)
plt.show()
