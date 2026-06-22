# A Comparative Study of Neural Networks and IDyOM's Performance for Modeling Musical Expectation

Master's thesis — Data Science, 2025–2026

This repository contains all code and results for a study comparing Transformer-based neural networks against IDyOM (Information Dynamics of Music) for predicting melodic expectation. The models are trained and evaluated on two folk music corpora (Essen and Meertens) and validated against human expectedness ratings (Pearce et al., 2010).

---

## Repository Structure

```
thesis_ds/
├── data/                        Preprocessed files (raw corpora not included)
│   ├── essen_unique_melodies.csv
│   ├── essen_unique_meta_melodies.csv
│   ├── essen_fold_assignments.csv
│   ├── meertens_unique_melodies.csv
│   ├── meertens_unique_meta_melodies.csv
│   ├── meertens_fold_assignments.csv
│   ├── hymns.lisp               IDyOM lisp export of hymn melodies
│   └── PearceEtAl2010.dat       Human expectedness ratings (Pearce et al., 2010)
│
├── transformers/                Main training scripts 
│   ├── full_window_transformer.py
│   ├── sliding_window_transformer.py
│   └── viewpoints_transformer.py
│
├── helpers/                          Shared utility modules
│   ├── statistical_test.py           Paired tests and cross-corpus comparisons
│   ├── filter_meertens_durations.py
│   ├── nill_checking.py
│   └── relevant_plots.py
│
├── analysis/                    Post-training evaluation and plotting
│   ├── load_run_models.py       Load a pretrained model and compute IC on new data
│   ├── human_ratings_comparison_pipeline.py
│   ├── probe_expectedness_ratings.py
│   └── staff_plots.py
│
├── eda/                         Exploratory data analysis
│   ├── eda_pipeline.py
│   ├── eda_preprocessing.ipynb
│   └── list_polyphonic.py
│
├── notebooks/
│   ├── full_window_transformer.ipynb
│   ├── sliding_window_transformer.ipynb
│   └── viewpoints_transformer.ipynb
│
├── results/
│   ├── ic_values/
│   │   ├── transformer/         Per-note and per-melody IC CSVs from transformer models
│   │   │   ├── cross_corpus/
│   │   │   └── human_ratings/
│   │   └── idyom/               IDyOM .dat output files
│   │       ├── cross_corpus/
│   │       └── human_ratings/
│   ├── hymn_comparisons/        Per-melody IC comparison CSVs
│   └── probe_data_paired.csv    Merged IC + human ratings table
│
├── figures/                     Generated plots
└── jobs/                        HPC job output logs (.out files)
```

---

## Requirements

Python 3.10+ is recommended. Install dependencies with:

```bash
pip install keras numpy pandas scikit-learn scipy matplotlib music21
```

A GPU with CUDA support is recommended for training. The scripts set Keras environment flags for GPU compatibility automatically.

---

## Models

Three transformer architectures are implemented, all trained to predict the next pitch in a melody sequence:

| Script | Architecture | IDyOM analogue |
|---|---|---|
| `full_window_transformer.py` | Causal self-attention over the full melody | IDyOM `both` (unbounded) |
| `sliding_window_transformer.py` | Sliding-window attention (fixed context) | IDyOM `ltm` with order bound 10 |
| `viewpoints_transformer.py` | Multi-viewpoint input (pitch, interval, tonal) | IDyOM viewpoint combination |

---

## Usage

All transformer scripts are run from inside the `transformers/` directory. Each takes a positional `experiment` argument.

### Training from scratch

**Full-window transformer:**
```bash
cd transformers
python full_window_transformer.py kfold_essen
python full_window_transformer.py kfold_meertens
python full_window_transformer.py cross_essen2meertens
python full_window_transformer.py cross_meertens2essen
```

**Sliding-window transformer:**
```bash
cd transformers
python sliding_window_transformer.py kfold_essen
python sliding_window_transformer.py kfold_meertens
python sliding_window_transformer.py cross_essen2meertens
python sliding_window_transformer.py cross_meertens2essen
```

**Viewpoints transformer:**
```bash
cd transformers
python viewpoints_transformer.py kfold_essen
python viewpoints_transformer.py kfold_meertens
python viewpoints_transformer.py full_essen
```

All experiments default to saving models under `../pretrained_models/`. Override with `--save-models-dir`.

### Computing IC on hymns (human ratings experiment)

```bash
cd transformers
python full_window_transformer.py hymn_ic --trained-model-dir ../pretrained_models/full_window/full_window_essen
python sliding_window_transformer.py hymn_ic --trained-model-dir ../pretrained_models/sliding_window/sliding_essen
python viewpoints_transformer.py hymn_ic --trained-model-dir ../pretrained_models/viewpoints/full_viewpoints_essen
```

### Loading a pretrained model on new data

```bash
python analysis/load_run_models.py pretrained_models/full_window/full_window_essen \
    --model-type full \
    data/essen_unique_melodies.csv
```

### Human ratings analysis

```bash
# Correlation of IC values with Pearce et al. (2010) ratings
python analysis/probe_expectedness_ratings.py

# Staff notation plots with IC overlay
python analysis/staff_plots.py --experiment full
```

### Statistical tests (cross-corpus comparison)

```bash
python helpers/statistical_test.py
```

---

## Data

The raw corpora are not included in this repository. The preprocessed files required to run all experiments are included in `data/`:

| File | Description |
|---|---|
| `essen_unique_melodies.csv` | Deduplicated pitch sequences from the Essen Folksong Collection |
| `essen_unique_meta_melodies.csv` | Melody metadata (filename, path) for IDyOM alignment |
| `essen_fold_assignments.csv` | 10-fold split assignments used for k-fold evaluation |
| `meertens_unique_melodies.csv` | Deduplicated pitch sequences from the Meertens Tune Collection |
| `meertens_unique_meta_melodies.csv` | Melody metadata for IDyOM alignment |
| `meertens_fold_assignments.csv` | 10-fold split assignments |
| `hymns.lisp` | IDyOM lisp export of hymn melodies used in the human ratings experiment |
| `PearceEtAl2010.dat` | Human melodic expectedness ratings (Pearce et al., 2010) |

The original corpora can be obtained from:
- **Essen Folksong Collection**: Schaffrath (1995) — [github.com/ccarh/essen-folksong-collection](https://github.com/ccarh/essen-folksong-collection)
- **Meertens Tune Collection**: [liederenbank.nl/mtc](https://www.liederenbank.nl/mtc/)

---

## Results

Transformer IC values and comparison outputs are included in the repository:

- `results/ic_values/transformer/` — per-note and per-melody IC for all transformer experiments
- `results/hymn_comparisons/` — per-melody IC comparison tables used for figures

The following are not included due to file size but are available on request:
- Pretrained `.keras` model weights (`pretrained_models/`)
- IDyOM raw output files (`results/ic_values/idyom/`)

---

## Reference

Pearce, M. T., Ruiz, M. H., Kapasi, S., Wiggins, G. A., & Bhattacharya, J. (2010).
Unsupervised statistical learning underpins computational, behavioural, and neural manifestations of musical expectation.
*NeuroImage*, 50(1), 302–313.


