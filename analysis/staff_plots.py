import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch
import matplotlib.lines as mlines
import numpy as np
from music21 import pitch as m21pitch

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ========================================
# CONFIG
# ========================================

RATINGS_PATH = ROOT / 'data' / 'PearceEtAl2010.dat'

# IDyOM configs
IDYOM_DIR = ROOT / 'results' / 'ic_values' / 'idyom' / 'human_ratings'

IDYOM_CONFIGS = {
    'full': IDYOM_DIR / 'hymn_idyom_both_unbounded' /
        'full_window_1.dat',
    'sliding': IDYOM_DIR / 'hymn_idyom_ltm_ob10' /
        'sliding_window_1.dat',
    'viewpoints': IDYOM_DIR / 'hymn_idyom_ltm_ob10_viewpoints' /
        'viewpoints_1.dat',
}

# Transformer configs
TRANSFORMER_DIR = ROOT / 'results' / 'ic_values' / 'transformer' / 'human_ratings'

TRANSFORMER_CONFIGS = {
    'full': TRANSFORMER_DIR / 'full_window/hymn_full_per_note_ic.csv',
    'sliding': TRANSFORMER_DIR / 'sliding_window/hymn_sliding_per_note_ic.csv',
    'viewpoints': TRANSFORMER_DIR / 'viewpoints/hymn_viewpoints_per_note_ic.csv',
}

EXPERIMENT_LABELS = {
    'full': 'Full-Window',
    'sliding': 'Sliding-Window',
    'viewpoints': 'Viewpoints',
}

OUTPUT_DIR = ROOT / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


# ========================================
# DATA LOADING
# ========================================
 
def load_idyom_predictions(dat_path):
    """
    Load IDyOM output file and extract predicted pitch (argmax of cpitch.XX
    columns) for every note of every melody.
    """
    df = pd.read_csv(dat_path, sep=' ')
    df['melody'] = df['melody.name'].str.strip('"')
 
    # Find pitch distribution columns
    pitch_cols = sorted(
        [c for c in df.columns if c.startswith('cpitch.') and c.split('.')[1].isdigit()],
        key=lambda c: int(c.split('.')[1])
    )
    pitch_values = np.array([int(c.split('.')[1]) for c in pitch_cols])
 
    # Argmax = predicted pitch
    probs_matrix = df[pitch_cols].values
    predicted_indices = np.argmax(probs_matrix, axis=1)
    df['idyom_predicted_pitch'] = pitch_values[predicted_indices]
    df['idyom_correct'] = df['idyom_predicted_pitch'] == df['cpitch'].astype(int)
 
    # Get IDyOM IC column
    df['idyom_ic'] = df['ic']
 
    melodies = {}
    for name, group in df.groupby('melody'):
        group = group.sort_values('note.id').reset_index(drop=True)
        melodies[name] = group[
          ['note.id', 'cpitch', 'idyom_predicted_pitch','idyom_correct', 'idyom_ic']
        ].copy()
 
    return melodies
 
 
def load_transformer_predictions(csv_path):
    """Load Transformer predictions CSV"""
    if csv_path is None or not Path(csv_path).exists():
        return None
 
    df = pd.read_csv(csv_path)
    mel_col = 'melody_name' if 'melody_name' in df.columns else 'melody'
    df = df.rename(columns={mel_col: 'melody'})
 
    melodies = {}
    for name, group in df.groupby('melody'):
        group = group.sort_values('note').reset_index(drop=True)
        melodies[name] = group
 
    return melodies
 
 
def load_probe_info():
    """Load human ratings and return probe position per melody."""
    df = pd.read_csv(RATINGS_PATH)
    probes = (
        df.groupby(['melody', 'note', 'pitch'])
        .agg(mean_rating=('response', 'mean'))
        .reset_index()
    )
    # dict: melody_name -> {note_position, pitch, mean_rating}
    probe_map = {}
    for _, row in probes.iterrows():
        probe_map[row['melody']] = {
            'position': int(row['note']),
            'pitch': int(row['pitch']),
            'mean_rating': row['mean_rating'],
        }
    return probe_map
 
 
def build_staff_data(
    melody_name,
    idyom_melodies,
    transformer_melodies,
    probe_map
):
    """
    Build the data structures the staff drawing function needs for one melody.
    """
    if melody_name not in idyom_melodies:
        raise ValueError(f"Melody '{melody_name}' not found in IDyOM data. "
                         f"Available: {list(idyom_melodies.keys())}")
 
    idyom_df = idyom_melodies[melody_name]
 
    # IDyOM melody pitches (cpitch column)
    melody_pitches = idyom_df['cpitch'].astype(int).tolist()
 
    # Probe position for this melody
    probe = probe_map.get(melody_name)
    probe_pos = probe['position'] if probe else None
 
    # Build predictions list and show at probe position
    predictions = []
    all_idyom_preds = idyom_df['idyom_predicted_pitch'].astype(int).tolist()
 
    # Transformer predictions
    all_trans_preds = None
    if transformer_melodies and melody_name in transformer_melodies:
        trans_df = transformer_melodies[melody_name]
        all_trans_preds = [None] * len(melody_pitches)
 
        for _, row in trans_df.iterrows():
            # Transformer note N corresponds to IDyOM note N+1
            idyom_pos = int(row['note']) + 1
            if 'predicted_pitch' in row and pd.notna(row['predicted_pitch']):
              # Convert to 0-indexed
              idx = idyom_pos - 1
              if 0 <= idx < len(all_trans_preds):
                all_trans_preds[idx] = int(row['predicted_pitch'])
 
    # Build the predictions tuples for all positions
    for i in range(len(melody_pitches)):
        pos = i + 1
        actual = melody_pitches[i]
        idyom_pred = all_idyom_preds[i] if i < len(all_idyom_preds) else actual
 
        if all_trans_preds and all_trans_preds[i] is not None:
            trans_pred = all_trans_preds[i]
        else:
            # if no prediction available, show as correct
            trans_pred = actual
 
        predictions.append((pos, actual, idyom_pred, trans_pred))
 
    return melody_pitches, predictions, probe


# ============================================================================
# MUSIC NOTATION CONSTANTS
# ============================================================================
 
def midi_to_name(midi_pitch):
    """Convert MIDI pitch number to note name using music21"""
    return m21pitch.Pitch(midi=int(midi_pitch)).nameWithOctave
 
 
# Treble clef staff, lines are E4, G4, B4, D5, F5
STAFF_LINE_PITCHES = [64, 67, 71, 74, 77]
 
 
def midi_to_staff_y(midi_pitch):
    """
    Convert MIDI pitch to vertical staff position using music21's diatonic number
    """
    p = m21pitch.Pitch(midi=int(midi_pitch))
    return p.diatonicNoteNum - 29
 
 
# Staff line positions in diatonic steps from C4
STAFF_LINE_Y = [midi_to_staff_y(p) for p in STAFF_LINE_PITCHES]
 
 
def ledger_lines(y_pos):
    """Return list of y-positions where ledger lines are needed."""
    ledgers = []
    # Below staff, ledger at C4 (y=0), A3 (y=-2)
    if y_pos <= STAFF_LINE_Y[0] - 2:
        y = STAFF_LINE_Y[0] - 2
        while y >= y_pos - 0.5:
            ledgers.append(y)
            y -= 2
    # Above staff, ledger at A5 (y=12), C6 (y=14)
    if y_pos >= STAFF_LINE_Y[-1] + 2:
        y = STAFF_LINE_Y[-1] + 2
        while y <= y_pos + 0.5:
            ledgers.append(y)
            y += 2
    # Special case: middle C (y=0) always gets a ledger
    if abs(y_pos - 0) < 0.5:
        if 0 not in ledgers:
            ledgers.append(0)
    return ledgers


  # ============================================================================
# DRAWING FUNCTIONS
# ============================================================================
 
def draw_staff(ax, x_start, x_end, y_scale=0.3):
    """Draw five staff lines."""
    for y in STAFF_LINE_Y:
        ax.plot([x_start, x_end], [y * y_scale, y * y_scale],
                color="#2c3e50", linewidth=0.8, zorder=1)
 
 
def draw_notehead(ax, x, y, y_scale=0.3, color="black", alpha=1.0,
                  filled=True, size=0.38, zorder=5):
    """Draw an oval notehead at position (x, y)."""
    notehead = Ellipse(
        (x, y * y_scale), width=size, height=0.22 * y_scale * 7,
        angle=-10, facecolor=color if filled else "white",
        edgecolor=color, linewidth=1.2, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(notehead)
 
 
def draw_ledger_lines(ax, x, y_pos, y_scale=0.3):
    """Draw ledger lines if the note is outside the staff."""
    ledgers = ledger_lines(y_pos)
    for ly in ledgers:
        ax.plot([x - 0.25, x + 0.25], [ly * y_scale, ly * y_scale],
                color="#2c3e50", linewidth=0.8, zorder=2)
 
 
def draw_stem(ax, x, y, y_scale=0.3, color="black", alpha=1.0):
    """Draw a note stem."""
    # B4 and above: stem down
    stem_dir = -1 if y >= 6 else 1
    stem_length = 3.5 * y_scale
    x_off = 0.18 if stem_dir == 1 else -0.18
    y_start = y * y_scale
    y_end = y_start + stem_dir * stem_length
    ax.plot([x + x_off, x + x_off], [y_start, y_end],
            color=color, linewidth=1.0, alpha=alpha, zorder=4)
 
 
def draw_accidental(ax, x, midi_pitch, y_scale=0.3, color="black", alpha=1.0):
    """Draw an accidental (sharp/flat) if the pitch has one, using music21."""
    p = m21pitch.Pitch(midi=int(midi_pitch))
    if p.accidental is not None and p.accidental.alter != 0:
        symbol = p.accidental.unicode
        y = midi_to_staff_y(midi_pitch) * y_scale
        ax.text(x - 0.3, y, symbol, fontsize=10, ha="center", va="center",
                color=color, alpha=alpha, zorder=5, fontweight="bold")
 
 
def draw_treble_clef(ax, x, y_scale=0.3):
    """Draw a stylized treble clef using basic drawing primitives."""
    # Use a simple "G" + curly line to represent treble clef
    y_g = STAFF_LINE_Y[1] * y_scale
    ax.text(x, y_g + 0.15, "G", fontsize=16, ha="center", va="center",
            color="#2c3e50", zorder=10, fontfamily="serif",
            fontweight="bold", fontstyle="italic")
    # Small vertical line through it
    ax.plot([x, x],
            [STAFF_LINE_Y[0] * y_scale - 0.15, STAFF_LINE_Y[-1] * y_scale + 0.15],
            color="#2c3e50", linewidth=1.2, zorder=9, alpha=0.4)


# ============================================================================
# MAIN FIGURE
# ============================================================================
 
def create_staff_figure(
    melody,
    predictions=None,
    all_idyom=None,
    all_transformer=None,
    show_all=False,
    probe_pos=None,
    output_png="staff.png",
    title="Hymn Melody: Actual vs Predicted Notes",
):
    n = len(melody)
    y_scale = 0.3
    note_spacing = 1.0
    margin_left = 1.5
    margin_right = 0.8
 
    fig_width = max(12, margin_left + n * note_spacing + margin_right + 1)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
 
    x_start = 0.2
    x_end = margin_left + n * note_spacing + 0.5
 
    # Draw staff
    draw_staff(ax, x_start, x_end, y_scale)
 
    # Draw treble clef
    draw_treble_clef(ax, 0.7, y_scale)
 
    # Prediction lookup
    pred_lookup = {}
    if show_all and all_idyom and all_transformer:
        for i in range(n):
            pred_lookup[i + 1] = (all_idyom[i], all_transformer[i])
    elif predictions:
        for pos, actual, idyom_p, trans_p in predictions:
            pred_lookup[pos] = (idyom_p, trans_p)
 
    # Draw each note
    for i, pitch in enumerate(melody):
        x = margin_left + i * note_spacing
        y = midi_to_staff_y(pitch)
 
        is_pred_pos = (i + 1) in pred_lookup
 
        # Draw ledger lines for actual note
        draw_ledger_lines(ax, x, y, y_scale)
 
        if is_pred_pos:
            idyom_pred, trans_pred = pred_lookup[i + 1]
            y_idyom = midi_to_staff_y(idyom_pred)
            y_trans = midi_to_staff_y(trans_pred)
 
            # Light background highlight for prediction positions
            is_probe = probe_pos is not None and (i + 1) == probe_pos
            bg_color = "#fff3cd" if is_probe else "#f8f9fa"
            edge_color = "#f0ad4e" if is_probe else "#dee2e6"
            edge_width = 1.2 if is_probe else 0.5
 
            rect = FancyBboxPatch(
                (x - 0.35, min(STAFF_LINE_Y) * y_scale - 0.3),
                0.7, (max(STAFF_LINE_Y) - min(STAFF_LINE_Y)) * y_scale + 0.6,
                boxstyle="round,pad=0.05", facecolor=bg_color,
                edgecolor=edge_color, linewidth=edge_width, zorder=0, alpha=0.7,
            )
            ax.add_patch(rect)
 
            if is_probe:
                ax.text(x, max(STAFF_LINE_Y) * y_scale + 1.2,
                        "▼ probe", fontsize=7, ha="center", va="bottom",
                        color="#d35400", fontweight="bold")
 
            # Draw predicted noteheads
            # IDyOM prediction (in blue), offset slightly left
            if idyom_pred != pitch:
                draw_ledger_lines(ax, x - 0.12, y_idyom, y_scale)
                draw_notehead(ax, x - 0.12, y_idyom, y_scale,
                              color="#3498db", alpha=0.7, size=0.32, zorder=3)
                draw_accidental(ax, x - 0.12, idyom_pred, y_scale,
                           color="#3498db", alpha=0.7)
            else:
                # IDyOM got it right, then show a small blue ring around actual
                ring = Ellipse(
                    (x, y * y_scale), width=0.52, height=0.28 * y_scale * 7,
                    angle=-10, facecolor="none", edgecolor="#3498db",
                    linewidth=1.8, alpha=0.6, zorder=6, linestyle="-",
                )
                ax.add_patch(ring)
 
            # Transformer prediction (in red), offset slightly right
            if trans_pred != pitch:
                draw_ledger_lines(ax, x + 0.12, y_trans, y_scale)
                draw_notehead(ax, x + 0.12, y_trans, y_scale,
                              color="#e74c3c", alpha=0.7, size=0.32, zorder=3)
                draw_accidental(ax, x + 0.12, trans_pred, y_scale,
                           color="#e74c3c", alpha=0.7)
            else:
                # Transformer got it right, then show a small red ring
                ring = Ellipse(
                    (x, y * y_scale), width=0.58, height=0.32 * y_scale * 7,
                    angle=-10, facecolor="none", edgecolor="#e74c3c",
                    linewidth=1.8, alpha=0.6, zorder=6, linestyle="-",
                )
                ax.add_patch(ring)
 
            # Draw actual notehead on top (always black, always visible)
            draw_notehead(ax, x, y, y_scale, color="#2c3e50", zorder=7)
            draw_stem(ax, x, y, y_scale, color="#2c3e50")
            draw_accidental(ax, x, pitch, y_scale, color="#2c3e50")
 
            # Position number below
            ax.text(x, min(STAFF_LINE_Y) * y_scale - 0.55, str(i + 1),
                    fontsize=7, ha="center", va="top", color="#7f8c8d")
 
        else:
            # Normal note (no prediction shown)
            draw_notehead(ax, x, y, y_scale, color="#2c3e50")
            draw_stem(ax, x, y, y_scale, color="#2c3e50")
            draw_accidental(ax, x, pitch, y_scale, color="#2c3e50")
 
            # Position number below (lighter for non-prediction positions)
            ax.text(x, min(STAFF_LINE_Y) * y_scale - 0.55, str(i + 1),
                    fontsize=6, ha="center", va="top", color="#bdc3c7")
 
    # Legend
    legend_elements = [
        mlines.Line2D([], [], marker="o", color="#2c3e50", markersize=8,
                      linestyle="None", label="Actual note"),
        mlines.Line2D([], [], marker="o", color="#3498db", markersize=7,
                      linestyle="None", alpha=0.7, label="IDyOM prediction"),
        mlines.Line2D([], [], marker="o", color="#e74c3c", markersize=7,
                      linestyle="None", alpha=0.7, label="Transformer prediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#dee2e6")
 
    # Note name labels for prediction positions
    for i, pitch in enumerate(melody):
        if (i + 1) in pred_lookup:
            x = margin_left + i * note_spacing
            idyom_pred, trans_pred = pred_lookup[i + 1]
 
            label_parts = [f"Actual: {midi_to_name(pitch)}"]
            if idyom_pred != pitch:
                label_parts.append(f"IDyOM: {midi_to_name(idyom_pred)}")
            else:
                label_parts.append(f"IDyOM: {midi_to_name(idyom_pred)}")
            if trans_pred != pitch:
                label_parts.append(f"Transformer: {midi_to_name(trans_pred)}")
            else:
                label_parts.append(f"Transformer: {midi_to_name(trans_pred)}")
 
            ax.text(x, max(STAFF_LINE_Y) * y_scale + 0.55,
                    "\n".join(label_parts),
                    fontsize=6, ha="center", va="bottom", color="#555",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#dee2e6", alpha=0.85),
                    linespacing=1.4)
 
    # Formatting
    ax.set_xlim(0, x_end + 0.3)
 
    # Y limits: accommodate notes above/below staff + labels
    all_pitches = list(melody)
    for pos in pred_lookup:
        ip, tp = pred_lookup[pos]
        all_pitches.extend([ip, tp])
    all_y = [midi_to_staff_y(p) for p in all_pitches]
    y_min = min(min(all_y), min(STAFF_LINE_Y)) - 4
    y_max = max(max(all_y), max(STAFF_LINE_Y)) + 6
    ax.set_ylim(y_min * y_scale - 0.3, y_max * y_scale + 0.3)
 
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
 
    plt.tight_layout()
    fig.savefig(output_png, dpi=250, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_png}")
 
    plt.close(fig)


# ======================================
# RUN
# ======================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw hymn staff with actual vs predicted notes from IDyOM and Transformer"
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=["full", "sliding", "viewpoints"],
        default="sliding",
        help="Which experiment config to compare (default: sliding)"
    )
    parser.add_argument(
        "--melody", "-m",
        default=None,
        help="Melody name to plot (default: first available). "
             "Use --list to see available melodies."
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available melody names and exit"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate staff figures for ALL melodies"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="figures",
        help="Output directory for figures (default: figures)"
    )
    args = parser.parse_args()
 
    # Resolve config
    experiment = args.experiment
    exp_label = EXPERIMENT_LABELS.get(experiment, experiment)
    idyom_path = IDYOM_CONFIGS[experiment]
    transformer_path = TRANSFORMER_CONFIGS[experiment]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
 
    print(f"Hymn Staff Prediction Figure: {exp_label}")
 
    # Load data
    idyom_melodies = load_idyom_predictions(idyom_path)
    transformer_melodies = load_transformer_predictions(transformer_path)
    probe_map = load_probe_info()
 
    available = sorted(idyom_melodies.keys())
 
    # --list: print melody names and exit
    if args.list:
        print(f"\nAvailable melodies ({len(available)}):")
        for name in available:
            probe = probe_map.get(name)
            probe_str = f"probe at note {probe['position']}, rating {probe['mean_rating']:.2f}" if probe else ""
            print(f"{name}{probe_str}")
        exit(0)
 
    # Decide which melodies to plot
    if args.all:
        melodies_to_plot = available
    elif args.melody:
        if args.melody in available:
            melodies_to_plot = [args.melody]
        else:
            print(f"\n'{args.melody}' not found.")
            print(f"Available: {available}")
            print(f"Use --list to see all melody names.")
            exit(1)
    else:
        melodies_to_plot = [available[0]]
 
    # Generate figures
    for melody_name in melodies_to_plot:
        melody_pitches, predictions, probe_info = build_staff_data(
            melody_name, idyom_melodies, transformer_melodies, probe_map
        )
 
        probe_pos = probe_info['position'] if probe_info else None
        rating = probe_info['mean_rating'] if probe_info else None
        title = f"{melody_name} ({exp_label})"
        title += f"\nProbe at note {probe_pos} / human expectedness: {rating:.2f}/7"
 
        output_png = output_dir / f"staff_{experiment}_{melody_name}.png"
 
        create_staff_figure(
            melody=melody_pitches,
            predictions=predictions,
            show_all=True,
            probe_pos=probe_pos,
            output_png=str(output_png),
            title=title,
        )
