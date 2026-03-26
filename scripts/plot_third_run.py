"""Generate 4 training-curve PNGs from the 3rd SLURM run TensorBoard logs.

Style: TensorBoard-like — faded raw data + bold smoothed (EMA) line on top.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

BASE = os.path.join(os.path.dirname(__file__), '..',
                    'models_data_v2_12_labels', 'third_run_on_server')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'graphs', 'third_run')
os.makedirs(OUT_DIR, exist_ok=True)

EMA_WEIGHT = 0.6  # moderate smoothing (binning handles the noise)


def load_scalar(logdir, tag):
    """Load a scalar tag, averaged per step + all raw events."""
    ea = EventAccumulator(logdir, size_guidance={SCALARS: 0})
    ea.Reload()
    events = ea.Scalars(tag)
    # All raw events (for faded line)
    raw_steps = np.array([e.step for e in events])
    raw_vals = np.array([e.value for e in events])
    # Averaged per step (for smoothed line)
    from collections import defaultdict
    by_step = defaultdict(list)
    for e in events:
        by_step[e.step].append(e.value)
    steps = sorted(by_step)
    avg_vals = [sum(by_step[s]) / len(by_step[s]) for s in steps]
    return np.array(steps), np.array(avg_vals), raw_steps, raw_vals


def bin_average(steps, vals, bin_size=50):
    """Average data into bins of bin_size steps."""
    if len(steps) <= bin_size:
        return steps, vals
    n_bins = len(steps) // bin_size
    trimmed = n_bins * bin_size
    bin_steps = steps[:trimmed].reshape(n_bins, bin_size).mean(axis=1)
    bin_vals = vals[:trimmed].reshape(n_bins, bin_size).mean(axis=1)
    # Append leftover points as one final bin
    if trimmed < len(steps):
        bin_steps = np.append(bin_steps, steps[trimmed:].mean())
        bin_vals = np.append(bin_vals, vals[trimmed:].mean())
    return bin_steps, bin_vals


def ema_smooth(values, weight=EMA_WEIGHT):
    """Exponential moving average, same as TensorBoard smoothing."""
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


# ---------- Batch-load all data first ----------
print("Loading event logs...")

configs = {
    'baseline': os.path.join(BASE, 'kwt1_baseline'),
    'distill':  os.path.join(BASE, 'kwt1_distill'),
}
data = {}
for name, model_dir in configs.items():
    for split in ('train', 'validation'):
        logdir = os.path.join(model_dir, 'logs', split)
        for tag in ('accuracy', 'loss'):
            key = (name, split, tag)
            print(f"  loading {name}/{split}/{tag} ...")
            data[key] = load_scalar(logdir, tag)
print("All data loaded.\n")

# ---------- Plot helper ----------

COLORS = {
    'train': '#FF9800',       # orange
    'validation': '#2196F3',  # blue
}


def plot_metric(name, metric, ylabel, title, filename,
                ylim=None, target_lines=None, legend_loc='lower right'):
    fig, ax = plt.subplots(figsize=(12, 6))

    for split, color in COLORS.items():
        steps, vals, raw_steps, raw_vals = data[(name, split, metric)]
        if metric == 'accuracy':
            vals = vals * 100
            raw_vals = raw_vals * 100

        if split == 'train':
            smooth_steps, smooth_vals = bin_average(steps, vals, bin_size=50)
            fade_steps, fade_vals = bin_average(steps, vals, bin_size=5)
        else:
            # Validation: smooth the step-averaged, fade uses all raw batch events
            smooth_steps, smooth_vals = steps, vals
            fade_steps, fade_vals = raw_steps, raw_vals
        smoothed = ema_smooth(smooth_vals)

        # Faded raw data
        ax.plot(fade_steps, fade_vals, color=color, linewidth=0.5, alpha=0.25)
        # Bold smoothed line
        ax.plot(smooth_steps, smoothed, color=color, linewidth=2, alpha=0.9,
                label=f'{split.capitalize()}')

    if target_lines:
        for label, val, color in target_lines:
            ax.axhline(y=val, color=color, linestyle='--', alpha=0.5,
                        linewidth=1.5, label=label)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc=legend_loc)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


# ---------- Generate all 4 plots ----------

plot_metric('baseline', 'accuracy',
            'Accuracy (%)', 'Baseline Accuracy (Train vs Validation)',
            'acc-3-b.png', ylim=(70, 100),
            target_lines=[('Paper target: 97.72%', 97.72, '#4CAF50')])

plot_metric('baseline', 'loss',
            'Loss', 'Baseline Loss (Train vs Validation)',
            'loss-3-b.png', legend_loc='upper right')

plot_metric('distill', 'accuracy',
            'Accuracy (%)', 'Distillation Accuracy (Train vs Validation)',
            'acc-3-d.png', ylim=(70, 100),
            target_lines=[('Paper target: 98.08%', 98.08, '#4CAF50')])

plot_metric('distill', 'loss',
            'Loss', 'Distillation Loss (Train vs Validation)',
            'loss-3-d.png', legend_loc='upper right')
