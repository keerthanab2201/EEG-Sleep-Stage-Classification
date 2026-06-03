import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
import os

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
OUTPUT_DIR = "figures"

FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
}


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_example_epochs(
    X: np.ndarray, y: np.ndarray, sfreq: float, stage_indices: dict
):
    ensure_output_dir()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    stages_to_plot = [0, 2, 3, 4]
    stage_labels = ["Wake", "N2", "N3", "REM"]
    t = np.arange(X.shape[1]) / sfreq

    for ax, stage, label in zip(axes, stages_to_plot, stage_labels):
        idx = stage_indices[stage][0]
        ax.plot(t, X[idx], color="black", linewidth=0.5)
        ax.set_ylabel("Amplitude (std)")
        ax.set_title(f"Example {label} Epoch (30s)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlim(0, 30)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "example_epochs.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/example_epochs.png")


def plot_fft_psd(X: np.ndarray, y: np.ndarray, sfreq: float, stage_indices: dict):
    ensure_output_dir()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = ["blue", "green", "orange", "red", "purple"]
    stages_to_plot = [0, 2, 3, 4]
    stage_labels = ["Wake", "N2", "N3", "REM"]

    ax_fft = axes[0]
    ax_psd = axes[1]

    for stage, color, label in zip(
        stages_to_plot, [colors[i] for i in [0, 2, 3, 4]], stage_labels
    ):
        idx = stage_indices[stage][0]
        epoch = X[idx]

        freqs = np.fft.rfftfreq(len(epoch), d=1.0 / sfreq)
        fft_vals = np.abs(np.fft.rfft(epoch))
        ax_fft.semilogy(
            freqs, fft_vals, color=color, alpha=0.8, label=label, linewidth=0.8
        )

        freqs_psd, psd = signal.welch(epoch, fs=sfreq, nperseg=min(256, len(epoch)))
        ax_psd.semilogy(
            freqs_psd, psd, color=color, alpha=0.8, label=label, linewidth=0.8
        )

    for ax, title in zip(
        axes, ["FFT Magnitude Spectrum", "Power Spectral Density (Welch)"]
    ):
        ax.set_xlim(0.5, 30)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude" if "FFT" in title else "PSD (V²/Hz)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for band_name, (lo, hi) in FREQ_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.08, color="gray")
            mid = (lo + hi) / 2
            ax.text(
                mid,
                ax.get_ylim()[1] * 0.95,
                band_name,
                ha="center",
                va="top",
                fontsize=8,
                alpha=0.6,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fft_psd_analysis.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fft_psd_analysis.png")


def plot_stage_averages(X: np.ndarray, y: np.ndarray, sfreq: float):
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(12, 6))
    t = np.arange(X.shape[1]) / sfreq
    colors = ["blue", "green", "orange", "red", "purple"]

    for stage in range(5):
        mask = y == stage
        if np.sum(mask) == 0:
            continue
        avg = np.mean(X[mask], axis=0)
        ax.plot(t, avg, color=colors[stage], label=STAGE_NAMES[stage], linewidth=1.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (std)")
    ax.set_title("Average EEG Waveform by Sleep Stage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stage_averages.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/stage_averages.png")


def run_visualization(X: np.ndarray, y: np.ndarray, sfreq: float):
    print("\nGenerating visualizations...")
    stage_indices = {stage: np.where(y == stage)[0] for stage in range(5)}
    plot_example_epochs(X, y, sfreq, stage_indices)
    plot_fft_psd(X, y, sfreq, stage_indices)
    plot_stage_averages(X, y, sfreq)
    print("Visualization complete.")
