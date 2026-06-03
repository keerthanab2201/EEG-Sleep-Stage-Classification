import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os, sys

sys.path.insert(0, ".")
from data_loader import load_data, STAGE_NAMES
from preprocess import preprocess_data
from train import train_model
from model import SleepStageCNN

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(model, X, y):
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        _, preds = torch.max(model(X_t), 1)
    return preds.cpu().numpy()


def run_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test, input_length, balanced, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        input_length=input_length,
        n_classes=5,
        batch_size=32,
        epochs=10,
        lr=0.001,
        balanced=balanced,
    )
    y_pred = evaluate(model, X_test, y_test)
    tag = "balanced" if balanced else "baseline"

    report = classification_report(
        y_test, y_pred, target_names=STAGE_NAMES, digits=4, output_dict=True
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    torch.save(model.state_dict(), f"sleep_stage_cnn_{tag}.pth")
    print(f"  Model saved: sleep_stage_cnn_{tag}.pth")

    return model, y_pred, report, macro_f1, cm, history


def plot_comparison(
    baseline_report,
    balanced_report,
    baseline_cm,
    balanced_cm,
    y_test,
    baseline_pred,
    balanced_pred,
):
    stages = STAGE_NAMES

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Baseline vs Class-Balanced Training Comparison", fontsize=14, y=1.01)

    metrics = ["precision", "recall", "f1-score"]
    titles = ["Precision", "Recall", "F1-Score"]
    colors_baseline = "#3366cc"
    colors_balanced = "#cc3333"

    for col, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[0, col]
        b_vals = [baseline_report[s][metric] for s in stages]
        bal_vals = [balanced_report[s][metric] for s in stages]
        x = np.arange(len(stages))
        w = 0.35
        ax.bar(
            x - w / 2, b_vals, w, label="Baseline", color=colors_baseline, alpha=0.85
        )
        ax.bar(
            x + w / 2, bal_vals, w, label="Balanced", color=colors_balanced, alpha=0.85
        )
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(f"Per-Class {title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylim(0, 1.05)

    ax = axes[1, 0]
    ax.axis("off")
    lines = ["Comparison of N1 and REM Performance:", ""]
    for s in ["N1", "REM"]:
        for metric in ["precision", "recall", "f1-score"]:
            bv = baseline_report[s][metric]
            balv = balanced_report[s][metric]
            delta = balv - bv
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "—"
            lines.append(
                f"  {s} {metric:>10s}: baseline={bv:.3f}  balanced={balv:.3f}  {arrow}{abs(delta):.3f}"
            )
    ax.text(
        0,
        0.95,
        "\n".join(lines),
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        transform=ax.transAxes,
    )

    for col, (cm, pred, tag) in enumerate(
        [
            (baseline_cm, baseline_pred, "Baseline"),
            (balanced_cm, balanced_pred, "Balanced"),
        ],
        start=1,
    ):
        ax = axes[1, col]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(stages, fontsize=8, rotation=45)
        ax.set_yticklabels(stages, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{tag} Confusion Matrix")
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.colorbar(im, ax=ax, fraction=0.046)

    b_macro = f1_score(y_test, baseline_pred, average="macro")
    bal_macro = f1_score(y_test, balanced_pred, average="macro")
    b_weighted = f1_score(y_test, baseline_pred, average="weighted")
    bal_weighted = f1_score(y_test, balanced_pred, average="weighted")

    ax = axes[1, 0] if False else axes[1, 0]
    lines2 = [
        f"Macro F1:    baseline={b_macro:.4f}  balanced={bal_macro:.4f}",
        f"Weighted F1: baseline={b_weighted:.4f}  balanced={bal_weighted:.4f}",
    ]
    ax2 = axes[1, 0]
    old_text = ax2.texts[0] if ax2.texts else None

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "training_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/training_comparison.png")


def main():
    print("=" * 70)
    print("TRAINING COMPARISON: BASELINE vs CLASS-BALANCED")
    print("=" * 70)

    X, y, sfreq, epoch_samples = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

    print("\nClass distribution in training set:")
    for i, name in enumerate(STAGE_NAMES):
        print(f"  {name}: {(y_train == i).sum()}")

    (
        baseline_model,
        baseline_pred,
        baseline_report,
        baseline_mf1,
        baseline_cm,
        baseline_hist,
    ) = run_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test, epoch_samples, balanced=False
    )

    (
        balanced_model,
        balanced_pred,
        balanced_report,
        balanced_mf1,
        balanced_cm,
        balanced_hist,
    ) = run_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test, epoch_samples, balanced=True
    )

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':>20s}  {'Baseline':>10s}  {'Balanced':>10s}  {'Diff':>10s}")
    print("-" * 54)
    print(
        f"{'Macro F1':>20s}  {baseline_mf1:10.4f}  {balanced_mf1:10.4f}  {balanced_mf1 - baseline_mf1:+10.4f}"
    )
    print(
        f"{'Weighted F1':>20s}  {f1_score(y_test, baseline_pred, average='weighted'):10.4f}  {f1_score(y_test, balanced_pred, average='weighted'):10.4f}  {'':>10s}"
    )

    print("\nPer-class breakdown:")
    print(
        f"{'Class':>6s}  {'Prec base':>9s}  {'Prec bal':>9s}  {'Diff prec':>8s}  "
        f"{'Rec base':>8s}  {'Rec bal':>8s}  {'Diff rec':>8s}  "
        f"{'F1 base':>7s}  {'F1 bal':>7s}  {'Diff F1':>7s}"
    )
    print("-" * 90)
    for s in STAGE_NAMES:
        pb = baseline_report[s]["precision"]
        pbal = balanced_report[s]["precision"]
        rb = baseline_report[s]["recall"]
        rbal = balanced_report[s]["recall"]
        fb = baseline_report[s]["f1-score"]
        fbal = balanced_report[s]["f1-score"]
        print(
            f"{s:>6s}  {pb:9.4f}  {pbal:9.4f}  {pbal - pb:+8.4f}  "
            f"{rb:8.4f}  {rbal:8.4f}  {rbal - rb:+8.4f}  "
            f"{fb:7.4f}  {fbal:7.4f}  {fbal - fb:+7.4f}"
        )

    print("\n--- N1 and REM Analysis ---")
    for s in ["N1", "REM"]:
        old_f1 = baseline_report[s]["f1-score"]
        new_f1 = balanced_report[s]["f1-score"]
        verdict = (
            "IMPROVED"
            if new_f1 > old_f1
            else "DEGRADED"
            if new_f1 < old_f1
            else "UNCHANGED"
        )
        print(f"  {s} F1: {old_f1:.4f} -> {new_f1:.4f} ({verdict})")

    plot_comparison(
        baseline_report,
        balanced_report,
        baseline_cm,
        balanced_cm,
        y_test,
        baseline_pred,
        balanced_pred,
    )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
