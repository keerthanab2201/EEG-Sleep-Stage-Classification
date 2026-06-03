import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import os, sys, json

sys.path.insert(0, ".")
from data_loader import load_data, STAGE_NAMES
from preprocess import preprocess_data, normalize_epoch
from feature_extraction import extract_features_bulk, FEATURE_NAMES
from model import SleepStageCNN

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_sklearn(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    report = classification_report(
        y_test, y_pred, target_names=STAGE_NAMES, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, acc, macro_f1, per_class_f1, report, cm


def evaluate_cnn(model, X_test, y_test, device):
    X_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(y_test, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        _, preds = torch.max(model(X_t), 1)
    y_pred = preds.cpu().numpy()
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    report = classification_report(
        y_test, y_pred, target_names=STAGE_NAMES, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, acc, macro_f1, per_class_f1, report, cm


def plot_comparison(results, feature_importances=None):
    models = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overall metrics
    ax = axes[0]
    metrics = ["Accuracy", "Macro F1"]
    x = np.arange(len(metrics))
    w = 0.25
    colors = ["#3366cc", "#ff9900", "#109618"]
    for i, name in enumerate(models):
        vals = [results[name]["acc"], results[name]["macro_f1"]]
        ax.bar(x + i * w - w, vals, w, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.05)

    # Per-class F1
    ax = axes[1]
    x = np.arange(len(STAGE_NAMES))
    for i, name in enumerate(models):
        vals = results[name]["per_class_f1"]
        ax.bar(x + i * w - w, vals, w, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.05)

    # Feature importance (Random Forest)
    ax = axes[2]
    if feature_importances is not None:
        idx = np.argsort(feature_importances)[::-1]
        ax.barh(range(len(idx)), feature_importances[idx], color="#ff9900", alpha=0.8)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in idx])
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        ax.grid(True, alpha=0.2, axis="x")
    else:
        ax.text(
            0.5,
            0.5,
            "Feature importance\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Random Forest Feature Importance")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "classical_vs_cnn.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/classical_vs_cnn.png")


def main():
    print("=" * 70)
    print("CLASSICAL ML BASELINES vs CNN")
    print("=" * 70)

    X, y, sfreq, epoch_samples = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

    # --- Feature extraction ---
    print("\nExtracting features (mean, var, RMS, band powers)...")
    X_train_feat, y_train_feat = extract_features_bulk(
        np.array([normalize_epoch(e) for e in X_train]), y_train, sfreq
    )
    X_test_feat, y_test_feat = extract_features_bulk(
        np.array([normalize_epoch(e) for e in X_test]), y_test, sfreq
    )
    print(f"Feature matrix: {X_train_feat.shape} (n_samples x n_features)")
    print(f"Features: {FEATURE_NAMES}")

    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=42)
    lr.fit(X_train_feat, y_train_feat)
    lr_pred, lr_acc, lr_mf1, lr_pcf1, lr_report, lr_cm = evaluate_sklearn(
        lr, X_test_feat, y_test, "LR"
    )
    print(f"Accuracy: {lr_acc:.4f}  Macro F1: {lr_mf1:.4f}")
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s}: F1={lr_pcf1[i]:.4f}")

    # --- Random Forest ---
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train_feat, y_train_feat)
    rf_pred, rf_acc, rf_mf1, rf_pcf1, rf_report, rf_cm = evaluate_sklearn(
        rf, X_test_feat, y_test, "RF"
    )
    print(f"Accuracy: {rf_acc:.4f}  Macro F1: {rf_mf1:.4f}")
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s}: F1={rf_pcf1[i]:.4f}")

    # --- CNN ---
    print("\n--- CNN (baseline) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = SleepStageCNN(epoch_samples, 5).to(device)
    cnn_model.load_state_dict(
        torch.load("sleep_stage_cnn.pth", map_location=device, weights_only=True)
    )
    cnn_pred, cnn_acc, cnn_mf1, cnn_pcf1, cnn_report, cnn_cm = evaluate_cnn(
        cnn_model, X_test, y_test, device
    )
    print(f"Accuracy: {cnn_acc:.4f}  Macro F1: {cnn_mf1:.4f}")
    for i, s in enumerate(STAGE_NAMES):
        print(f"  {s}: F1={cnn_pcf1[i]:.4f}")

    # --- Comparison Table ---
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Model':>16s}  {'Accuracy':>9s}  {'Macro F1':>9s}"
    for s in STAGE_NAMES:
        header += f"  {s:>6s}"
    print(header)
    print("-" * len(header))

    results = {}
    for name, acc, mf1, pcf1 in [
        ("Logistic Regression", lr_acc, lr_mf1, lr_pcf1),
        ("Random Forest", rf_acc, rf_mf1, rf_pcf1),
        ("CNN", cnn_acc, cnn_mf1, cnn_pcf1),
    ]:
        results[name] = {"acc": acc, "macro_f1": mf1, "per_class_f1": pcf1}
        line = f"{name:>16s}  {acc:9.4f}  {mf1:9.4f}"
        for v in pcf1:
            line += f"  {v:6.4f}"
        print(line)

    # --- N1/REM spotlight ---
    print("\n--- Minority Class Spotlight (N1, REM) ---")
    for s_idx, s_name in [(1, "N1"), (4, "REM")]:
        print(f"\n  {s_name}:")
        for name in ["Logistic Regression", "Random Forest", "CNN"]:
            prec = results[name]["per_class_f1"][s_idx]
            print(f"    {name:>20s}: F1={prec:.4f}")

    # --- Discussion ---
    print("\n" + "=" * 70)
    print("DISCUSSION")
    print("=" * 70)
    print("""
Feature-based vs. Deep Learning Approaches
-------------------------------------------
Strengths of feature-based (LR/RF):
  + Interpretable: feature importances reveal which frequency bands drive decisions
  + Fast training (seconds vs minutes)
  + Works with small datasets
  + No GPU required

Weaknesses of feature-based:
  - Hand-crafted features may miss transient events (spindles, K-complexes)
  - No temporal structure preservation (features collapse 30s into 7 scalars)
  - Fixed frequency bands may not be optimal for every subject

Strengths of deep learning (CNN):
  + Learns optimal temporal features directly from raw EEG
  + Can detect transient patterns (spindles, sharp waves)
  + Hierarchical: edges -> patterns -> stage-level abstractions

Weaknesses of deep learning:
  - Requires more data
  - Black-box: harder to interpret
  - Longer training, needs GPU for scaling
  - Risk of overfitting on small datasets

Why the comparison matters:
  If classical methods match CNN performance, the hand-crafted features
  (band powers, RMS) capture most relevant signal information. If CNN
  significantly outperforms, there is meaningful temporal structure that
  simple features miss -- justifying deep learning for this task.
""")

    plot_comparison(results, feature_importances=rf.feature_importances_)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
