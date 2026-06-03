import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
OUTPUT_DIR = "figures"


def evaluate_model(model, X_test, y_test, class_names=None):
    if class_names is None:
        class_names = STAGE_NAMES

    device = next(model.parameters()).device
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)

    y_pred = predicted.cpu().numpy()
    y_true = y_test

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(report)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")

    return acc, cm, report
