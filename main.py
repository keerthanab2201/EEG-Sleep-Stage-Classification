import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

from data_loader import load_data, STAGE_NAMES
from preprocess import preprocess_data
from visualize import run_visualization
from evaluate import evaluate_model
from compare_training import main as compare_main


def main():
    print("=" * 60)
    print("EEG SLEEP-STAGE CLASSIFICATION PIPELINE")
    print("=" * 60)

    X, y, sfreq, epoch_samples = load_data()
    print(f"\nEpoch samples: {epoch_samples} (at {sfreq} Hz = 30s)")

    print("\nStage distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {STAGE_NAMES[u]}: {c}")

    run_visualization(X, y, sfreq)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

    from train import train_model

    model, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        input_length=epoch_samples,
        n_classes=5,
        batch_size=32,
        epochs=10,
        lr=0.001,
        balanced=False,
    )

    evaluate_model(model, X_test, y_test, class_names=STAGE_NAMES)
    torch.save(model.state_dict(), "sleep_stage_cnn.pth")
    print("Model saved to: sleep_stage_cnn.pth")

    print("\n" + "=" * 60)
    print("RUNNING CLASS-BALANCE COMPARISON")
    print("=" * 60)
    compare_main()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
