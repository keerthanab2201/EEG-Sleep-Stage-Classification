import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import SleepStageCNN


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    input_length,
    n_classes=5,
    batch_size=32,
    epochs=10,
    lr=0.001,
    balanced=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "CLASS-BALANCED" if balanced else "BASELINE"
    print(f"\n--- {tag} TRAINING ---")
    print(f"Training on: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if balanced:
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (n_classes * class_counts)
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        sample_weights = 1.0 / class_counts[y_train]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        print(
            f"Class weights: {dict(zip(range(n_classes), [f'{w:.3f}' for w in class_weights]))}"
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()

    model = SleepStageCNN(input_length, n_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_acc = correct / total
        history["val_acc"].append(val_acc)
        scheduler.step()

        print(
            f"  Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f} ({correct}/{total})"
        )

    return model, history
