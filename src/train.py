import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from data.generate_dataset import generate_dataset
from src.model import ShapeMLP
import os
from typing import List, Tuple, Dict, Any
import torch.nn.functional as F
import random

SHAPE_NAMES = ["circle", "square", "triangle"]


def _forward_capture(model: ShapeMLP, sample_x: torch.Tensor, epoch: int = 0, capture_dropout: bool = True,
                     base_seed: int = 1234) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, int, float, List[np.ndarray]]:
    """Run a manual forward pass to capture per-layer activations and weights.

    Returns:
    - acts: list of post-activation vectors for hidden layers and softmax probs for output.
    - weights_for_viz: list of weight matrices between visualized layers (exclude input->h1 to keep viz light).
    - logits: raw output logits as numpy array (1, C)
    - pred_label: int predicted class
    - pred_conf: float confidence (softmax max)
    """
    acts: List[np.ndarray] = []
    weights_for_viz: List[np.ndarray] = []
    dropout_masks: List[np.ndarray] = []

    x = sample_x
    linear_index = 0
    last_activation = None

    # Walk through layers capturing activations after ReLU and final softmax
    for layer in model.model:
        if isinstance(layer, torch.nn.Linear):
            # linear transform
            z = x @ layer.weight.T + layer.bias
            x = z
            # Store weights for viz except for the very first Linear (input->h1)
            if linear_index > 0:
                weights_for_viz.append(layer.weight.detach().cpu().numpy())
            linear_index += 1
        elif isinstance(layer, torch.nn.ReLU):
            x = torch.relu(x)
            acts.append(x.detach().cpu().numpy().squeeze())
        elif isinstance(layer, torch.nn.Dropout):
            # No-op in eval visualization path; masks captured in a second pass below
            pass

    # x is now logits
    logits = x.detach().cpu().numpy()
    probs = torch.softmax(x, dim=1).detach().cpu().numpy().squeeze()
    acts.append(probs)  # treat output probs as final "activation"
    pred_label = int(np.argmax(probs))
    pred_conf = float(np.max(probs))

    # Optionally capture dropout masks using a deterministic training=True pass
    if capture_dropout:
        torch.manual_seed(base_seed + int(epoch))
        x2 = sample_x
        for layer in model.model:
            if isinstance(layer, torch.nn.Linear):
                x2 = x2 @ layer.weight.T + layer.bias
            elif isinstance(layer, torch.nn.ReLU):
                x2 = torch.relu(x2)
                pre = x2.clone()
            elif isinstance(layer, torch.nn.Dropout):
                x2_d = F.dropout(x2, p=layer.p, training=True)
                dropped = (pre != 0) & (x2_d == 0)
                dropout_masks.append(dropped.detach().cpu().numpy().squeeze().astype(bool))
                x2 = x2_d
        # Final linear has no dropout

    return acts, weights_for_viz, logits, pred_label, pred_conf, dropout_masks


def train_model(epochs: int = 60, sample_every: int = 5, hidden_sizes: List[int] = [128, 64], dropout: float = 0.3,
                n_per_class: int = 300, seed: int = 42):
    """Train an MLP and capture visualization snapshots.

    Saves outputs/snapshots.npz containing a list of dict snapshots.
    Each snapshot has keys: epoch, acts, weights, loss_hist, acc_hist, img, label, pred, conf, hidden_sizes.
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = generate_dataset(n_per_class, centered=True, thickness=2, jitter=2, fill=True)
    # Standardize inputs for better conditioning
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeMLP(X.shape[1], hidden_sizes, 3, dropout).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    snapshots = []
    loss_history = []
    acc_history = []

    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        # Evaluate
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss_val = criterion(outputs, yb)
                val_loss += loss_val.item()
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        acc = correct / total
        loss_history.append(val_loss / len(val_loader))
        acc_history.append(acc)

        if epoch % sample_every == 0 or epoch == 1 or epoch == epochs:
            # pick a fixed validation sample for consistent viz
            sample_idx = 0
            sample_x = torch.tensor(X_val[sample_idx]).unsqueeze(0).to(device)
            acts, weights_viz, logits, pred_label, pred_conf, dropout_masks = _forward_capture(
                model, sample_x, epoch=epoch, capture_dropout=True, base_seed=seed
            )

            snapshots.append({
                "epoch": epoch,
                "acts": acts,  # list: [h1, h2, ..., probs]
                "weights": weights_viz,  # between hidden layers (and hidden->out)
                "loss_hist": loss_history.copy(),
                "acc_hist": acc_history.copy(),
                "img": X_val[sample_idx].reshape(32, 32),
                "label": int(y_val[sample_idx]),
                "pred": pred_label,
                "conf": pred_conf,
                "hidden_sizes": hidden_sizes,
                "dropout_masks": dropout_masks,
                "dropout_p": dropout,
                "classes": SHAPE_NAMES,
            })

    os.makedirs("outputs", exist_ok=True)
    # Save as npz with pickling
    np.savez("outputs/snapshots.npz", snapshots=np.array(snapshots, dtype=object))
    return snapshots

if __name__ == "__main__":
    train_model()
