import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data.generate_dataset import SHAPE_NAMES, generate_dataset
from src.model import ShapeMLP


def _forward_capture(
    model: ShapeMLP,
    sample_x: torch.Tensor,
    epoch: int = 0,
    capture_dropout: bool = True,
    base_seed: int = 1234,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, int, float, List[np.ndarray]]:
    """Run a manual forward pass to capture per-layer activations and weights.

    Args:
        model: The neural network model
        sample_x: Input tensor (1, input_dim)
        epoch: Current training epoch
        capture_dropout: Whether to capture dropout masks
        base_seed: Base random seed for reproducibility

    Returns:
        Tuple containing:
        - acts: List of post-activation vectors for hidden layers and softmax probs for output
        - weights_for_viz: List of weight matrices between visualized layers
        - logits: Raw output logits as numpy array (1, C)
        - pred_label: Predicted class (int)
        - pred_conf: Prediction confidence (float)
        - dropout_masks: List of boolean dropout masks
    """
    acts: List[np.ndarray] = []
    weights_for_viz: List[np.ndarray] = []
    dropout_masks: List[np.ndarray] = []

    x = sample_x
    linear_index = 0

    # Walk through layers capturing activations after ReLU and final softmax
    with torch.no_grad():
        for layer in model.model:
            if isinstance(layer, torch.nn.Linear):
                x = layer(x)
                # Store weights for viz except for the very first Linear (input->h1)
                if linear_index > 0:
                    weights_for_viz.append(layer.weight.detach().cpu().numpy())
                linear_index += 1
            elif isinstance(layer, torch.nn.ReLU):
                x = layer(x)
                acts.append(x.detach().cpu().numpy().squeeze())
            elif isinstance(layer, torch.nn.Dropout):
                # No-op in eval mode; masks captured in a second pass below
                pass

        # x is now logits
        logits = x.detach().cpu().numpy()
        probs = torch.softmax(x, dim=1).detach().cpu().numpy().squeeze()
    acts.append(probs)  # Treat output probs as final "activation"
    pred_label = int(np.argmax(probs))
    pred_conf = float(np.max(probs))

    # Optionally capture dropout masks using a deterministic training=True pass.
    # fork_rng keeps this seeding from perturbing the training RNG stream.
    if capture_dropout:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base_seed + int(epoch))
            with torch.no_grad():
                x2 = sample_x
                for layer in model.model:
                    if isinstance(layer, torch.nn.Linear):
                        x2 = layer(x2)
                    elif isinstance(layer, torch.nn.ReLU):
                        x2 = layer(x2)
                    elif isinstance(layer, torch.nn.Dropout):
                        x2_pre = x2.clone()
                        x2_dropped = F.dropout(x2, p=layer.p, training=True)
                        # Identify dropped neurons: non-zero before dropout, zero after
                        dropped = (x2_pre != 0) & (x2_dropped == 0)
                        dropout_masks.append(
                            dropped.detach().cpu().numpy().squeeze().astype(bool)
                        )
                        x2 = x2_dropped

    return acts, weights_for_viz, logits, pred_label, pred_conf, dropout_masks


def train_model(
    epochs: int = 60,
    sample_every: int = 5,
    hidden_sizes: Optional[Sequence[int]] = None,
    dropout: float = 0.3,
    n_per_class: int = 300,
    seed: int = 42,
    lr: float = 3e-4,
    batch_size: int = 32,
    val_split: float = 0.2,
    device: Optional[str] = None,
    output_dir: str = "outputs",
    verbose: bool = True,
    centered: bool = True,
    jitter: int = 2,
    fill: bool = True,
    thickness: int = 2,
) -> List[Dict[str, Any]]:
    """Train an MLP and capture visualization snapshots.

    Args:
        epochs: Number of training epochs
        sample_every: Frequency of snapshot capture
        hidden_sizes: List of hidden layer sizes
        dropout: Dropout probability
        n_per_class: Number of samples per class
        seed: Random seed for reproducibility
        lr: Learning rate for the Adam optimizer
        batch_size: Mini-batch size
        val_split: Fraction of the dataset held out for validation
        device: Torch device string ("cpu", "cuda"); auto-detected when None
        output_dir: Directory for saved snapshots
        verbose: Print per-epoch progress
        centered: Center shapes (with jitter); False scatters them randomly,
            making the task considerably harder
        jitter: Max pixel jitter applied to centered shapes
        fill: Draw filled shapes; False draws outlines only (harder)
        thickness: Outline thickness when fill is False

    Returns:
        List of snapshot dictionaries containing training state at various epochs

    Saves:
        <output_dir>/snapshots.npz containing a list of dict snapshots.
        Each snapshot has keys: epoch, acts, weights, loss_hist, train_loss_hist,
        acc_hist, img, label, pred, conf, hidden_sizes, dropout_masks, dropout_p,
        classes.
    """
    if hidden_sizes is None:
        hidden_sizes = [128, 64]
    hidden_sizes = list(hidden_sizes)

    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if sample_every <= 0:
        raise ValueError(f"sample_every must be positive, got {sample_every}")
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Generate dataset and split BEFORE standardizing so validation statistics
    # never leak into the normalization applied to training data.
    X, y = generate_dataset(
        n_per_class, centered=centered, thickness=thickness, jitter=jitter, fill=fill
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    # Standardize inputs using training-set statistics only. A single global
    # mean/std is used rather than per-pixel stats: near-constant pixels have
    # std ~ 0, and dividing by it explodes small validation-set differences
    # into huge feature values (and huge validation losses).
    mean = float(X_train.mean())
    std = float(X_train.std()) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
        ),
        batch_size=batch_size,
    )

    # Initialize model and training components
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = ShapeMLP(X.shape[1], hidden_sizes, len(SHAPE_NAMES), dropout).to(resolved_device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    snapshots: List[Dict[str, Any]] = []
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    acc_history: List[float] = []

    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(resolved_device), yb.to(resolved_device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_batches += 1

        # Evaluation phase
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(resolved_device), yb.to(resolved_device)
                outputs = model(xb)
                loss_val = criterion(outputs, yb)
                val_loss += loss_val.item()
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

        acc = correct / total
        train_loss_history.append(epoch_train_loss / max(train_batches, 1))
        val_loss_history.append(val_loss / len(val_loader))
        acc_history.append(acc)

        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss_history[-1]:.4f}  "
                f"val_loss={val_loss_history[-1]:.4f}  "
                f"val_acc={acc * 100:.1f}%"
            )

        # Capture snapshots at specified intervals
        if epoch % sample_every == 0 or epoch == 1 or epoch == epochs:
            # Pick a fixed validation sample for consistent visualization
            sample_idx = 0
            sample_x = (
                torch.tensor(X_val[sample_idx], dtype=torch.float32)
                .unsqueeze(0)
                .to(resolved_device)
            )
            acts, weights_viz, logits, pred_label, pred_conf, dropout_masks = _forward_capture(
                model, sample_x, epoch=epoch, capture_dropout=True, base_seed=seed
            )

            snapshots.append(
                {
                    "epoch": epoch,
                    "acts": acts,  # List: [h1, h2, ..., probs]
                    "weights": weights_viz,  # Between hidden layers (and hidden->out)
                    "loss_hist": val_loss_history.copy(),
                    "train_loss_hist": train_loss_history.copy(),
                    "acc_hist": acc_history.copy(),
                    "img": X_val[sample_idx].reshape(32, 32),
                    "label": int(y_val[sample_idx]),
                    "pred": pred_label,
                    "conf": pred_conf,
                    "hidden_sizes": hidden_sizes,
                    "dropout_masks": dropout_masks,
                    "dropout_p": dropout,
                    "classes": list(SHAPE_NAMES),
                }
            )

    # Save snapshots to disk
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "snapshots.npz"),
        snapshots=np.array(snapshots, dtype=object),
    )
    return snapshots


if __name__ == "__main__":
    train_model()
