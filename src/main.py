```python
"""Main entry point for the neural network mapper application."""

from src.train import train_model
from src.visualize import visualize_snapshots


def main():
    """Train a neural network model and visualize training snapshots."""
    snapshots = train_model()
    visualize_snapshots(snapshots)


if __name__ == "__main__":
    main()
```