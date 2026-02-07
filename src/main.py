```python
"""Main entry point for the neural network mapper application."""

import sys
from pathlib import Path

# Add src directory to path to allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train_model
from src.visualize import visualize_snapshots


def main():
    """Train a neural network model and visualize training snapshots.
    
    This function orchestrates the main workflow:
    1. Trains a neural network model while capturing periodic snapshots
    2. Visualizes the captured snapshots to show training progression
    
    Returns:
        None
    """
    try:
        snapshots = train_model()
        if snapshots:
            visualize_snapshots(snapshots)
        else:
            print("Warning: No snapshots were generated during training.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```