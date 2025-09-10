import numpy as np
from src.train import train_model
from src.visualize import visualize_snapshots


def main():
    snapshots = train_model()
    visualize_snapshots(snapshots)


if __name__ == "__main__":
    main()
