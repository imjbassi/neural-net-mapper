# neural-net-mapper

An interactive Python tool that trains a multi-layer perceptron (MLP) with dropout on a synthetic shapes dataset and maps its inner workings over time. It visualizes neuron activations, weight magnitudes/signs, predictions, and live training loss/accuracy through animated network diagrams using Matplotlib.

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train and render (saves `outputs/animation.mp4`, falls back to `.gif` if needed)

```bash
python -m src.main
```

Re-render from saved snapshots:

```bash
python -m src.visualize
```

Tune training in `src/train.py` (`train_model()` args: `epochs`, `sample_every`, `hidden_sizes`, `dropout`, `seed`, dataset options).

## What You'll See

- **Left**: Input sample (32×32) and predicted class with confidence
- **Middle**: Network diagram
  - **Nodes**: Color/size represent activation (consistent global scale)
  - **Edges**: Green = positive weight, red = negative weight; thickness/alpha ∝ |weight|; top-k per node for clarity
  - **Red ring**: Neuron dropped by dropout in that epoch snapshot
- **Right top**: Loss and accuracy over epochs
- **Right bottom**: Class probability bars

## Files

- `requirements.txt` — Dependencies
- `data/generate_dataset.py` — Synthetic dataset generation (centered shapes, jitter, fill/outline)
- `src/model.py` — MLP with dropout (Kaiming initialization)
- `src/train.py` — Training loop and snapshot capture
- `src/visualize.py` — Animation renderer
- `src/main.py` — Entry point