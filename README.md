# neural-net-mapper

[![CI](https://github.com/imjbassi/neural-net-mapper/actions/workflows/ci.yml/badge.svg)](https://github.com/imjbassi/neural-net-mapper/actions/workflows/ci.yml)

An interactive Python tool that trains a multi-layer perceptron (MLP) with dropout on a synthetic shapes dataset and maps its inner workings over time. It visualizes neuron activations, weight magnitudes/signs, predictions, and live training loss/accuracy through animated network diagrams using Matplotlib.

![Sample visualization frame](assets/sample_frame.png)

## Architecture / Pipeline

![Pipeline diagram](assets/architecture.svg)

## Demo

![Training animation demo](assets/demo_animation.gif)

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and render** (saves `outputs/animation.mp4`, falls back to `.gif` if ffmpeg is unavailable)

   ```bash
   python -m src.main
   ```

3. **Re-render from saved snapshots** (skips training)

   ```bash
   python -m src.main --render-only
   ```

## Command-Line Options

Everything is configurable from the CLI — no source edits needed:

```bash
python -m src.main --epochs 100 --hidden-sizes 256 128 64 --dropout 0.4 \
    --lr 1e-3 --format gif --fps 4 --save-frame --seed 7
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 60 | Number of training epochs |
| `--sample-every` | 5 | Capture a snapshot every N epochs |
| `--hidden-sizes` | 128 64 | Hidden layer sizes (space-separated) |
| `--dropout` | 0.3 | Dropout probability |
| `--n-per-class` | 300 | Samples per shape class |
| `--lr` | 3e-4 | Adam learning rate |
| `--batch-size` | 32 | Mini-batch size |
| `--seed` | 42 | Random seed |
| `--device` | auto | Torch device (`cpu` / `cuda`) |
| `--output-dir` | outputs | Where snapshots and animations are written |
| `--format` | mp4 | Animation format (`mp4` or `gif`) |
| `--fps` | 2 | Animation frames per second |
| `--top-k-edges` | 8 | Strongest edges drawn per node |
| `--save-frame` | off | Also save the final frame as a PNG |
| `--render-only` | off | Skip training, re-render from `snapshots.npz` |
| `--quiet` | off | Suppress per-epoch progress output |

Run `python -m src.main --help` for the full list. The standalone renderer also has its own CLI: `python -m src.visualize --input outputs/snapshots.npz --output anim.gif`.

## Visualization Guide

- **Left Panel**: Input sample (32x32 image) with predicted class and confidence score
- **Middle Panel**: Network diagram
  - **Nodes**: Color and size represent activation magnitude (consistent global scale across frames)
  - **Edges**:
    - Green = positive weight, red = negative weight
    - Thickness and opacity proportional to |weight|
    - Top-k connections per node shown for clarity
  - **Red ring**: Indicates neuron dropped by dropout in current epoch snapshot
- **Right Top Panel**: Training/validation loss and validation accuracy curves over epochs
- **Right Bottom Panel**: Class probability distribution bars

## Project Structure

```
neural-net-mapper/
├── pyproject.toml                # Project metadata and pytest configuration
├── requirements.txt              # Python dependencies
├── assets/                       # README images/diagrams
├── .github/workflows/ci.yml     # GitHub Actions test workflow
├── data/
│   ├── __init__.py
│   └── generate_dataset.py       # Synthetic dataset generation (centered shapes, jitter, fill/outline)
├── src/
│   ├── __init__.py
│   ├── model.py                  # MLP architecture with dropout (Kaiming initialization)
│   ├── train.py                  # Training loop and snapshot capture
│   ├── visualize.py              # Animation renderer
│   └── main.py                   # CLI entry point
├── tests/                        # Pytest suite (dataset, model, training, rendering, CLI)
└── outputs/                      # Generated animations and snapshots
```

## Testing

```bash
pip install pytest
pytest
```

The suite covers dataset generation, model construction/validation, a tiny end-to-end training run, snapshot integrity, animation rendering, and the CLI. It runs in CI on every push and pull request.

## Features

- **Synthetic Dataset**: Generates geometric shapes (circles, squares, triangles) with configurable jitter and rendering styles
- **Dropout Visualization**: Real-time display of which neurons are dropped during training
- **Weight Analysis**: Visual representation of connection strengths and signs
- **Training Metrics**: Train loss, validation loss, and validation accuracy tracked and plotted live
- **Leak-Free Preprocessing**: Input standardization uses training-set statistics only (computed after the train/val split)
- **Flexible Architecture**: Configurable hidden layer sizes and dropout rates from the CLI
- **Reproducible**: Seed-based initialization; the dropout-mask capture pass is RNG-isolated so it never perturbs training

## Requirements

- Python 3.9+
- PyTorch
- Matplotlib
- NumPy
- Pillow
- scikit-learn
