```markdown
# neural-net-mapper

An interactive Python tool that trains a multi-layer perceptron (MLP) with dropout on a synthetic shapes dataset and maps its inner workings over time. It visualizes neuron activations, weight magnitudes/signs, predictions, and live training loss/accuracy through animated network diagrams using Matplotlib.

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train and render** (saves `outputs/animation.mp4`, falls back to `.gif` if needed)

   ```bash
   python -m src.main
   ```

3. **Re-render from saved snapshots**

   ```bash
   python -m src.visualize
   ```

**Configuration**: Tune training parameters in `src/train.py` (`train_model()` function arguments: `epochs`, `sample_every`, `hidden_sizes`, `dropout`, `seed`, and dataset options).

## Visualization Guide

- **Left Panel**: Input sample (32×32 image) with predicted class and confidence score
- **Middle Panel**: Network diagram
  - **Nodes**: Color and size represent activation magnitude (consistent global scale across frames)
  - **Edges**: 
    - Green = positive weight, red = negative weight
    - Thickness and opacity proportional to |weight|
    - Top-k connections per node shown for clarity
  - **Red ring**: Indicates neuron dropped by dropout in current epoch snapshot
- **Right Top Panel**: Training loss and accuracy curves over epochs
- **Right Bottom Panel**: Class probability distribution bars

## Project Structure

```
neural-net-mapper/
├── requirements.txt              # Python dependencies
├── data/
│   └── generate_dataset.py      # Synthetic dataset generation (centered shapes, jitter, fill/outline)
├── src/
│   ├── model.py                 # MLP architecture with dropout (Kaiming initialization)
│   ├── train.py                 # Training loop and snapshot capture
│   ├── visualize.py             # Animation renderer
│   └── main.py                  # Entry point
└── outputs/                     # Generated animations and snapshots
```

## Features

- **Synthetic Dataset**: Generates geometric shapes (circles, squares, triangles) with configurable jitter and rendering styles
- **Dropout Visualization**: Real-time display of which neurons are dropped during training
- **Weight Analysis**: Visual representation of connection strengths and signs
- **Training Metrics**: Live loss and accuracy tracking
- **Flexible Architecture**: Configurable hidden layer sizes and dropout rates
- **Reproducible**: Seed-based random initialization for consistent results

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pillow (PIL)

## License

MIT
```