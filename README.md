# neural-net-mapper

An interactive Python tool that trains a multi-layer perceptron (MLP) with dropout on a synthetic shapes dataset and maps its inner workings over time. It visualizes neuron activations, weight magnitudes/signs, predictions, and live training loss/accuracy through animated network diagrams using Matplotlib.

## Quick start

1. Install deps

```powershell
pip install -r requirements.txt
```

2. Train and render (saves `outputs/animation.mp4`, falls back to `.gif` if needed)

```powershell
python -m src.main
```

Re-render from saved snapshots:

```powershell
python -m src.visualize
```

Tune training in `src/train.py` (`train_model()` args: epochs, sample_every, hidden_sizes, dropout, seed, dataset options).

## What you’ll see
- Left: input sample (32x32) and predicted class with confidence
- Middle: network diagram
	- Nodes: color/size = activation (consistent global scale)
	- Edges: green=positive, red=negative; thickness/alpha ∝ |weight|; top-k per node for clarity
	- Red ring = dropped by dropout in that epoch snapshot
- Right top: loss and accuracy over epochs
- Right bottom: class probability bars

## Files
- `requirements.txt` dependencies
- `data/generate_dataset.py` synthetic dataset generation (centered shapes, jitter, fill/outline)
- `src/model.py` MLP with dropout (Kaiming init)
- `src/train.py` training loop and snapshot capture
- `src/visualize.py` animation renderer
- `src/main.py` simple entrypoint
