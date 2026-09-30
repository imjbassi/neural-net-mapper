# neural-net-mapper

[![CI](https://github.com/imjbassi/neural-net-mapper/actions/workflows/ci.yml/badge.svg)](https://github.com/imjbassi/neural-net-mapper/actions/workflows/ci.yml)

Trains an MLP on a synthetic shapes dataset (circles, squares, triangles) and renders an animated map of its inner workings: neuron activations, weight signs and magnitudes, dropout, predictions, and live loss/accuracy curves.

![Training animation demo](assets/demo_animation.gif)

*The network learning to classify randomly placed outline shapes over 80 epochs. Blue edges = positive weights, red = negative; node color/size = activation; orange ring = neuron dropped by dropout.*

## Quick Start

```bash
pip install -r requirements.txt
python -m src.main                  # train + render outputs/animation.mp4 (or .gif)
python -m src.main --render-only    # re-render from saved snapshots
```

MP4 export uses system ffmpeg, the `imageio-ffmpeg` bundled binary (`pip install imageio-ffmpeg`), or falls back to GIF.

## Configuration

All options are CLI flags — see `python -m src.main --help`. The most useful:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs`, `--sample-every` | 60, 5 | Training length and snapshot frequency |
| `--hidden-sizes` | 128 64 | Hidden layer sizes |
| `--dropout`, `--lr`, `--batch-size`, `--seed` | 0.3, 3e-4, 32, 42 | Training hyperparameters |
| `--uncentered`, `--outline`, `--jitter` | off, off, 2 | Dataset difficulty |
| `--theme` | light | `light`, `dark` (neon glow), or `paper` (academic style) |
| `--smooth`, `--fps`, `--dpi` | 1, 2, 120 | Frame interpolation and output quality |
| `--format`, `--output-dir`, `--save-frame` | mp4, outputs, off | Output options |

Reproduce the demo above:

```bash
python -m src.main --epochs 80 --sample-every 1 --n-per-class 400 \
    --uncentered --outline --theme paper --smooth 4 --fps 24 --dpi 150
```

## Development

```bash
pip install pytest && pytest
```

The suite (56 tests) covers dataset generation, the model, an end-to-end training run, rendering, and the CLI, and runs in CI on every push. Notable implementation details: inputs are standardized with global train-set statistics only (no leakage, no per-pixel blow-ups), and the dropout-mask capture pass is RNG-isolated so snapshots never perturb training.
