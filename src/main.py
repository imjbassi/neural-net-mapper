"""Main entry point for the neural network mapper application."""

import argparse
import os
import sys
import traceback
from pathlib import Path

# Add project root to path to allow running from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train_model
from src.visualize import load_snapshots, visualize_snapshots


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="neural-net-mapper",
        description=(
            "Train an MLP on a synthetic shapes dataset and render an animated "
            "visualization of its inner workings over training."
        ),
    )
    train_group = parser.add_argument_group("training")
    train_group.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    train_group.add_argument("--sample-every", type=int, default=5,
                             help="Capture a snapshot every N epochs (default: 5)")
    train_group.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64],
                             metavar="N", help="Hidden layer sizes (default: 128 64)")
    train_group.add_argument("--dropout", type=float, default=0.3,
                             help="Dropout probability (default: 0.3)")
    train_group.add_argument("--n-per-class", type=int, default=300,
                             help="Samples per shape class (default: 300)")
    train_group.add_argument("--lr", type=float, default=3e-4,
                             help="Adam learning rate (default: 3e-4)")
    train_group.add_argument("--batch-size", type=int, default=32,
                             help="Mini-batch size (default: 32)")
    train_group.add_argument("--seed", type=int, default=42,
                             help="Random seed (default: 42)")
    train_group.add_argument("--device", default=None,
                             help="Torch device, e.g. cpu or cuda (default: auto)")
    train_group.add_argument("--quiet", action="store_true",
                             help="Suppress per-epoch progress output")

    data_group = parser.add_argument_group("dataset difficulty")
    data_group.add_argument("--uncentered", action="store_true",
                            help="Scatter shapes randomly instead of centering (harder)")
    data_group.add_argument("--jitter", type=int, default=2,
                            help="Max pixel jitter for centered shapes (default: 2)")
    data_group.add_argument("--outline", action="store_true",
                            help="Draw shape outlines instead of filled shapes (harder)")
    data_group.add_argument("--thickness", type=int, default=2,
                            help="Outline thickness with --outline (default: 2)")

    viz_group = parser.add_argument_group("visualization")
    viz_group.add_argument("--output-dir", default="outputs",
                           help="Directory for snapshots and animations (default: outputs)")
    viz_group.add_argument("--format", choices=["mp4", "gif"], default="mp4",
                           help="Animation format; mp4 falls back to gif without ffmpeg")
    viz_group.add_argument("--fps", type=int, default=2,
                           help="Animation frames per second (default: 2)")
    viz_group.add_argument("--top-k-edges", type=int, default=8,
                           help="Strongest edges drawn per node (default: 8)")
    viz_group.add_argument("--theme", choices=["light", "dark"], default="light",
                           help="Visual theme (default: light)")
    viz_group.add_argument("--smooth", type=int, default=1,
                           help="Interpolated frames per snapshot for fluid motion "
                                "(default: 1 = off); raise --fps accordingly")
    viz_group.add_argument("--dpi", type=int, default=120,
                           help="Output resolution (default: 120)")
    viz_group.add_argument("--save-frame", action="store_true",
                           help="Also save the final frame as a PNG still")
    viz_group.add_argument("--render-only", action="store_true",
                           help="Skip training; re-render from saved snapshots.npz")

    return parser.parse_args(argv)


def main(argv=None):
    """Train a neural network model and visualize training snapshots.

    Returns:
        int: Exit code (0 for success, 1 for failure, 130 for user interrupt)
    """
    args = parse_args(argv)
    try:
        if args.render_only:
            snapshot_path = os.path.join(args.output_dir, "snapshots.npz")
            if not os.path.exists(snapshot_path):
                print(f"Error: no snapshots found at {snapshot_path}. "
                      "Run without --render-only first.", file=sys.stderr)
                return 1
            snapshots = load_snapshots(snapshot_path)
        else:
            snapshots = train_model(
                epochs=args.epochs,
                sample_every=args.sample_every,
                hidden_sizes=args.hidden_sizes,
                dropout=args.dropout,
                n_per_class=args.n_per_class,
                seed=args.seed,
                lr=args.lr,
                batch_size=args.batch_size,
                device=args.device,
                output_dir=args.output_dir,
                verbose=not args.quiet,
                centered=not args.uncentered,
                jitter=args.jitter,
                fill=not args.outline,
                thickness=args.thickness,
            )

        if not snapshots:
            print("Warning: No snapshots were generated during training.", file=sys.stderr)
            return 1

        os.makedirs(args.output_dir, exist_ok=True)
        png_path = os.path.join(args.output_dir, "final_frame.png") if args.save_frame else None
        viz_kwargs = dict(
            save_path_png=png_path,
            fps=args.fps,
            top_k_edges=args.top_k_edges,
            theme=args.theme,
            smooth=args.smooth,
            dpi=args.dpi,
        )
        if args.format == "gif":
            visualize_snapshots(
                snapshots,
                save_path_mp4=None,
                save_path_gif=os.path.join(args.output_dir, "animation.gif"),
                **viz_kwargs,
            )
        else:
            visualize_snapshots(
                snapshots,
                save_path_mp4=os.path.join(args.output_dir, "animation.mp4"),
                **viz_kwargs,
            )
        print(f"Done. Outputs written to {args.output_dir}/")
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
