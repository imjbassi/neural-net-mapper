import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as mpl_cmaps

DEFAULT_CLASSES = ["circle", "square", "triangle"]

# Visual themes for the rendered animation. "light" is the classic look;
# "dark" is a high-contrast neon style with glow effects, made for demos.
THEMES = {
    "light": {
        "bg": "#ffffff",
        "text": "#222222",
        "grid": "#888888",
        "pos_edge": (0.10, 0.60, 0.20),
        "neg_edge": (0.80, 0.20, 0.20),
        "node_cmap": "viridis",
        "node_border": "black",
        "dropout_ring": "red",
        "glow": False,
        "invert_input": False,
        "loss_color": "#1f77b4",
        "acc_color": "#ff7f0e",
        "bar_colors": ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdd0a2"],
        "legend_face": "white",
    },
    "dark": {
        "bg": "#0b0f19",
        "text": "#dbe4ff",
        "grid": "#94a3b8",
        "pos_edge": (0.00, 0.88, 0.56),   # neon green
        "neg_edge": (1.00, 0.30, 0.43),   # neon red/pink
        "node_cmap": "plasma",
        "node_border": "#dbe4ff",
        "dropout_ring": "#facc15",        # gold ring stands out on dark
        "glow": True,
        "invert_input": True,
        "loss_color": "#60a5fa",
        "acc_color": "#fbbf24",
        "bar_colors": ["#60a5fa", "#c084fc", "#34d399", "#f472b6", "#fbbf24"],
        "legend_face": "#141a2a",
    },
}


def _ensure_ffmpeg():
    """Make the ffmpeg writer available, using imageio-ffmpeg's bundled binary
    when no system ffmpeg is installed. Returns True if ffmpeg can be used."""
    if animation.writers.is_available("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return animation.writers.is_available("ffmpeg")


def _draw_network(
    ax,
    acts,
    weights,
    layer_x,
    dropout_masks=None,
    node_radius=0.03,
    top_k_edges=8,
    cmap=None,
    act_min=None,
    act_max=None,
    layer_labels=None,
    output_labels=None,
    theme=None,
):
    """Draw nodes and connecting weights between layers.

    Args:
        ax: Matplotlib axis to draw on
        acts: List of activation arrays [h1, h2, ..., probs]
        weights: List of weight matrices [W(h1->h2), ..., W(hk->out)] with shape (out, in)
        layer_x: X-coordinates for each layer
        dropout_masks: Optional list of boolean masks indicating dropped nodes
        node_radius: Radius for drawing nodes (unused, kept for API compatibility)
        top_k_edges: Number of strongest edges to draw per output node
        cmap: Colormap for node coloring
        act_min: Minimum activation value for normalization
        act_max: Maximum activation value for normalization
        layer_labels: Labels for each layer
        output_labels: Labels for output nodes
        theme: Theme dict from THEMES (defaults to light)
    """
    theme = theme or THEMES["light"]
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Normalize activations for color intensity (prefer global range if provided)
    if act_min is None or act_max is None:
        all_act = np.concatenate([a.flatten() for a in acts[:-1]]) if len(acts) > 1 else acts[0]
        if all_act.size == 0:
            a_min, a_max = 0.0, 1.0
        else:
            a_min = float(np.min(all_act))
            a_max = float(np.max(all_act))
            if a_max - a_min < 1e-6:
                a_max = a_min + 1e-6
    else:
        a_min, a_max = float(act_min), float(act_max)
        if a_max - a_min < 1e-6:
            a_max = a_min + 1e-6

    cmap = cmap or mpl_cmaps.get_cmap(theme["node_cmap"])

    # Determine node positions per layer (max 12 nodes displayed per layer)
    max_nodes_per_layer = 12
    layer_sizes = [acts[0].shape[0]] + [a.shape[0] for a in acts[1:]]
    y_positions = []
    for lsize in layer_sizes:
        ys = np.linspace(0.1, 0.9, num=min(lsize, max_nodes_per_layer))
        y_positions.append(ys)

    # Draw connections using provided weights
    for li, W in enumerate(weights):
        # weights[li] connects acts[li] (in) -> acts[li+1] (out)
        x0, x1 = layer_x[li], layer_x[li + 1]
        in_sz = W.shape[1]
        out_sz = W.shape[0]
        in_idx = np.linspace(0, in_sz - 1, num=min(in_sz, max_nodes_per_layer), dtype=int)
        out_idx = np.linspace(0, out_sz - 1, num=min(out_sz, max_nodes_per_layer), dtype=int)

        w_std = np.std(W) + 1e-6

        for oi, out_node_idx in enumerate(out_idx):
            y_out = y_positions[li + 1][oi]
            # Draw only top-k strongest incoming edges for clarity
            sub_w = np.abs(W[out_node_idx][in_idx])
            k = min(top_k_edges, len(in_idx))
            top_idx = np.argsort(sub_w)[-k:]

            for ii in top_idx:
                ix = in_idx[ii]
                weight = W[out_node_idx, ix]
                # Color by sign; thickness/alpha by magnitude
                mag = float(abs(weight)) / w_std
                mag = float(np.clip(mag, 0.0, 1.0))
                alpha = 0.25 + 0.6 * mag
                base = theme["pos_edge"] if weight >= 0 else theme["neg_edge"]
                y_in = y_positions[li][ii]
                lw = 0.6 + 2.4 * mag
                if theme["glow"]:
                    # Wide translucent pass underneath gives a neon glow
                    ax.plot([x0, x1], [y_in, y_out], color=(*base, alpha * 0.25),
                            linewidth=lw * 3.0, zorder=1.5, solid_capstyle='round')
                ax.plot([x0, x1], [y_in, y_out], color=(*base, alpha),
                        linewidth=lw, zorder=2, solid_capstyle='round')

    # Draw nodes
    for li, a in enumerate(acts):
        xs = np.full_like(y_positions[li], layer_x[li], dtype=float)
        # Subsample activations to max 12 nodes for clarity
        idx = np.linspace(0, a.shape[0] - 1, num=min(a.shape[0], max_nodes_per_layer), dtype=int)
        a_sub = a[idx]
        norm = (a_sub - a_min) / (a_max - a_min)
        colors = cmap(np.clip(norm, 0, 1))
        sizes = 250 * (0.4 + 0.9 * np.clip(norm, 0, 1))  # Scale node size by activation
        if theme["glow"]:
            # Soft halo behind each node
            ax.scatter(xs, y_positions[li], s=sizes * 2.6, c=colors, alpha=0.25,
                       edgecolors='none', zorder=2.5)
        ax.scatter(xs, y_positions[li], s=sizes, c=colors,
                   edgecolors=theme["node_border"], linewidths=0.4, zorder=3)

        # If dropout mask available for this layer
        if dropout_masks and li < len(dropout_masks):
            dmask = dropout_masks[li]
            d_idx = np.linspace(0, dmask.shape[0] - 1, num=min(dmask.shape[0], max_nodes_per_layer), dtype=int)
            dropped = dmask[d_idx]
            for j, dropped_flag in enumerate(dropped):
                if dropped_flag:
                    ax.scatter([xs[j]], [y_positions[li][j]], s=300, facecolors='none',
                             edgecolors=theme["dropout_ring"], linewidths=1.4, zorder=4)

        # Layer labels at the top of each column
        if layer_labels and li < len(layer_labels):
            ax.text(xs[0], 1.0, layer_labels[li], ha='center', va='top', fontsize=8,
                    color=theme["text"])

    # Output labels next to last layer
    if output_labels:
        xs_last = np.full_like(y_positions[-1], layer_x[-1], dtype=float)
        for j, name in enumerate(output_labels[:len(y_positions[-1])]):
            ax.text(xs_last[0] + 0.03, y_positions[-1][j], name, fontsize=8, va='center',
                    color=theme["text"])

    # Legend
    ax.text(
        0.02,
        0.02,
        "Edges: green=+ red=-, thickness/alpha=|w|\nNodes: color/size=activation, ring=dropout",
        transform=ax.transAxes,
        fontsize=7,
        va='bottom',
        ha='left',
        color=theme["text"],
        bbox=dict(boxstyle='round,pad=0.25', facecolor=theme["legend_face"],
                  edgecolor='none', alpha=0.75),
    )


def _build_frames(snapshots, smooth):
    """Expand snapshots into render frames, linearly interpolating activations,
    weights, and output probabilities between consecutive snapshots.

    Args:
        snapshots: List of snapshot dicts
        smooth: Frames per snapshot transition (1 = no interpolation)

    Returns:
        List of frame dicts (shallow copies of snapshots with blended arrays)
    """
    if smooth <= 1 or len(snapshots) < 2:
        return list(snapshots)

    frames = []
    for i in range(len(snapshots) - 1):
        a, b = snapshots[i], snapshots[i + 1]
        compatible = (
            len(a["acts"]) == len(b["acts"])
            and all(x.shape == y.shape for x, y in zip(a["acts"], b["acts"]))
            and len(a["weights"]) == len(b["weights"])
        )
        for k in range(smooth):
            t = k / smooth
            base = a if t < 0.5 else b
            frame = dict(base)
            if compatible:
                blended_acts = [(1 - t) * x + t * y for x, y in zip(a["acts"], b["acts"])]
                frame["acts"] = blended_acts
                frame["weights"] = [
                    (1 - t) * x + t * y for x, y in zip(a["weights"], b["weights"])
                ]
                probs = blended_acts[-1]
                frame["pred"] = int(np.argmax(probs))
                frame["conf"] = float(np.max(probs))
            frames.append(frame)
    frames.append(dict(snapshots[-1]))
    return frames


def visualize_snapshots(
    snapshots,
    save_path_mp4="outputs/animation.mp4",
    save_path_gif=None,
    save_path_png=None,
    fps=2,
    top_k_edges=8,
    theme="light",
    smooth=1,
    dpi=120,
):
    """Create an animated visualization of network training snapshots.

    Args:
        snapshots: List of snapshot dictionaries containing network state
        save_path_mp4: Path to save MP4 animation (None to skip)
        save_path_gif: Path to save GIF animation (None to skip, used as fallback if MP4 fails)
        save_path_png: Optional path to save the final frame as a still image
        fps: Frames per second for animation
        top_k_edges: Number of strongest edges to draw per output node
        theme: Visual theme name ("light" or "dark")
        smooth: Interpolated frames per snapshot transition (1 = off).
            Raise fps accordingly, e.g. smooth=4 with fps=8.
        dpi: Output resolution
    """
    if isinstance(snapshots, np.ndarray):
        snapshots = list(snapshots)

    if not snapshots:
        return

    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}. Available: {sorted(THEMES)}")
    th = THEMES[theme]

    classes = [str(c) for c in snapshots[0].get("classes", DEFAULT_CLASSES)]
    frames = _build_frames(snapshots, smooth)

    fig = plt.figure(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(th["bg"])

    # Layout: left input image; center network; right metrics
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 2.2, 1.6], height_ratios=[1, 1],
                          left=0.04, right=0.96, top=0.88, bottom=0.07,
                          wspace=0.25, hspace=0.35)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_net = fig.add_subplot(gs[:, 1])
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 2])

    for ax in (ax_img, ax_net, ax_metrics, ax_bar):
        ax.set_facecolor(th["bg"])
    for ax in (ax_metrics, ax_bar):
        for spine in ax.spines.values():
            spine.set_color(th["grid"])
            spine.set_alpha(0.5)
        ax.tick_params(colors=th["text"], labelsize=8)

    # Prepare static elements
    ax_img.set_title("Input sample", color=th["text"])
    ax_img.axis('off')
    im = ax_img.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1)
    pred_txt = ax_img.text(0.5, -0.08, "", transform=ax_img.transAxes, ha='center',
                           va='top', fontsize=11, color=th["text"])

    has_train_loss = "train_loss_hist" in snapshots[0]
    ax_metrics.set_title("Loss & Accuracy", color=th["text"])
    ax_metrics.set_xlim(1, max(2, len(snapshots)))
    ax_metrics.set_ylim(0, 1.1)
    ax_metrics.grid(True, alpha=0.15, color=th["grid"])
    (loss_line,) = ax_metrics.plot([], [], label="Val loss", color=th["loss_color"], linewidth=1.8)
    (train_loss_line,) = ax_metrics.plot(
        [], [], label="Train loss", color=th["loss_color"], linestyle="--", alpha=0.6
    )
    (acc_line,) = ax_metrics.plot([], [], label="Val acc", color=th["acc_color"], linewidth=1.8)
    if not has_train_loss:
        train_loss_line.set_visible(False)
    legend = ax_metrics.legend(loc="upper right", fontsize=7, facecolor=th["legend_face"],
                               edgecolor='none', labelcolor=th["text"])
    legend.get_frame().set_alpha(0.75)

    ax_bar.set_title("Class probabilities", color=th["text"])
    bars = ax_bar.bar(classes, [0] * len(classes), color=th["bar_colors"][:len(classes)])
    ax_bar.set_ylim(0, 1)

    # Precompute network x positions
    max_layers = max(len(s["acts"]) for s in snapshots)
    layer_x = np.linspace(0.1, 0.9, max_layers)

    # Compute global activation scale across all snapshots (hidden layers only)
    hidden_vals = []
    for s in snapshots:
        for a in s["acts"][:-1]:
            if isinstance(a, np.ndarray):
                hidden_vals.append(a.ravel())
    if hidden_vals:
        hidden_vals = np.concatenate(hidden_vals)
        act_min = float(np.min(hidden_vals))
        act_max = float(np.max(hidden_vals))
    else:
        act_min, act_max = 0.0, 1.0
    cmap = mpl_cmaps.get_cmap(th["node_cmap"])

    title = fig.suptitle("", fontsize=14, color=th["text"], fontweight='bold')

    def update(i):
        s = frames[i]
        epoch = s["epoch"]
        acts = s["acts"]
        weights = s["weights"]
        loss_hist = s["loss_hist"]
        train_loss_hist = s.get("train_loss_hist", [])
        acc_hist = s["acc_hist"]

        # Input image (inverted on dark themes so shapes glow on the dark bg)
        img = s["img"]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
        im.set_data(1.0 - img_norm if th["invert_input"] else img_norm)
        pred = s.get("pred", None)
        conf = s.get("conf", None)
        if pred is not None and conf is not None:
            pred_txt.set_text(f"Pred: {classes[pred]} ({conf * 100:.1f}%)")
        else:
            pred_txt.set_text("")

        # Metrics
        xs = np.arange(1, len(loss_hist) + 1)
        loss_line.set_data(xs, loss_hist)
        if train_loss_hist:
            train_loss_line.set_data(np.arange(1, len(train_loss_hist) + 1), train_loss_hist)
        acc_line.set_data(xs, acc_hist)
        ax_metrics.set_xlim(1, max(2, len(loss_hist)))
        y_candidates = [1.0]
        if loss_hist:
            y_candidates.append(max(loss_hist))
        if train_loss_hist:
            y_candidates.append(max(train_loss_hist))
        if acc_hist:
            y_candidates.append(max(acc_hist))
        ax_metrics.set_ylim(0, max(y_candidates) + 0.1)

        # Bar chart of class probabilities
        probs = acts[-1]
        for b, v in zip(bars, probs):
            b.set_height(float(v))

        # Network visualization
        layer_labels = [f"H{i + 1}" for i in range(len(acts) - 1)] + ["Out"]
        _draw_network(
            ax_net,
            acts,
            weights,
            layer_x,
            dropout_masks=s.get("dropout_masks", None),
            top_k_edges=top_k_edges,
            cmap=cmap,
            act_min=act_min,
            act_max=act_max,
            layer_labels=layer_labels,
            output_labels=classes,
            theme=th,
        )

        title.set_text(f"Epoch {epoch}  |  Loss: {loss_hist[-1]:.3f}  Acc: {acc_hist[-1] * 100:.1f}%")
        return [im, loss_line, train_loss_line, acc_line, *bars]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False, repeat=False)

    # Save with fallback if mp4 writer (ffmpeg) is unavailable
    if save_path_mp4:
        try:
            if not _ensure_ffmpeg():
                raise RuntimeError("ffmpeg is not available")
            ani.save(save_path_mp4, fps=fps, dpi=dpi,
                     savefig_kwargs={"facecolor": th["bg"]})
        except Exception:
            if save_path_gif is None:
                save_path_gif = save_path_mp4.replace('.mp4', '.gif')
            ani.save(save_path_gif, fps=fps, savefig_kwargs={"facecolor": th["bg"]})
    elif save_path_gif:
        ani.save(save_path_gif, fps=fps, savefig_kwargs={"facecolor": th["bg"]})

    # Optionally save the final frame as a still image
    if save_path_png:
        update(len(frames) - 1)
        fig.savefig(save_path_png, dpi=dpi, bbox_inches='tight', facecolor=th["bg"])

    plt.close(fig)


def load_snapshots(path="outputs/snapshots.npz"):
    """Load saved snapshots from an .npz file produced by train_model."""
    return list(np.load(path, allow_pickle=True)["snapshots"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-render an animation from saved snapshots.")
    parser.add_argument("--input", default="outputs/snapshots.npz", help="Path to snapshots.npz")
    parser.add_argument("--output", default="outputs/animation.mp4",
                        help="Output animation path (.mp4 or .gif)")
    parser.add_argument("--png", default=None, help="Optional path for a final-frame still image")
    parser.add_argument("--fps", type=int, default=2, help="Animation frames per second")
    parser.add_argument("--top-k-edges", type=int, default=8,
                        help="Strongest edges drawn per node")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light",
                        help="Visual theme")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Interpolated frames per snapshot (1 = off)")
    parser.add_argument("--dpi", type=int, default=120, help="Output resolution")
    args = parser.parse_args()

    data = load_snapshots(args.input)
    common = dict(save_path_png=args.png, fps=args.fps, top_k_edges=args.top_k_edges,
                  theme=args.theme, smooth=args.smooth, dpi=args.dpi)
    if args.output.endswith(".gif"):
        visualize_snapshots(data, save_path_mp4=None, save_path_gif=args.output, **common)
    else:
        visualize_snapshots(data, save_path_mp4=args.output, **common)
