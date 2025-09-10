import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colormaps as mpl_cmaps


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
):
    """Draw nodes and connecting weights between layers.
    acts: [h1, h2, ..., probs]
    weights: [W(h1->h2), ..., W(hk->out)]  shape (out, in)
    """
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
            a_min, a_max = float(np.min(all_act)), float(np.max(all_act + 1e-8))
            if a_max - a_min < 1e-6:
                a_max = a_min + 1e-6
    else:
        a_min, a_max = float(act_min), float(act_max)
        if a_max - a_min < 1e-6:
            a_max = a_min + 1e-6
    cmap = cmap or mpl_cmaps.get_cmap('viridis')

    # Determine node positions per layer
    layer_sizes = [acts[0].shape[0]] + [a.shape[0] for a in acts[1:]]
    y_positions = []
    for lsize in layer_sizes:
        ys = np.linspace(0.1, 0.9, num=min(lsize, 12))
        y_positions.append(ys)

    # Draw connections using provided weights
    for li, W in enumerate(weights):
        # weights[li] connects acts[li] (in) -> acts[li+1] (out)
        x0, x1 = layer_x[li+1], layer_x[li+2]
        in_sz = W.shape[1]
        out_sz = W.shape[0]
        in_idx = np.linspace(0, min(in_sz, 12)-1, num=min(in_sz, 12), dtype=int)
        out_idx = np.linspace(0, min(out_sz, 12)-1, num=min(out_sz, 12), dtype=int)
        for oi, _ in enumerate(out_idx):
            y_out = y_positions[li+1][oi]
            # Draw only top-k strongest incoming edges for clarity
            sub_w = np.abs(W[out_idx[oi]][in_idx])
            k = min(top_k_edges, len(in_idx))
            top_idx = np.argsort(sub_w)[-k:]
            for ii in top_idx:
                ix = in_idx[ii]
                weight = W[out_idx[oi], ix]
                # color by sign; thickness/alpha by magnitude
                mag = float(abs(weight)) / (np.std(W) + 1e-6)
                mag = float(np.clip(mag, 0.0, 1.0))
                alpha = 0.25 + 0.6 * mag
                color = (0.1, 0.6, 0.2, alpha) if weight >= 0 else (0.8, 0.2, 0.2, alpha)
                y_in = y_positions[li][ii % len(y_positions[li])]
                ax.plot([x0, x1], [y_in, y_out], color=color, linewidth=0.6 + 2.4 * mag)

    # Draw nodes
    for li, a in enumerate(acts):
        xs = np.full_like(y_positions[li], layer_x[li+1], dtype=float)
        # subsample activations to max 12 nodes for clarity
        idx = np.linspace(0, a.shape[0]-1, num=min(a.shape[0], 12), dtype=int)
        a_sub = a[idx]
        norm = (a_sub - a_min) / (a_max - a_min)
        colors = cmap(np.clip(norm, 0, 1))
        sizes = 250 * (0.4 + 0.9 * np.clip(norm, 0, 1))  # scale node size by activation
        ax.scatter(xs, y_positions[li], s=sizes, c=colors, edgecolors='k', linewidths=0.4, zorder=3)
        # If dropout mask available for this layer (it's captured per hidden layer right after ReLU)
        if dropout_masks and li < len(dropout_masks):
            dmask = dropout_masks[li]
            d_idx = np.linspace(0, dmask.shape[0]-1, num=min(dmask.shape[0], 12), dtype=int)
            dropped = dmask[d_idx]
            for j, dropped_flag in enumerate(dropped):
                if dropped_flag:
                    ax.scatter([xs[j]], [y_positions[li][j]], s=300, facecolors='none', edgecolors='red', linewidths=1.2, zorder=4)

        # Layer labels at the top of each column
        if layer_labels and li < len(layer_labels):
            ax.text(xs[0], 0.98, layer_labels[li], ha='center', va='top', fontsize=8)

    # Output labels next to last layer
    if output_labels:
        xs_last = np.full_like(y_positions[-1], layer_x[-1], dtype=float)
        for j, name in enumerate(output_labels[: len(y_positions[-1])]):
            ax.text(xs_last[0] + 0.03, y_positions[-1][j], name, fontsize=8, va='center')

    # Legend
    ax.text(
        0.02,
        0.98,
        "Edges: green=+ red=-, thickness/alpha=|w|\nNodes: color/size=activation, red ring=dropout",
        transform=ax.transAxes,
        fontsize=8,
        va='top',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
    )


def visualize_snapshots(snapshots, save_path_mp4="outputs/animation.mp4", save_path_gif=None, fps=2, top_k_edges=8):
    # snapshots: list of dicts
    if isinstance(snapshots, np.ndarray):
        snapshots = list(snapshots)

    fig = plt.figure(figsize=(10, 6))

    # Layout: left input image; center network; right metrics
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 2.2, 1.6], height_ratios=[1, 1])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_net = fig.add_subplot(gs[:, 1])
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 2])

    # Prepare static elements
    ax_img.set_title("Input sample")
    ax_img.axis('off')
    im = ax_img.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1)
    pred_txt = ax_img.text(0.5, -0.08, "", transform=ax_img.transAxes, ha='center', va='top', fontsize=10)

    ax_metrics.set_title("Loss & Accuracy")
    ax_metrics.set_xlim(1, max(2, len(snapshots)))
    ax_metrics.set_ylim(0, 1.1)
    ax_metrics.grid(True, alpha=0.3)
    (loss_line,) = ax_metrics.plot([], [], label="Loss", color="#1f77b4")
    (acc_line,) = ax_metrics.plot([], [], label="Acc", color="#ff7f0e")
    ax_metrics.legend(loc="upper right")

    ax_bar.set_title("Class probabilities")
    classes = ["circle", "square", "triangle"]
    bars = ax_bar.bar(classes, [0, 0, 0], color=['#6baed6', '#9ecae1', '#c6dbef'])
    ax_bar.set_ylim(0, 1)

    # Precompute network x positions
    # acts: [h1, h2, ..., probs] -> L layers for drawing
    max_layers = max(len(s["acts"]) for s in snapshots)
    layer_x = np.linspace(0.1, 0.9, max_layers + 1)  # index from 1..L used in draw

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
    cmap = mpl_cmaps.get_cmap('viridis')

    title = fig.suptitle("", fontsize=12)

    def update(i):
        s = snapshots[i]
        epoch = s["epoch"]
        acts = s["acts"]
        weights = s["weights"]
        loss_hist = s["loss_hist"]
        acc_hist = s["acc_hist"]

        # input image
        im.set_data(s["img"])
        pred = s.get("pred", None)
        conf = s.get("conf", None)
        if pred is not None and conf is not None:
            classes = ["circle", "square", "triangle"]
            pred_txt.set_text(f"Pred: {classes[pred]} ({conf*100:.1f}%)")
        else:
            pred_txt.set_text("")

        # metrics
        xs = np.arange(1, len(loss_hist) + 1)
        loss_line.set_data(xs, loss_hist)
        acc_line.set_data(xs, acc_hist)
        ax_metrics.set_xlim(1, max(2, len(loss_hist)))
        y_max = max(1.0, max(loss_hist[-1], max(acc_hist) if acc_hist else 1.0) + 0.1)
        ax_metrics.set_ylim(0, y_max)

        # bar probs from last activation
        probs = acts[-1]
        for b, v in zip(bars, probs):
            b.set_height(float(v))

        # network
        layer_labels = [f"H{i+1}" for i in range(len(acts)-1)] + ["Out"]
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
            output_labels=s.get("classes", ["circle","square","triangle"]),
        )

        title.set_text(f"Epoch {epoch}  |  Loss: {loss_hist[-1]:.3f}  Acc: {acc_hist[-1]*100:.1f}%")
        return [im, loss_line, acc_line, *bars]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), blit=False, repeat=False)

    # Save with fallback if mp4 writer (ffmpeg) is unavailable
    if save_path_mp4:
        try:
            ani.save(save_path_mp4, fps=fps, dpi=120)
        except Exception:
            if save_path_gif is None:
                save_path_gif = save_path_mp4.replace('.mp4', '.gif')
            ani.save(save_path_gif, fps=fps)
    elif save_path_gif:
        ani.save(save_path_gif, fps=fps)
    plt.close(fig)


if __name__ == "__main__":
    data = np.load("outputs/snapshots.npz", allow_pickle=True)["snapshots"]
    visualize_snapshots(data)
