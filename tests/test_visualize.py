import os

import numpy as np
import pytest

from src.visualize import load_snapshots, visualize_snapshots


def _make_snapshot(epoch, n_hist):
    rng = np.random.default_rng(epoch)
    probs = rng.random(3)
    probs /= probs.sum()
    return {
        "epoch": epoch,
        "acts": [rng.random(16).astype(np.float32),
                 rng.random(8).astype(np.float32),
                 probs.astype(np.float32)],
        "weights": [rng.standard_normal((8, 16)).astype(np.float32),
                    rng.standard_normal((3, 8)).astype(np.float32)],
        "loss_hist": list(rng.random(n_hist)),
        "train_loss_hist": list(rng.random(n_hist)),
        "acc_hist": list(rng.random(n_hist)),
        "img": rng.random((32, 32)).astype(np.float32),
        "label": 0,
        "pred": 1,
        "conf": float(probs.max()),
        "hidden_sizes": [16, 8],
        "dropout_masks": [rng.random(16) < 0.3, rng.random(8) < 0.3],
        "dropout_p": 0.3,
        "classes": ["circle", "square", "triangle"],
    }


@pytest.fixture
def snapshots():
    return [_make_snapshot(e, e) for e in (1, 2, 3)]


def test_renders_gif(snapshots, tmp_path):
    gif = str(tmp_path / "anim.gif")
    visualize_snapshots(snapshots, save_path_mp4=None, save_path_gif=gif, fps=2)
    assert os.path.exists(gif)
    assert os.path.getsize(gif) > 0


def test_renders_final_frame_png(snapshots, tmp_path):
    gif = str(tmp_path / "anim.gif")
    png = str(tmp_path / "frame.png")
    visualize_snapshots(snapshots, save_path_mp4=None, save_path_gif=gif, save_path_png=png)
    assert os.path.exists(png)
    assert os.path.getsize(png) > 0


def test_empty_snapshots_is_noop(tmp_path):
    gif = str(tmp_path / "anim.gif")
    visualize_snapshots([], save_path_mp4=None, save_path_gif=gif)
    assert not os.path.exists(gif)


def test_handles_legacy_snapshots_without_train_loss(snapshots, tmp_path):
    for s in snapshots:
        del s["train_loss_hist"]
    gif = str(tmp_path / "anim.gif")
    visualize_snapshots(snapshots, save_path_mp4=None, save_path_gif=gif)
    assert os.path.exists(gif)


def test_load_snapshots_roundtrip(snapshots, tmp_path):
    path = str(tmp_path / "snapshots.npz")
    np.savez(path, snapshots=np.array(snapshots, dtype=object))
    loaded = load_snapshots(path)
    assert len(loaded) == 3
    assert loaded[0]["epoch"] == 1
