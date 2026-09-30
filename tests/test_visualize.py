import os

import numpy as np
import pytest

from src.visualize import THEMES, _build_frames, load_snapshots, visualize_snapshots


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


def test_renders_dark_theme(snapshots, tmp_path):
    gif = str(tmp_path / "anim.gif")
    png = str(tmp_path / "frame.png")
    visualize_snapshots(snapshots, save_path_mp4=None, save_path_gif=gif,
                        save_path_png=png, theme="dark")
    assert os.path.exists(gif)
    assert os.path.exists(png)


def test_renders_paper_theme(snapshots, tmp_path):
    gif = str(tmp_path / "anim.gif")
    png = str(tmp_path / "frame.png")
    visualize_snapshots(snapshots, save_path_mp4=None, save_path_gif=gif,
                        save_path_png=png, theme="paper")
    assert os.path.exists(gif)
    assert os.path.exists(png)


def test_unknown_theme_raises(snapshots, tmp_path):
    with pytest.raises(ValueError, match="Unknown theme"):
        visualize_snapshots(snapshots, save_path_mp4=None,
                            save_path_gif=str(tmp_path / "x.gif"), theme="neon")


def test_themes_have_consistent_keys():
    keys = {frozenset(t.keys()) for t in THEMES.values()}
    assert len(keys) == 1


def test_build_frames_no_smoothing(snapshots):
    frames = _build_frames(snapshots, smooth=1)
    assert len(frames) == len(snapshots)


def test_build_frames_interpolates(snapshots):
    frames = _build_frames(snapshots, smooth=4)
    # (n-1) * smooth transitions + final frame
    assert len(frames) == (len(snapshots) - 1) * 4 + 1
    # Midpoint frame between snapshots 0 and 1 blends activations
    mid = frames[2]  # t = 0.5 of first transition
    expected = 0.5 * snapshots[0]["acts"][0] + 0.5 * snapshots[1]["acts"][0]
    np.testing.assert_allclose(mid["acts"][0], expected, rtol=1e-5)
    # Blended probabilities stay normalized
    np.testing.assert_allclose(mid["acts"][-1].sum(), 1.0, atol=1e-5)
    # First and last frames match the original endpoints
    np.testing.assert_allclose(frames[0]["acts"][0], snapshots[0]["acts"][0])
    np.testing.assert_allclose(frames[-1]["acts"][0], snapshots[-1]["acts"][0])


def test_renders_smoothed_gif(snapshots, tmp_path):
    gif = str(tmp_path / "smooth.gif")
    visualize_snapshots(snapshots, save_path_mp4=None, save_path_gif=gif, smooth=3)
    assert os.path.exists(gif)
    assert os.path.getsize(gif) > 0


def test_load_snapshots_roundtrip(snapshots, tmp_path):
    path = str(tmp_path / "snapshots.npz")
    np.savez(path, snapshots=np.array(snapshots, dtype=object))
    loaded = load_snapshots(path)
    assert len(loaded) == 3
    assert loaded[0]["epoch"] == 1
