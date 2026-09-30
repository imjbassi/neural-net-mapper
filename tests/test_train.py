import os

import numpy as np
import pytest
import torch

from data.generate_dataset import SHAPE_NAMES
from src.model import ShapeMLP
from src.train import _forward_capture, train_model


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A tiny end-to-end training run shared across tests in this module."""
    out_dir = str(tmp_path_factory.mktemp("outputs"))
    snapshots = train_model(
        epochs=3,
        sample_every=2,
        hidden_sizes=[16, 8],
        dropout=0.3,
        n_per_class=12,
        seed=0,
        output_dir=out_dir,
        verbose=False,
    )
    return snapshots, out_dir


class TestTrainModel:
    def test_snapshots_captured(self, tiny_run):
        snapshots, _ = tiny_run
        # Epochs 1, 2, 3 (first, sample_every, last)
        assert [s["epoch"] for s in snapshots] == [1, 2, 3]

    def test_snapshot_contents(self, tiny_run):
        snapshots, _ = tiny_run
        s = snapshots[-1]
        expected_keys = {
            "epoch", "acts", "weights", "loss_hist", "train_loss_hist", "acc_hist",
            "img", "label", "pred", "conf", "hidden_sizes", "dropout_masks",
            "dropout_p", "classes",
        }
        assert expected_keys <= set(s.keys())
        # acts: one per hidden layer + output probs
        assert len(s["acts"]) == 3
        assert s["acts"][0].shape == (16,)
        assert s["acts"][1].shape == (8,)
        assert s["acts"][2].shape == (3,)
        np.testing.assert_allclose(s["acts"][2].sum(), 1.0, atol=1e-5)
        # weights: h1->h2 and h2->out
        assert len(s["weights"]) == 2
        assert s["weights"][0].shape == (8, 16)
        assert s["weights"][1].shape == (3, 8)
        assert s["img"].shape == (32, 32)
        assert s["classes"] == SHAPE_NAMES
        assert 0 <= s["conf"] <= 1
        assert len(s["loss_hist"]) == s["epoch"]
        assert len(s["train_loss_hist"]) == s["epoch"]
        assert len(s["acc_hist"]) == s["epoch"]

    def test_snapshots_saved_to_disk(self, tiny_run):
        snapshots, out_dir = tiny_run
        path = os.path.join(out_dir, "snapshots.npz")
        assert os.path.exists(path)
        loaded = np.load(path, allow_pickle=True)["snapshots"]
        assert len(loaded) == len(snapshots)

    def test_dropout_masks_shapes(self, tiny_run):
        snapshots, _ = tiny_run
        masks = snapshots[-1]["dropout_masks"]
        assert len(masks) == 2
        assert masks[0].shape == (16,)
        assert masks[1].shape == (8,)
        assert masks[0].dtype == bool

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epochs": 0},
            {"sample_every": 0},
            {"val_split": 0.0},
            {"val_split": 1.0},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, tmp_path):
        with pytest.raises(ValueError):
            train_model(n_per_class=6, output_dir=str(tmp_path), **kwargs)

    def test_reproducible_with_same_seed(self, tmp_path):
        common = dict(
            epochs=2, sample_every=2, hidden_sizes=[8], n_per_class=8, verbose=False
        )
        s1 = train_model(seed=7, output_dir=str(tmp_path / "a"), **common)
        s2 = train_model(seed=7, output_dir=str(tmp_path / "b"), **common)
        np.testing.assert_allclose(s1[-1]["loss_hist"], s2[-1]["loss_hist"])
        np.testing.assert_allclose(s1[-1]["acts"][0], s2[-1]["acts"][0])


class TestForwardCapture:
    def test_capture_does_not_perturb_global_rng(self):
        model = ShapeMLP(input_size=4, hidden_sizes=[6], num_classes=3)
        model.eval()
        x = torch.randn(1, 4)

        torch.manual_seed(999)
        before = torch.rand(3)
        torch.manual_seed(999)
        _forward_capture(model, x, epoch=5, capture_dropout=True)
        after = torch.rand(3)
        assert torch.equal(before, after)

    def test_capture_without_dropout(self):
        model = ShapeMLP(input_size=4, hidden_sizes=[6], num_classes=3)
        model.eval()
        acts, weights, logits, pred, conf, masks = _forward_capture(
            model, torch.randn(1, 4), capture_dropout=False
        )
        assert masks == []
        assert len(acts) == 2
        assert logits.shape == (1, 3)
        assert 0 <= pred < 3
