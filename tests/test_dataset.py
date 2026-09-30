import numpy as np
import pytest

from data.generate_dataset import SHAPE_NAMES, generate_dataset, generate_shape_image


class TestGenerateShapeImage:
    @pytest.mark.parametrize("shape", SHAPE_NAMES)
    def test_returns_normalized_image(self, shape):
        img = generate_shape_image(shape, size=32)
        assert img.shape == (32, 32)
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    @pytest.mark.parametrize("shape", SHAPE_NAMES)
    def test_image_contains_shape_pixels(self, shape):
        # Background is white (1.0); the drawn shape adds dark pixels
        img = generate_shape_image(shape, size=32, fill=True)
        assert (img < 0.5).sum() > 10

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            generate_shape_image("hexagon")

    @pytest.mark.parametrize("shape", SHAPE_NAMES)
    def test_uncentered_and_outline_variants(self, shape):
        img = generate_shape_image(shape, size=32, centered=False, fill=False)
        assert img.shape == (32, 32)
        assert (img < 0.5).sum() > 0

    def test_custom_size(self):
        img = generate_shape_image("circle", size=64)
        assert img.shape == (64, 64)


class TestGenerateDataset:
    def test_shapes_and_labels(self):
        X, y = generate_dataset(n_per_class=5, size=32)
        assert X.shape == (15, 32 * 32)
        assert y.shape == (15,)
        assert X.dtype == np.float32
        assert y.dtype == np.int64
        assert set(np.unique(y)) == {0, 1, 2}
        # Balanced classes
        for label in range(3):
            assert (y == label).sum() == 5

    def test_seed_reproducibility(self):
        X1, y1 = generate_dataset(n_per_class=4, seed=123)
        X2, y2 = generate_dataset(n_per_class=4, seed=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_invalid_n_per_class(self):
        with pytest.raises(ValueError):
            generate_dataset(n_per_class=0)
