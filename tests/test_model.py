import pytest
import torch

from src.model import ShapeMLP


class TestShapeMLP:
    def test_forward_output_shape(self):
        model = ShapeMLP(input_size=16, hidden_sizes=[8, 4], num_classes=3)
        x = torch.randn(5, 16)
        out = model(x)
        assert out.shape == (5, 3)

    def test_single_hidden_layer(self):
        model = ShapeMLP(input_size=10, hidden_sizes=[6], num_classes=2)
        out = model(torch.randn(1, 10))
        assert out.shape == (1, 2)

    def test_config_is_stored_and_copied(self):
        hidden = [8, 4]
        model = ShapeMLP(input_size=16, hidden_sizes=hidden, num_classes=3, dropout_prob=0.2)
        hidden.append(99)  # external mutation must not affect the model
        assert model.hidden_sizes == [8, 4]
        assert model.input_size == 16
        assert model.num_classes == 3
        assert model.dropout_prob == 0.2

    def test_dropout_layers_present(self):
        model = ShapeMLP(input_size=16, hidden_sizes=[8, 4], dropout_prob=0.4)
        dropouts = [m for m in model.model if isinstance(m, torch.nn.Dropout)]
        assert len(dropouts) == 2
        assert all(d.p == 0.4 for d in dropouts)

    def test_eval_mode_is_deterministic(self):
        model = ShapeMLP(input_size=16, hidden_sizes=[8], dropout_prob=0.5)
        model.eval()
        x = torch.randn(3, 16)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_size": 0, "hidden_sizes": [8]},
            {"input_size": -1, "hidden_sizes": [8]},
            {"input_size": 16, "hidden_sizes": []},
            {"input_size": 16, "hidden_sizes": [8, 0]},
            {"input_size": 16, "hidden_sizes": [8], "num_classes": 0},
            {"input_size": 16, "hidden_sizes": [8], "dropout_prob": 1.0},
            {"input_size": 16, "hidden_sizes": [8], "dropout_prob": -0.1},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            ShapeMLP(**kwargs)

    def test_bias_initialized_to_zero(self):
        model = ShapeMLP(input_size=16, hidden_sizes=[8], num_classes=3)
        for module in model.model:
            if isinstance(module, torch.nn.Linear):
                assert torch.all(module.bias == 0)
