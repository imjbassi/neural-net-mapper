import torch.nn as nn

class ShapeMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=3, dropout_prob=0.5):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)

        # Kaiming init for ReLU MLPs
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
