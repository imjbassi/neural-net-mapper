```python
import torch.nn as nn


class ShapeMLP(nn.Module):
    """Multi-layer perceptron for shape classification.
    
    Args:
        input_size: Dimension of input features
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes (default: 3)
        dropout_prob: Dropout probability for regularization (default: 0.5)
    """
    
    def __init__(self, input_size, hidden_sizes, num_classes=3, dropout_prob=0.5):
        super().__init__()
        
        # Build sequential layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_size = hidden_size
        
        # Output layer (no activation or dropout)
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights using Kaiming initialization for ReLU networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize linear layer weights with Kaiming normal initialization."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.model(x)
```