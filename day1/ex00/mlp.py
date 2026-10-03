import torch
from torch import nn


class MLP(nn.Module):
    """A two-layer neural network."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        """Initialize the neural network with the size of each layer."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward step of the neural network."""
        return self.net.forward(x)


def main():
    """Test day1/ex00."""
    torch.manual_seed(42)
    model = MLP(4, 8, 3)
    print(model)
    out = model(torch.randn(5, 4))
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
