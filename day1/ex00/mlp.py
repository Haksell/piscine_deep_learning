import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)


def main():
    torch.manual_seed(42)
    model = MLP(4, 8, 3)
    print(model)
    out = model(torch.randn(5, 4))
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
