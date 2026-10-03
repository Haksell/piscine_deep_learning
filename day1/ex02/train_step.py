import torch
from torch import nn, optim

# Due to this import, run with `uv run -m ex02.train_step`
from ex00.mlp import MLP


def one_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    """Run one gradient descent step and return the loss."""
    out = model(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    """Test day1/ex02."""
    torch.manual_seed(0)
    x = torch.randn(5)
    y = torch.rand(3)
    model = MLP(5, 8, 3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss1 = one_step(model, x, y, loss_fn, optimizer)
    loss2 = one_step(model, x, y, loss_fn, optimizer)
    print(f"Loss before step 1: {loss1:.4f}")
    print(f"Loss before step 2: {loss2:.4f}")
    print(f"Loss decreased: {loss2 < loss1}")


if __name__ == "__main__":
    main()
