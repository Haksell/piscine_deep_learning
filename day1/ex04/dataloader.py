import torch
from torch import nn, optim


class MLP(nn.Module):
    """A two-layer neural network."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        """Initialize the neural network with the size of each layer."""
        super().__init__()
        self.net = nn.Sequential(
            # Normalize height, weight and hair length on the same scale
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward step of the neural network."""
        return self.net(x)


def generate_dataset():
    """Generate a dataset of men and women with heights, weights and hair lengths."""
    N = 1000

    men_heights = torch.randn(N) * 7 + 177
    men_bmis = torch.randn(N) * 4 + 25
    men_weights = men_bmis * (men_heights / 100) ** 2
    men_hair_lengths = torch.nn.ReLU()(6 + torch.randn(N) * 4)

    women_heights = torch.randn(N) * 5 + 162
    women_bmis = torch.randn(N) * 2 + 22
    women_weights = women_bmis * (women_heights / 100) ** 2
    women_hair_lengths = torch.nn.ReLU()(15 + torch.randn(N) * 6)

    heights = torch.cat((men_heights, women_heights))
    weights = torch.cat((men_weights, women_weights))
    hair_lengths = torch.cat((men_hair_lengths, women_hair_lengths))
    labels = torch.cat((torch.zeros(N), torch.ones(N)))  # 0: man, 1: woman
    dataset = torch.column_stack((heights, weights, hair_lengths, labels))

    random_permutation = torch.randperm(len(dataset))
    train_size = int(len(dataset) * 0.75)
    train_set = dataset[random_permutation[:train_size]]
    validation_set = dataset[random_permutation[train_size:]]
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1:]
    x_val = validation_set[:, :-1]
    y_val = validation_set[:, -1:]

    return x_train, x_val, y_train, y_val


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


def train(model, loss_fn, optimizer, epochs, x_train, x_val, y_train, y_val):
    """Train a MLP to classify gender based on height, weight and hair length."""
    for epoch in range(1, epochs + 1):
        train_out = model(x_train)
        train_loss = loss_fn(train_out, y_train)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            val_out = model(x_val)
            val_loss = loss_fn(val_out, y_val)
            train_accuracy = ((train_out > 0) == y_train).sum() / len(y_train)
            val_accuracy = ((val_out > 0) == y_val).sum() / len(y_val)
            print(
                f"Epoch {epoch:3}",
                f"train loss: {train_loss:.3f} ({100 * train_accuracy:.1f}% accuracy)",
                f"validation loss: {val_loss:.3f} ({100 * val_accuracy:.1f}% accuracy)",
                sep=" | ",
            )


def main():
    """Test day1/ex03."""
    torch.manual_seed(0)
    x_train, x_val, y_train, y_val = generate_dataset()
    model = MLP(3, 8, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(model, loss_fn, optimizer, 200, x_train, x_val, y_train, y_val)
    val_accuracy = ((model(x_val) > 0) == y_val).sum() / len(y_val)
    print(f"Validation accuracy: {val_accuracy:.3f}")


if __name__ == "__main__":
    main()
