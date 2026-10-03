import matplotlib.pyplot as plt
import torch


def activate(x) -> dict[str, torch.Tensor]:
    """Compute multiple activations on the given input."""
    return {
        "ReLU": torch.nn.ReLU()(x),
        "Sigmoid": torch.nn.Sigmoid()(x),
        "Tanh": torch.nn.Tanh()(x),
    }


def plot():
    """Draw a plot comparing multiple activations."""
    x = torch.linspace(-3.0, 3.0, 128)
    plt.plot(torch.nn.ReLU()(x), label="ReLU")
    plt.plot(torch.nn.Sigmoid()(x), label="Sigmoid")
    plt.plot(torch.nn.Tanh()(x), label="Tanh")
    plt.legend()
    plt.show()


def main():
    """Test day1/ex01."""
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    activations = activate(x)
    print(f"{'Input':8}: {x}")
    for activation, output in activations.items():
        print(f"{activation:8}: {output}")
    plot()


if __name__ == "__main__":
    main()
