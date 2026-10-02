import torch

W = torch.tensor([[0.1, 0.2, 0.3]])
B = torch.tensor([0.0])


def neuron(x: torch.Tensor) -> torch.Tensor:
    """Compute the output of a neuron with predefined weights and bias."""
    linear = torch.nn.Linear(len(x), 1)
    # https://discuss.pytorch.org/t/initalizing-weights-and-biases-to-a-specific-vector-in-python/162979/2
    with torch.no_grad():
        linear.weight.copy_(W)
        linear.bias.copy_(B)

    x = linear(x)
    x = torch.nn.Sigmoid()(x)
    return x.detach().clone()


def main():
    """Test day0/ex04."""
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"Input: {x}")
    print(f"Weights: {W}")
    print(f"Bias: {B}")
    print(f"Output: {neuron(x)}")


if __name__ == "__main__":
    main()
