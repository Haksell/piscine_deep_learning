import torch


def gradient_at(value: float) -> float:
    """Compute the derivative of 3x² + 2x + 1."""
    tensor = torch.tensor(float(value), requires_grad=True)
    f = 3 * tensor**2 + 2 * tensor + 1
    f.backward()
    assert tensor.grad is not None
    return float(tensor.grad)


def main():
    """Test day0/ex02."""
    value = 3
    actual = gradient_at(value)
    expected = 6 * float(value) + 2
    print(f"f'({value}) computed by autograd: {actual}")
    print(f"f'({value}) analytical (6x+2): {expected}")
    print(f"Match: {actual == expected}")


if __name__ == "__main__":
    main()
