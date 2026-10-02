import torch


def operations(a: torch.Tensor, b: torch.Tensor):
    """Execute multiple operations on the input tensors."""
    assert a.ndim == 2 and b.ndim == 2, "a and b should be matrices"
    shapes_str = f"{a.shape[0]}x{a.shape[1]} and {b.shape[0]}x{b.shape[1]}"
    assert a.shape == b.shape, (
        f"a and b are not compatible for element-wise addition ({shapes_str})"
    )
    assert a.shape[1] == b.shape[0], (
        f"a and b are not compatible for matrix multiplication ({shapes_str})"
    )
    print(f"Sum:\n{a + b}")
    print(f"Matmul:\n{a @ b}")
    print(f"Mean of a: {a.mean(dtype=torch.float)}")
    print(f"Reshaped a: {a.reshape(1, -1)}")


def main():
    """Test day0/ex01."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    operations(a, b)


if __name__ == "__main__":
    main()
