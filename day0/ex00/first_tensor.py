import torch


def describe(x: torch.Tensor):
    """Describe the input tensor."""
    print(f"Values:\n{x}")
    print(f"Shape: {x.shape}")
    print(f"Dimensions: {x.ndim}")
    print(f"Dtype: {x.dtype}")
    print(f"Device: {x.device}")


def main():
    """Test day0/ex00."""
    describe(torch.tensor([[1, 2, 3], [4, 5, 6]]))


if __name__ == "__main__":
    main()
