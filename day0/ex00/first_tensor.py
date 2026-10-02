import torch


def describe(x: torch.Tensor):
    """Describe the input tensor."""
    print(f"Values:\n{x}")
    print(f"Shape: {x.shape}")
    print(f"Dimensions: {x.ndim}")
    print(f"Dtype: {x.dtype}")
    print(f"Device: {x.device}")
