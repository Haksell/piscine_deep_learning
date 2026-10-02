import torch


def manual_grads(x: float, w: float, b: float) -> tuple[float, float]:
    """Compute the derivative of (wx + b)² with the chain rule.

    (wx + b)² = w²x² + 2wxb + b²
    ∂y/∂w = 2wx² + 2bx = 2x(wx + b)
    ∂y/∂b = 2wx + 2b = 2(wx + b)
    """
    dy_db = 2 * (w * x + b)
    dy_dw = x * dy_db
    return (dy_dw, dy_db)


def autograd_grads(x: float, w: float, b: float) -> tuple[float, float]:
    """Compute the derivative of (wx + b)² with autograd."""
    tx = torch.tensor(x, requires_grad=True)
    tw = torch.tensor(w, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    f = (tw * tx + tb) ** 2
    f.backward()
    assert tw.grad is not None and tb.grad is not None
    return (float(tw.grad), float(tb.grad))


def main():
    """Test day0/ex03."""
    x = 2.0
    w = 3.0
    b = 1.0
    y = (w * x + b) ** 2
    print(f"{x=} {w=} {b=} -> {y=}")
    manual_dy_dw, manual_dy_db = manual_grads(x, w, b)
    autograd_dy_dw, autograd_dy_db = autograd_grads(x, w, b)
    print(f"Manual  : dy/dw={manual_dy_dw} dy/db={manual_dy_db}")
    print(f"Autograd: dy/dw={autograd_dy_dw} dy/db={autograd_dy_db}")
    print(f"Match: {manual_dy_dw == autograd_dy_dw and manual_dy_db == autograd_dy_db}")


if __name__ == "__main__":
    main()
