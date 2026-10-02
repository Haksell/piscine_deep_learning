## day creation

- `uv init dayN`
- remove readme
- remove useless fields from `pyproject.toml`
- change 3.14 to 3.12
- remove `main.py`
- `uv add torch matplotlib numpy`
- `uv add --dev flake8 flake8-docstrings`
- copy `.flake8` from previous day

## push checks

- `import torch` (no alias)
- `import matplotlib.pyplot as plt`
- no library allowed except `torch` and `matplotlib`
- each program must have its main and not be a simple script (???)
- no global variables
- `uv run flake8 ex*`

## feedback

- create `tester.py` files or have the tests in the main?
  - ex00 runs tester.py
  - ex01 runs tester.py
  - ex02 runs autograd.py
- day0/ex01: there should be an `AssertionError` if the shapes are not compatible for the matrix sum too
