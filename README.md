## day creation

- `uv init dayN`
- change 3.14 to 3.12
- remove `main.py`
- `uv add --dev flake8 flake8-docstrings`
- copy `.flake8` from previous day

## push checks

- `import torch` (no alias)
- `import matplotlib.pyplot as plt`
- no library allowed except `torch` and `matplotlib`
- each program must have its main and not be a simple script (???)
- no global variables
- `uv run flake8 ex*`