# Contributing

## Code Style

### Formatting

```bash
pip install -e ".[dev]"

# Format
black vljepa/ scripts/ tests/
isort vljepa/ scripts/ tests/

# Check (CI runs these)
black --check vljepa/ scripts/ tests/
isort --check-only vljepa/ scripts/ tests/
```

Settings in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 100
```

### Type Hints

All public functions need type hints:

```python
def compute_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

Check with:

```bash
mypy vljepa/ --ignore-missing-imports
```

### Docstrings

Google style:

```python
def calibrate_conformal(scores: np.ndarray, alpha: float = 0.05) -> float:
    """Compute conformal prediction threshold.

    Args:
        scores: Nonconformity scores from calibration set.
        alpha: Miscoverage level (default 0.05 for 95% coverage).

    Returns:
        q_hat: Conformal quantile threshold.
    """
```

## PR Process

### Branch Naming

```
feat/description
fix/description
docs/description
test/description
```

### Commit Messages

Conventional Commits:

```
feat(flow-matching): add SE(3) exponential map layer
fix(conformal): correct quantile for small calibration sets
docs(readme): update benchmark context
```

### Checklist

- [ ] `black` + `isort` pass
- [ ] Type hints on public API
- [ ] Tests for new functionality
- [ ] `pytest tests/` passes
- [ ] Docs updated if API changed

## Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_flow_matching.py -v

# With coverage
pytest tests/ --cov=vljepa --cov-report=html

# Skip GPU tests
pytest tests/ -v -m "not gpu"
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_flow_matching.py
├── test_neural_ode.py
├── test_conformal.py
├── test_backbone.py
├── test_datasets.py
└── test_integration.py
```

### Example Test

```python
import pytest
import torch
from vljepa.models.flow_matching import SE3FlowMatchingHead

class TestSE3FlowMatchingHead:
    def test_output_shape(self):
        head = SE3FlowMatchingHead(dim=512)
        features = torch.randn(4, 512)
        t = torch.rand(4)
        v_trans, v_rot, gripper = head(features, t)
        assert v_trans.shape == (4, 3)
        assert v_rot.shape == (4, 3)
        assert gripper.shape == (4, 1)
```

## Documentation

- Use relative links: `[setup](docs/setup.md)`
- Include exact commands, not pseudocode
- Tables for structured info
- Cross-link between docs

---

Next: [Setup Guide](docs/setup.md)
