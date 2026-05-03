# Contributing to VL-JEPA

## Code Style

- **Formatter**: `black` with default settings (88 char line length)
- **Import sorting**: `isort` with black profile
- **Linting**: `ruff` with default rules
- **Type hints**: Required on all public functions
- **Docstrings**: Google style on all public APIs

```bash
# Format
make format

# Lint
make lint

# Type check
make typecheck
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run `make test lint format`
5. Push and open PR against `main`
6. Ensure CI passes
7. Request review

## Testing

```bash
# All tests
make test

# Specific module
pytest tests/test_se3_utils.py -v

# With coverage
pytest --cov=src --cov-report=html tests/
```

All new code requires tests. Tests must:
- Be self-contained (no external dependencies mocked where possible)
- Use descriptive test names
- Cover edge cases (empty tensors, single batch, device transfers)

## Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

## Documentation

- All public functions need docstrings
- Use type hints (Python 3.10+ syntax: `def foo(x: int) -> bool:`)
- Update relevant docs in `docs/` when changing architecture or adding features
