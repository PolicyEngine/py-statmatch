# Contributing

Thank you for your interest in contributing to py-statmatch!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/PolicyEngine/py-statmatch.git
   cd py-statmatch
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

## Code Style

- Line length: 79 characters
- Formatter: Black
- Import sorter: isort

Run formatting:
```bash
black . -l 79
isort .
```

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=statmatch

# Specific test file
pytest tests/test_nnd_hotdeck.py
```

### R Comparison Tests

To run tests that compare against R's StatMatch:

```bash
# Prerequisites: R, rpy2, StatMatch R package
pytest tests/ -k "against_r" -v
```

## Pull Request Process

1. Create a branch for your changes
2. Write tests for new functionality
3. Ensure all tests pass
4. Add a changelog entry in `changelog_entry.yaml`
5. Submit a PR

## Changelog Entry

PRs require a `changelog_entry.yaml` file:

```yaml
- bump: minor  # or patch, major
  changes:
    added:
      - Description of what was added
    fixed:
      - Description of what was fixed
```

## Questions?

Open an issue on GitHub or reach out to the maintainers.
