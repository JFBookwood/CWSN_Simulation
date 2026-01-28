# Tests

Testing and validation scripts for the CWSN project.

## Available Tests

- **test_likelihood.py** - Validates the survey-aware void likelihood implementation with Cobaya
  - Tests likelihood initialization
  - Tests parameter evaluation
  - Generates diagnostic logs

### Running Tests

```bash
# Run likelihood test
python tests/test_likelihood.py
```

## Testing Best Practices

- All test scripts should be self-contained
- Use relative imports when possible to maintain portability
- Document test purposes and expected outputs
- Keep test data manageable (use fixtures/mocks for large datasets)
