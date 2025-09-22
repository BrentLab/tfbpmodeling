# Testing Guide

This guide covers testing practices, running tests, and writing new tests for tfbpmodeling.

## Overview

tfbpmodeling uses **pytest** as the testing framework with the following testing practices:

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows
- **Coverage tracking**: Monitor test coverage with codecov
- **Automated testing**: CI/CD pipeline runs tests on all PRs

## Running Tests

### Basic Test Execution

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tfbpmodeling/tests/test_interface.py

# Run specific test method
poetry run pytest tfbpmodeling/tests/test_interface.py::test_linear_perturbation_binding_modeling

# Run tests matching pattern
poetry run pytest -k "test_modeling"
```

### Coverage Testing

```bash
# Run with coverage
poetry run pytest --cov --cov-branch --cov-report=xml

# Generate HTML coverage report
poetry run pytest --cov=tfbpmodeling --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Verbose Testing

```bash
# Show detailed output
poetry run pytest -v

# Show print statements
poetry run pytest -s

# Stop on first failure
poetry run pytest -x

# Run in parallel (if pytest-xdist installed)
poetry run pytest -n auto
```

## Test Structure

### Directory Layout

```
tfbpmodeling/tests/
├── __init__.py
├── test_interface.py                    # Main workflow tests
├── test_modeling_input_data.py          # Data handling tests
├── test_bootstrapped_input_data.py      # Bootstrap tests
├── test_bootstrap_model_results.py      # Results tests
├── test_evaluation_modules.py           # Significance testing
├── test_utils.py                        # Utility function tests
├── fixtures/                            # Test data files
│   ├── sample_expression.csv
│   ├── sample_binding.csv
│   └── sample_blacklist.txt
└── conftest.py                          # Shared fixtures
```

### Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tfbpmodeling/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=tfbpmodeling",
    "--cov-branch",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

## Writing Tests

### Test Class Structure

```python
import pytest
import pandas as pd
from tfbpmodeling.modeling_input_data import ModelingInputData


class TestModelingInputData:
    \"\"\"Test suite for ModelingInputData class.\"\"\"

    def test_basic_initialization(self, sample_data_files):
        \"\"\"Test basic object creation.\"\"\"
        data = ModelingInputData(
            response_file=sample_data_files['response'],
            predictors_file=sample_data_files['predictors'],
            perturbed_tf='YPD1'
        )

        assert data is not None
        assert len(data.get_feature_names()) > 0
        assert data.perturbed_tf == 'YPD1'

    def test_file_validation(self, sample_data_files):
        \"\"\"Test input file validation.\"\"\"
        # Test missing response file
        with pytest.raises(FileNotFoundError):
            ModelingInputData(
                response_file='nonexistent.csv',
                predictors_file=sample_data_files['predictors'],
                perturbed_tf='YPD1'
            )

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalization_options(self, sample_data_files, normalize):
        \"\"\"Test different normalization settings.\"\"\"
        data = ModelingInputData(
            response_file=sample_data_files['response'],
            predictors_file=sample_data_files['predictors'],
            perturbed_tf='YPD1',
            normalize_weights=normalize
        )

        assert data.normalize_weights == normalize

    @pytest.mark.slow
    def test_large_dataset_handling(self, large_sample_data):
        \"\"\"Test performance with large datasets.\"\"\"
        # This test is marked as slow and can be skipped
        data = ModelingInputData(**large_sample_data)
        assert len(data.get_feature_names()) > 1000
```

### Fixtures

Create reusable test data with fixtures:

```python
# conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data_files(tmp_path):
    \"\"\"Create sample CSV files for testing.\"\"\"
    # Generate sample data
    np.random.seed(42)
    genes = [f"gene_{i}" for i in range(100)]
    samples = [f"sample_{i}" for i in range(20)]
    tfs = [f"TF_{i}" for i in range(10)]

    # Response data
    response_data = pd.DataFrame(
        np.random.normal(0, 1, (100, 20)),
        index=genes,
        columns=samples
    )
    response_data['YPD1'] = np.random.normal(-0.5, 0.8, 100)
    response_data.index.name = 'gene_id'

    # Predictor data
    predictor_data = pd.DataFrame(
        np.random.beta(0.5, 2, (100, 10)),
        index=genes,
        columns=tfs
    )
    predictor_data.index.name = 'gene_id'

    # Save files
    response_file = tmp_path / "response.csv"
    predictor_file = tmp_path / "predictors.csv"

    response_data.to_csv(response_file)
    predictor_data.to_csv(predictor_file)

    return {
        'response': str(response_file),
        'predictors': str(predictor_file)
    }


@pytest.fixture
def sample_blacklist_file(tmp_path):
    \"\"\"Create sample blacklist file.\"\"\"
    blacklist_file = tmp_path / "blacklist.txt"
    blacklist_file.write_text("gene_1\\ngene_2\\ngene_3\\n")
    return str(blacklist_file)


@pytest.fixture(scope="session")
def large_sample_data():
    \"\"\"Create large dataset for performance testing.\"\"\"
    # Only create once per test session
    # Implementation for large test data
    pass
```

### Testing Async and Complex Operations

```python
import pytest
from unittest.mock import patch, MagicMock


class TestBootstrapModeling:
    \"\"\"Test bootstrap modeling functionality.\"\"\"

    def test_bootstrap_sampling(self, sample_data_files):
        \"\"\"Test bootstrap sample generation.\"\"\"
        data = BootstrappedModelingInputData(
            base_data=ModelingInputData(**sample_data_files, perturbed_tf='YPD1'),
            n_bootstraps=100,
            random_state=42
        )

        # Test reproducibility
        indices1 = data.get_bootstrap_indices()

        data_copy = BootstrappedModelingInputData(
            base_data=ModelingInputData(**sample_data_files, perturbed_tf='YPD1'),
            n_bootstraps=100,
            random_state=42
        )
        indices2 = data_copy.get_bootstrap_indices()

        assert np.array_equal(indices1, indices2)

    @patch('tfbpmodeling.interface.LassoCV')
    def test_lasso_cv_integration(self, mock_lasso, sample_data_files):
        \"\"\"Test LassoCV integration with mocking.\"\"\"
        # Mock LassoCV behavior
        mock_estimator = MagicMock()
        mock_estimator.fit.return_value = mock_estimator
        mock_estimator.coef_ = np.random.normal(0, 1, 10)
        mock_lasso.return_value = mock_estimator

        # Test the integration
        args = create_test_args(sample_data_files)
        result = linear_perturbation_binding_modeling(args)

        # Verify LassoCV was called
        mock_lasso.assert_called()
        mock_estimator.fit.assert_called()

    def test_error_handling(self, sample_data_files):
        \"\"\"Test error handling in edge cases.\"\"\"
        # Test with insufficient data
        minimal_data = create_minimal_data(n_features=5, n_samples=3)

        with pytest.raises(ValueError, match="Insufficient data"):
            ModelingInputData(**minimal_data, perturbed_tf='YPD1')
```

## Integration Tests

### End-to-End Workflow Tests

```python
class TestCompleteWorkflow:
    \"\"\"Test complete analysis workflow.\"\"\"

    def test_full_pipeline(self, sample_data_files, tmp_path):
        \"\"\"Test complete analysis from start to finish.\"\"\"
        args = argparse.Namespace(
            response_file=sample_data_files['response'],
            predictors_file=sample_data_files['predictors'],
            perturbed_tf='YPD1',
            n_bootstraps=50,  # Reduced for testing
            top_n=30,
            all_data_ci_level=90.0,
            topn_ci_level=80.0,
            max_iter=1000,
            output_dir=str(tmp_path),
            output_suffix='_test',
            n_cpus=1,
            # ... other required args
        )

        # Run complete analysis
        linear_perturbation_binding_modeling(args)

        # Verify output files exist
        output_dirs = list(tmp_path.glob("YPD1_test_*"))
        assert len(output_dirs) == 1

        output_dir = output_dirs[0]
        assert (output_dir / "all_data_results" / "confidence_intervals.csv").exists()
        assert (output_dir / "topn_results" / "confidence_intervals.csv").exists()
        assert (output_dir / "interactor_significance" / "final_selection.csv").exists()

    def test_reproducible_results(self, sample_data_files, tmp_path):
        \"\"\"Test that results are reproducible with fixed seed.\"\"\"
        args = create_test_args(sample_data_files, tmp_path, random_state=42)

        # Run twice with same seed
        linear_perturbation_binding_modeling(args)
        result1_files = list(tmp_path.glob("YPD1_*"))

        args.output_suffix = '_run2'
        linear_perturbation_binding_modeling(args)
        result2_files = list(tmp_path.glob("YPD1_*run2*"))

        # Compare key results
        ci1 = pd.read_csv(result1_files[0] / "all_data_results" / "confidence_intervals.csv")
        ci2 = pd.read_csv(result2_files[0] / "all_data_results" / "confidence_intervals.csv")

        pd.testing.assert_frame_equal(ci1, ci2)
```

## Performance Testing

### Benchmarking

```python
import time
import pytest


class TestPerformance:
    \"\"\"Performance benchmarks for key operations.\"\"\"

    @pytest.mark.slow
    def test_bootstrap_performance(self, large_sample_data):
        \"\"\"Benchmark bootstrap modeling performance.\"\"\"
        start_time = time.time()

        data = BootstrappedModelingInputData(
            base_data=large_sample_data,
            n_bootstraps=1000
        )

        elapsed = time.time() - start_time

        # Performance assertion (adjust thresholds as needed)
        assert elapsed < 60, f"Bootstrap creation took {elapsed:.2f}s, expected < 60s"

    def test_memory_usage(self, sample_data_files):
        \"\"\"Test memory usage during analysis.\"\"\"
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run analysis
        args = create_test_args(sample_data_files, n_bootstraps=1000)
        linear_perturbation_binding_modeling(args)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory assertion (adjust threshold as needed)
        assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.2f}MB"
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests to `main` and `dev` branches
- Pushes to `main` and `dev` branches
- Scheduled runs (daily)

### Test Matrix

Tests run on multiple environments:
- **Python versions**: 3.11, 3.12
- **Operating systems**: Ubuntu, macOS, Windows
- **Dependencies**: Latest and pinned versions

### Coverage Requirements

- **Minimum coverage**: 80%
- **Coverage reporting**: codecov.io
- **Coverage enforcement**: CI fails if coverage drops

## Debugging Tests

### Running Specific Tests

```bash
# Debug specific test with verbose output
poetry run pytest -v -s tfbpmodeling/tests/test_interface.py::test_specific_function

# Run with debugger
poetry run pytest --pdb tfbpmodeling/tests/test_interface.py::test_specific_function

# Run last failed tests
poetry run pytest --lf
```

### Test Data Inspection

```python
def test_debug_data_inspection(sample_data_files):
    \"\"\"Template for debugging test data.\"\"\"
    response_df = pd.read_csv(sample_data_files['response'], index_col=0)
    predictor_df = pd.read_csv(sample_data_files['predictors'], index_col=0)

    print(f"Response shape: {response_df.shape}")
    print(f"Predictor shape: {predictor_df.shape}")
    print(f"Response columns: {response_df.columns.tolist()}")
    print(f"Predictor columns: {predictor_df.columns.tolist()}")

    # Add your debugging code here
    assert False  # Fail test to see output
```

## Best Practices

### Test Organization

1. **One concept per test**: Each test should verify one specific behavior
2. **Clear test names**: Use descriptive names that explain what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent tests**: Tests should not depend on each other

### Test Data

1. **Use fixtures**: Create reusable test data with pytest fixtures
2. **Minimal data**: Use smallest datasets that demonstrate the behavior
3. **Reproducible data**: Use fixed seeds for random data generation
4. **Clean up**: Use temporary directories that are automatically cleaned

### Assertions

1. **Specific assertions**: Use specific assertion methods (`assert_frame_equal` vs `assert`)
2. **Meaningful messages**: Include helpful error messages in assertions
3. **Expected exceptions**: Test error conditions with `pytest.raises`
4. **Floating point comparisons**: Use appropriate tolerance for numeric comparisons

### Mock and Patch

1. **External dependencies**: Mock external API calls, file system operations
2. **Expensive operations**: Mock slow computations during unit tests
3. **Isolation**: Use mocks to isolate the unit being tested
4. **Verification**: Assert that mocked methods were called correctly

## Common Testing Patterns

### Testing File I/O

```python
def test_file_loading(tmp_path):
    \"\"\"Test file loading with temporary files.\"\"\"
    # Create test file
    test_file = tmp_path / "test.csv"
    test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    test_data.to_csv(test_file, index=False)

    # Test loading
    result = load_data_function(str(test_file))

    # Verify
    pd.testing.assert_frame_equal(result, test_data)
```

### Testing Statistical Functions

```python
def test_confidence_interval_calculation():
    \"\"\"Test confidence interval calculation.\"\"\"
    # Known data with expected results
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ci_lower, ci_upper = calculate_confidence_interval(data, confidence=95)

    # Assert approximate equality for floating point
    assert abs(ci_lower - 2.5) < 0.1
    assert abs(ci_upper - 7.5) < 0.1
```

### Testing Error Conditions

```python
def test_invalid_input_handling():
    \"\"\"Test that invalid inputs raise appropriate errors.\"\"\"
    with pytest.raises(ValueError, match="must be positive"):
        some_function(negative_parameter=-1)

    with pytest.raises(FileNotFoundError):
        load_data_function("nonexistent_file.csv")
```

This testing guide provides comprehensive coverage of testing practices in tfbpmodeling. Regular testing ensures code quality, prevents regressions, and facilitates confident refactoring.