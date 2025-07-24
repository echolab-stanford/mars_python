# MARS Python Library

[![Python package](https://github.com/echolab-stanford/mars_python/actions/workflows/python-package.yml/badge.svg)](https://github.com/echolab-stanford/mars_python/actions/workflows/python-package.yml)

**MARS: A Unifying Framework for Robust and Efficient Inference with Unstructured Data**

This is a Python implementation of the MARS statistical framework, translated from the [original R package](https://github.com/jscarlson/mar-s). MARS provides robust estimators for statistical inference when you have:

- **Ground truth labels** for only a subset of your data (annotation indicators `A`)
- **Machine learning predictions** (`hat_mu`) for the entire dataset  
- **Known annotation probabilities** (`pi`) for the labeling process

The framework combines both human annotations and ML predictions while accounting for measurement error, providing consistent and efficient estimators for various econometric models.

## Installation

```bash
cd mars_python
pip install -e .
```

## Quick Start

```python
import numpy as np
from mars.mars_estimators import mars_mean_estimator
from mars.utils import simulate_mars_data

# Simulate data
data = simulate_mars_data(n=1000, annotation_rate=0.3)

# Estimate population mean
result = mars_mean_estimator(
    M=data['M'],           # Ground truth (only non-zero where annotated)
    A=data['A'],           # Annotation indicators (0/1)
    hat_mu=data['hat_mu'], # ML predictions for all observations
    pi=data['pi']          # Annotation probabilities
)

print(f"Estimate: {result['est']:.3f} ± {result['se']:.3f}")
print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

## Core Features

### MARS Estimators

All estimators use **pseudo-outcomes** of the form:
```
pseudo = hat_mu + (A / pi) * (M - hat_mu)
```

- **`mars_mean_estimator()`**: Population mean estimation
- **`mars_linreg_estimator()`**: Linear regression with measurement error correction
- **`mars_liv_estimator()`**: Linear instrumental variables estimation
- **`mars_did_estimator()`**: Difference-in-differences (Callaway-Sant'Anna style)
- **`mars_lrdd_estimator()`**: Local regression discontinuity design

### Measurement Error Corrections

- **`me_ls_iid()`**: Classical measurement error corrected least squares (iid errors)
- **`me_ls_inid()`**: ME-corrected least squares (heteroskedastic errors)

Both support:
- One-way and two-way cluster-robust standard errors
- Eigenvalue corrections for numerical stability
- Multiple outcome variables

## Mathematical Framework

The MARS framework addresses the challenge of statistical inference with partially annotated data. The key insight is to use **influence functions** to combine ML predictions with human annotations:

1. **Pseudo-outcomes**: `ψ = ĥ + (A/π)(M - ĥ)`
2. **Consistent estimation**: E[ψ] = E[Y] under mild conditions
3. **Asymptotic variance**: Uses influence function approach for robust inference

## Example Applications

### 1. Population Mean Estimation

```python
from mars.mars_estimators import mars_mean_estimator

result = mars_mean_estimator(
    M=ground_truth_values,
    A=annotation_indicators,
    hat_mu=ml_predictions,
    pi=annotation_probabilities
)
```

### 2. Linear Regression

```python
from mars.mars_estimators import mars_linreg_estimator

# Note: C_resid must be residualized (mean-zero)
result = mars_linreg_estimator(
    C_resid=residualized_covariates,
    A=annotation_indicators,
    M=ground_truth_outcomes,
    hat_mu=ml_predictions,
    pi=annotation_probabilities
)
```

### 3. Measurement Error Correction

```python
from mars.me_ls import me_ls_iid

result = me_ls_iid(
    X=design_matrix,
    Y=outcomes,
    Sigma=measurement_error_covariance,
    cluster1=cluster_ids  # Optional clustering
)
```

## Testing

Run the test suite:

```bash
cd mars_python
python -m pytest tests/ -v
```

Run examples:

```bash
python examples/basic_usage.py
```

If you use this library, please cite the original paper:

> Carlson, J., & Dell, M. (2025). A Unifying Framework for Robust and Efficient Inference with Unstructured Data (No. arXiv:2505.00282). arXiv. https://doi.org/10.48550/arXiv.2505.00282

### Things left out from the R code
- **LLM Integration**: The original R package's OpenAI/HuggingFace API wrappers are not included (by design)
- **Specialized Plotting**: Advanced visualization functions from the R package
