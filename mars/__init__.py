"""
MARS: A Unifying Framework for Robust and Efficient Inference with Unstructured Data

This package provides Python implementations of the MARS statistical estimators
for robust inference when combining machine learning predictions with partial
human annotations.

Based on "A Unifying Framework for Robust and Efficient Inference with 
Unstructured Data" by Carlson and Dell (2025).
"""

__version__ = "0.1.0"

from .mars_estimators import (
    mars_mean_estimator,
    mars_linreg_estimator, 
    mars_liv_estimator,
    mars_did_estimator,
    mars_lrdd_estimator
)

from .me_ls import (
    me_ls_iid,
    me_ls_inid
)

__all__ = [
    "mars_mean_estimator",
    "mars_linreg_estimator", 
    "mars_liv_estimator",
    "mars_did_estimator",
    "mars_lrdd_estimator",
    "me_ls_iid",
    "me_ls_inid"
]