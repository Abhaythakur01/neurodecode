"""
Machine Learning decoders for neural signals.

Provides SVM, Random Forest, XGBoost, and Gaussian Process implementations
for neural decoding with built-in feature importance and uncertainty estimation.
"""

from src.decoders.ml.gaussian_process import (
    GaussianProcessDecoder,
    GPClassifier,
    SparseGPDecoder,
)
from src.decoders.ml.random_forest import (
    RandomForestClassifierDecoder,
    RandomForestDecoder,
)
from src.decoders.ml.svm import SVMClassifier, SVMDecoder

# XGBoost is optional
try:
    from src.decoders.ml.xgboost_decoder import XGBoostClassifier, XGBoostDecoder

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBoostDecoder = None
    XGBoostClassifier = None

__all__ = [
    # SVM
    "SVMDecoder",
    "SVMClassifier",
    # Random Forest
    "RandomForestDecoder",
    "RandomForestClassifierDecoder",
    # XGBoost (optional)
    "XGBoostDecoder",
    "XGBoostClassifier",
    "XGBOOST_AVAILABLE",
    # Gaussian Process
    "GaussianProcessDecoder",
    "SparseGPDecoder",
    "GPClassifier",
]
