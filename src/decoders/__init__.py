"""
Neural decoders for brain-computer interfaces.

This module contains implementations of various neural decoding algorithms:
- Classic: Kalman Filter, Wiener Filter, LDA, HMM
- ML: SVM, Random Forest, XGBoost, Gaussian Process
- Deep Learning: LSTM, Transformer, TCN, VAE
- Meta-Learner: Adaptive decoder selection and combination
"""

from src.decoders.base import BaseDecoder, OnlineDecoder

# Classic decoders
from src.decoders.classic.hmm import DiscreteHMM, GaussianHMM
from src.decoders.classic.kalman_filter import KalmanFilterDecoder, SteadyStateKalmanFilter
from src.decoders.classic.lda import LDADecoder, ShrinkageLDA
from src.decoders.classic.wiener_filter import (
    CausalWienerFilter,
    NonCausalWienerFilter,
    WienerFilterDecoder,
)

# ML decoders
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

# Deep learning decoders
from src.decoders.deep_learning.lstm import BidirectionalLSTMDecoder, LSTMDecoder

# Meta-Learner
from src.decoders.meta_learner import (
    AdaptiveMetaLearner,
    CombinationStrategy,
    DecoderCombiner,
    DecoderSelector,
    OnlineAdapter,
    SelectionStrategy,
    create_default_meta_learner,
)

__all__ = [
    # Base
    "BaseDecoder",
    "OnlineDecoder",
    # Classic
    "KalmanFilterDecoder",
    "SteadyStateKalmanFilter",
    "WienerFilterDecoder",
    "CausalWienerFilter",
    "NonCausalWienerFilter",
    "LDADecoder",
    "ShrinkageLDA",
    "GaussianHMM",
    "DiscreteHMM",
    # ML
    "SVMDecoder",
    "SVMClassifier",
    "RandomForestDecoder",
    "RandomForestClassifierDecoder",
    "XGBoostDecoder",
    "XGBoostClassifier",
    "XGBOOST_AVAILABLE",
    "GaussianProcessDecoder",
    "SparseGPDecoder",
    "GPClassifier",
    # Deep Learning
    "LSTMDecoder",
    "BidirectionalLSTMDecoder",
    # Meta-Learner
    "AdaptiveMetaLearner",
    "create_default_meta_learner",
    "DecoderSelector",
    "DecoderCombiner",
    "OnlineAdapter",
    "SelectionStrategy",
    "CombinationStrategy",
]
