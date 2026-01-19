"""
Classic neural decoders.

Implements traditional decoding algorithms including:
- Kalman Filter
- Wiener Filter
- LDA
- HMM
"""

from src.decoders.classic.hmm import DiscreteHMM, GaussianHMM
from src.decoders.classic.kalman_filter import KalmanFilterDecoder, SteadyStateKalmanFilter
from src.decoders.classic.lda import LDADecoder, ShrinkageLDA
from src.decoders.classic.wiener_filter import (
    CausalWienerFilter,
    NonCausalWienerFilter,
    WienerFilterDecoder,
)

__all__ = [
    "KalmanFilterDecoder",
    "SteadyStateKalmanFilter",
    "WienerFilterDecoder",
    "CausalWienerFilter",
    "NonCausalWienerFilter",
    "LDADecoder",
    "ShrinkageLDA",
    "GaussianHMM",
    "DiscreteHMM",
]
