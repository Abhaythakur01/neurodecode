"""
Deep learning neural decoders.

Implements neural network-based decoding algorithms including:
- LSTM: Long Short-Term Memory for temporal dynamics
- Transformer: Self-attention for long-range dependencies
- TCN: Temporal Convolutional Network for efficient temporal modeling
- VAE: Variational Autoencoder (TODO)
"""

from src.decoders.deep_learning.lstm import BidirectionalLSTMDecoder, LSTMDecoder
from src.decoders.deep_learning.tcn import TCNDecoder
from src.decoders.deep_learning.transformer import TransformerDecoder

__all__ = [
    "LSTMDecoder",
    "BidirectionalLSTMDecoder",
    "TransformerDecoder",
    "TCNDecoder",
]
