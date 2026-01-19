"""
Adaptive Meta-Learner for Neural Decoding.

The core innovation of the NeuroDecode system. Automatically selects
and combines multiple decoders based on performance, uncertainty,
and brain state with online adaptation.

Components:
- Selector: Chooses best decoder(s) based on metrics
- Combiner: Weighted ensemble of predictions
- Adapter: Online weight updates and degradation detection
- MetaLearner: Orchestrates the full system
"""

from src.decoders.meta_learner.adapter import OnlineAdapter
from src.decoders.meta_learner.base import (
    CombinationStrategy,
    DecoderMetrics,
    DecoderState,
    DecoderWrapper,
    EnsembleResult,
    PredictionResult,
    SelectionStrategy,
)
from src.decoders.meta_learner.combiner import DecoderCombiner
from src.decoders.meta_learner.meta_learner import (
    AdaptiveMetaLearner,
    create_default_meta_learner,
)
from src.decoders.meta_learner.selector import DecoderSelector

__all__ = [
    # Main class
    "AdaptiveMetaLearner",
    "create_default_meta_learner",
    # Components
    "DecoderSelector",
    "DecoderCombiner",
    "OnlineAdapter",
    # Data structures
    "DecoderWrapper",
    "DecoderMetrics",
    "DecoderState",
    "PredictionResult",
    "EnsembleResult",
    # Strategies
    "SelectionStrategy",
    "CombinationStrategy",
]
