# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroDecode is a Brain-Computer Interface (BCI) system implementing 17+ neural decoding algorithms with an adaptive meta-learning approach. Core innovation: a meta-learner that automatically selects and combines decoders based on brain state with online adaptation and uncertainty quantification.

**Critical constraint**: All processing must complete in <50ms for closed-loop control.

### Implemented Decoders

| Category | Decoders |
|----------|----------|
| Classic | KalmanFilterDecoder, SteadyStateKalmanFilter, WienerFilterDecoder, CausalWienerFilter, NonCausalWienerFilter, LDADecoder, ShrinkageLDA, GaussianHMM, DiscreteHMM |
| ML | SVMDecoder, SVMClassifier, RandomForestDecoder, RandomForestClassifierDecoder, XGBoostDecoder*, XGBoostClassifier*, GaussianProcessDecoder, SparseGPDecoder, GPClassifier |
| Deep Learning | LSTMDecoder*, BidirectionalLSTMDecoder* |
| Meta-Learner | AdaptiveMetaLearner, DecoderSelector, DecoderCombiner, OnlineAdapter |

*Optional dependencies: XGBoost decoders require `xgboost`, LSTM decoders require `torch`

## Development Environment

```bash
# Setup (Windows)
python -m venv venv
venv\Scripts\activate
pip install -e .

# Verify installation
python -m pytest tests/unit/test_sample.py -v
```

**Note**: Python 3.8+ required. If compilation fails for scientific packages: `pip install numpy scipy --only-binary=:all:`

## Key Commands

### Testing
```bash
# Fast unit tests (excludes slow tests)
python -m pytest tests/unit -v -m "not slow"

# Specific test file
python -m pytest tests/unit/decoders/test_kalman_filter.py -v

# With coverage
python -m pytest --cov=src --cov-report=html

# Integration tests (requires postgres/redis)
python -m pytest tests/integration -v -m integration

# Benchmarks
python -m pytest tests/benchmarks/ --benchmark-only
```

### Code Quality
```bash
python -m black src tests                    # Format (line length: 100)
python -m isort src tests                    # Sort imports
python -m flake8 src tests --max-line-length=100 --extend-ignore=E203,W503
python -m mypy src --ignore-missing-imports  # Type check (warnings OK)
```

### Docker
```bash
docker-compose up -d      # Start all services (backend, frontend, postgres, redis)
docker-compose logs -f backend
docker-compose down
```

## Architecture

### Three-Layer Design

1. **Data Processing** (`src/preprocessing/`, `src/features/`): Neural signal filtering, artifact removal, feature extraction (20ms bins default)

2. **Decoding** (`src/decoders/`):
   - `classic/`: Kalman, Wiener, LDA, HMM
   - `ml/`: SVM, Random Forest, XGBoost, Gaussian Process
   - `deep_learning/`: LSTM, Transformer, TCN, VAE
   - `meta_learner/`: Adaptive decoder selection and combination

3. **Application** (`src/backend/`, `frontend/`): FastAPI + WebSocket backend, React frontend, PostgreSQL + Redis

### Data Flow
```
Neural Data → Preprocessing (10ms) → Features (5ms) → Decoders (20ms) → Meta-Learner (10ms) → WebSocket → Frontend (5ms)
```

### Key Constraints
- **Neural data shape**: (n_samples, n_neurons, n_timebins)
- **Primary metric**: R² correlation with intended movement
- **Test coverage target**: 80%+

## Decoder Interface

All decoders inherit from `BaseDecoder` or `OnlineDecoder` in `src/decoders/base.py`:

```python
class BaseDecoder:
    def fit(self, X, y):       # Train on neural data X, kinematics y
    def predict(self, X):      # Decode movement
    def evaluate(self, X, y):  # Return R² score
    def save(self, path):      # Serialize model
    def load(self, path):      # Deserialize model

class OnlineDecoder(BaseDecoder):
    def update(self, X, y):    # Online learning update
    def predict_single(self, x):  # Single-sample prediction for real-time
```

The meta-learner calls `predict()` on all base decoders in parallel.

### Key Data Structures (meta_learner/base.py)

- `DecoderWrapper`: Wraps a decoder with state, weight, and metrics
- `DecoderMetrics`: Tracks R², MSE, latency, uncertainty over time
- `PredictionResult`: Single decoder output with uncertainty
- `EnsembleResult`: Combined prediction with decoder weights and metadata
- `DecoderState`: ACTIVE, STANDBY, DEGRADED, DISABLED

## Test Fixtures (tests/conftest.py)

- `sample_neural_data`: Returns (X, y) tuple - X:(100, 50, 20), y:(100, 2)
- `sample_firing_rates`: (100, 50) array
- `sample_spike_train`: Sorted spike times in 1s window
- `decoder_config`: Standard hyperparameters dict
- `temp_model_path`: Temporary directory for model saves

### Test Markers
```python
@pytest.mark.unit          # Fast unit tests
@pytest.mark.slow          # Tests >1 second
@pytest.mark.integration   # Requires external services
@pytest.mark.benchmark     # Performance tests
@pytest.mark.decoder       # Decoder algorithm tests
@pytest.mark.preprocessing # Preprocessing module tests
@pytest.mark.api           # API endpoint tests
```

## Adding a New Decoder

1. Create: `src/decoders/{classic,ml,deep_learning}/new_decoder.py`
2. Implement the decoder interface (fit, predict, evaluate, update)
3. Add tests: `tests/unit/decoders/test_new_decoder.py`
4. Register with meta-learner if applicable
5. Add benchmarks: `tests/benchmarks/`

## Neural Data Conventions

- **Bin size**: 20ms (1000 samples/sec → 50 bins/second)
- **Normalization**: Z-score per neuron across training set
- **Train/test split**: 80/20, preserve temporal order (no shuffle)
- **Data location**: `data/raw/` (raw), `data/processed/` (processed) - gitignored

## Meta-Learner Architecture (Fully Implemented)

The adaptive meta-learner is in `src/decoders/meta_learner/` with four components:

### Components

1. **Selector** (`selector.py`): Chooses decoders using configurable strategies:
   - `BEST`: Single best performer
   - `TOP_K`: Top K decoders by R² score
   - `THRESHOLD`: All above performance threshold
   - `UNCERTAINTY_AWARE`: Prefer low-uncertainty decoders
   - `ADAPTIVE`: Dynamic selection based on recent trends

2. **Combiner** (`combiner.py`): Combines predictions using strategies:
   - `MEAN`: Simple average
   - `WEIGHTED_MEAN`: Performance-weighted average
   - `MEDIAN`: Robust to outliers
   - `UNCERTAINTY_WEIGHTED`: Weight by inverse uncertainty (default)
   - `STACKING`: Learned meta-model weights

3. **Online Adapter** (`adapter.py`): Updates weights using recent prediction errors, detects performance degradation, handles electrode dropout

4. **AdaptiveMetaLearner** (`meta_learner.py`): Orchestrates all components with parallel decoder execution via ThreadPoolExecutor

### Usage Example
```python
from src.decoders import AdaptiveMetaLearner, KalmanFilterDecoder, SVMDecoder

meta = AdaptiveMetaLearner(
    selection_strategy=SelectionStrategy.ADAPTIVE,
    combination_strategy=CombinationStrategy.UNCERTAINTY_WEIGHTED,
    top_k=3,
    parallel=True,
)
meta.add_decoder(KalmanFilterDecoder())
meta.add_decoder(SVMDecoder())
meta.fit(X_train, y_train)
result = meta.predict_with_info(X_test)  # Returns EnsembleResult with prediction + uncertainty
```

## Code Style

- Line length: 100 (black, flake8)
- Import sorting: isort with black profile
- Type hints: Encouraged but mypy warnings acceptable
- Docstrings: Required for public API only
- Disabled pylint: C0103 (naming), R0903 (few methods), R0913 (many args)

## Known Issues

- Pre-commit hooks fail on Windows - use `git commit --no-verify` or run formatters manually
- PyTorch is CPU-only (GPU requires CUDA toolkit installation)
- Some neuroscience packages (MNE, Neo, Elephant) may need manual setup
- LSTM tests are skipped if PyTorch is not installed
- XGBoost decoders gracefully degrade if xgboost package not installed

## Dependencies

- `requirements.txt`: All packages (ML, scientific, web framework)
- `requirements-dev.txt`: Development tools (includes requirements.txt)

### Optional Dependencies
```bash
pip install torch      # For LSTM/deep learning decoders
pip install xgboost    # For XGBoost decoders
```

## Test Coverage

Test files in `tests/unit/decoders/`:
- `test_kalman_filter.py` - Kalman filter decoders
- `test_wiener_filter.py` - Wiener filter decoders
- `test_lda.py` - LDA decoders
- `test_hmm.py` - HMM decoders (GaussianHMM, DiscreteHMM)
- `test_svm.py` - SVM regression/classification
- `test_random_forest.py` - Random Forest decoders
- `test_gaussian_process.py` - GP decoders with uncertainty
- `test_lstm.py` - LSTM decoders (requires PyTorch)
- `test_meta_learner.py` - Full meta-learner system tests

Run specific decoder tests:
```bash
python -m pytest tests/unit/decoders/test_meta_learner.py -v
```
