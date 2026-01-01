# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroDecode is an expert-level Brain-Computer Interface (BCI) system implementing 13+ neural decoding algorithms with a novel adaptive meta-learning approach. The project aims to demonstrate a complete pipeline from raw neural data to deployed clinical application with real-time performance (<50ms latency).

**Core Innovation**: An adaptive meta-learner that automatically selects and combines multiple decoders based on brain state, with online adaptation and uncertainty quantification.

## Development Environment

### Setup (Windows)
```bash
# Virtual environment is required
python -m venv venv
venv\Scripts\activate

# Install in editable mode
pip install -e .

# Run tests to verify
python -m pytest tests/unit/test_sample.py -v
```

**Note**: Python 3.13+ is supported. Some packages (especially neuroscience-specific ones) may need binary wheels. If compilation fails, install pre-built packages first: `pip install numpy scipy --only-binary=:all:`

## Key Commands

### Testing
```bash
# Run all unit tests (fast)
python -m pytest tests/unit -v -m "not slow"

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/unit/decoders/test_kalman_filter.py -v

# Run benchmarks only
python -m pytest tests/benchmarks/ --benchmark-only

# Run integration tests (requires database)
python -m pytest tests/integration -v -m integration
```

### Code Quality
```bash
# Format code (line length: 100)
python -m black src tests
python -m isort src tests

# Lint
python -m flake8 src tests --max-line-length=100 --extend-ignore=E203,W503

# Type check (relaxed for ML code)
python -m mypy src --ignore-missing-imports
```

### Docker (Full Stack)
```bash
# Start all services (backend, frontend, postgres, redis)
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down
```

## Architecture

### Three-Layer Design

1. **Data Processing Layer** (`src/preprocessing/`, `src/features/`)
   - Input: Multi-channel neural signals (spike trains, LFPs)
   - Preprocessing: Filtering, artifact removal, spike detection
   - Feature extraction: Firing rates (20ms bins default), spectral features
   - Output: Normalized feature vectors

2. **Decoding Layer** (`src/decoders/`)
   - **Classic** (`classic/`): Kalman, Wiener, LDA, HMM
   - **ML** (`ml/`): SVM, Random Forest, XGBoost, Gaussian Process
   - **Deep Learning** (`deep_learning/`): LSTM, Transformer, TCN, VAE
   - **Meta-Learner** (`meta_learner/`): Selects/combines decoders, adapts online

3. **Application Layer** (`src/backend/`, `frontend/`)
   - FastAPI backend with WebSocket for real-time streaming
   - React frontend with live visualization
   - PostgreSQL for session data, Redis for caching

### Data Flow (Real-Time)
```
Neural Data → Preprocessing (10ms) → Feature Extraction (5ms) →
Parallel Decoder Inference (20ms) → Meta-Learner (10ms) →
WebSocket → Frontend (5ms)
Total: <50ms end-to-end
```

### Critical Design Constraints

- **Latency**: All processing must complete in <50ms for closed-loop control
- **Test Coverage**: Target 80%+ coverage (currently configured in pytest.ini)
- **Neural Data Format**: Expects shape (n_samples, n_neurons, n_timebins)
- **Evaluation Metric**: Primary metric is R² correlation with intended movement

## Decoder Implementation Pattern

All decoders must follow this interface:

```python
class BaseDecoder:
    def fit(self, X, y):
        """Train on neural data X and kinematic targets y"""

    def predict(self, X):
        """Decode movement from neural data"""

    def evaluate(self, X, y):
        """Return R² score"""

    def update(self, X, y):
        """Online learning (optional)"""
```

**Key Convention**: Decoders are instantiated with hyperparameters, trained with `fit()`, and used with `predict()`. The meta-learner calls `predict()` on all base decoders in parallel.

## Testing Philosophy

- **Unit tests** (`tests/unit/`): Test individual components in isolation
- **Integration tests** (`tests/integration/`): Test end-to-end pipelines
- **Benchmarks** (`tests/benchmarks/`): Measure latency and throughput
- **Fixtures** (`tests/conftest.py`): Shared test data generators
  - `sample_neural_data`: (100, 50, 20) shaped array
  - `sample_firing_rates`: (100, 50) firing rates
  - `decoder_config`: Standard hyperparameters

### Test Markers
```python
@pytest.mark.unit          # Fast unit tests
@pytest.mark.slow          # Tests >1 second
@pytest.mark.integration   # Requires external services
@pytest.mark.benchmark     # Performance tests
```

## Datasets

The project uses publicly available neural datasets:
- **Neural Latents Benchmark (NLB)**: Primary dataset for reaching tasks
- **CRCNS**: Motor cortex recordings during arm movements
- **DANDI Archive**: Diverse neurophysiology data
- **BCI Competition**: Historical benchmark datasets

**Data Location**: Store raw data in `data/raw/`, processed in `data/processed/`
**Not in Git**: All data files are gitignored (large binary files)

## Performance Optimization

Critical paths for <50ms latency:
1. Use **Cython** for preprocessing bottlenecks (future)
2. **Batch inference** where possible in deep learning models
3. **Model quantization** for PyTorch models
4. **Redis caching** for frequently accessed features
5. **Parallel execution** of base decoders in meta-learner

## Code Style

- **Line length**: 100 characters (black, flake8)
- **Import sorting**: isort with black profile
- **Type hints**: Encouraged but not strictly enforced (mypy warnings acceptable)
- **Docstrings**: Not required for private methods, required for public API
- **Disabled pylint checks**: C0103 (naming), R0903 (few public methods), R0913 (many arguments) - common in ML code

## Common Development Patterns

### Adding a New Decoder

1. Create file in appropriate subdirectory: `src/decoders/{classic,ml,deep_learning}/new_decoder.py`
2. Inherit from base class or implement standard interface
3. Add unit tests: `tests/unit/decoders/test_new_decoder.py`
4. Add to meta-learner registry (if applicable)
5. Benchmark against baseline: Add to `tests/benchmarks/`

### Neural Data Processing

- **Default bin size**: 20ms (1000 samples/sec → 50 bins/second)
- **Feature extraction**: Compute firing rates by counting spikes per bin
- **Normalization**: Z-score normalization per neuron across training set
- **Train/test split**: 80/20, preserve temporal order (no shuffle)

### Real-Time Streaming

Backend uses WebSocket for low-latency data streaming:
```python
# In src/backend/
@app.websocket("/ws/decode")
async def websocket_decode(websocket: WebSocket):
    # Receive neural data, decode, stream predictions
```

Frontend connects via WebSocket to display live trajectories.

## Documentation

- **Architecture**: `docs/architecture/DESIGN.md` - detailed system design
- **Literature**: `docs/papers/LITERATURE.md` - 25+ research papers with reading schedule
- **API Docs**: Auto-generated from FastAPI (not yet implemented)
- **Clinical**: `docs/clinical/` - regulatory and clinical protocol docs (future)

## Development Roadmap Context

**Current Phase**: Month 1 - Foundation
- Project structure: ✅ Complete
- Literature review: In progress
- Dataset acquisition: Pending
- Baseline Kalman Filter: Next milestone

**Critical Path**:
1. Implement preprocessing pipeline
2. Implement Kalman Filter (baseline decoder)
3. Add evaluation metrics (R², latency)
4. Implement meta-learner framework
5. Add deep learning decoders
6. Build real-time web interface

## Known Issues & Limitations

- Pre-commit hooks fail on Windows (requires Visual C++ Build Tools for ruamel.yaml compilation)
  - Workaround: Use `git commit --no-verify` or run formatters manually
- Some neuroscience packages (MNE, Neo, Elephant) not yet installed/tested
- PyTorch CPU-only version installed (GPU support requires CUDA toolkit)
- Frontend directory exists but no React code yet implemented

## When Adding Dependencies

New ML/scientific packages should be added to `requirements.txt`. Development tools go in `requirements-dev.txt`. After adding:
```bash
pip install -r requirements.txt
pip install -e .
```

Check compatibility with Python 3.8+ (declared minimum version).

## Meta-Learner Architecture (Novel Component)

The adaptive meta-learner is the core innovation:

**Three Components**:
1. **Selector**: Chooses best decoder based on uncertainty and recent performance
2. **Combiner**: Confidence-weighted ensemble of top-K decoders
3. **Online Adapter**: Updates weights using recent prediction errors

**Key Feature**: Handles electrode dropout by detecting performance degradation and reweighting decoders.

Implementation location: `src/decoders/meta_learner/`

## Contact & Resources

- Repository: https://github.com/Abhaythakur01/neurodecode
- Issues: https://github.com/Abhaythakur01/neurodecode/issues
- Author: Abhay Thakur (@Abhaythakur01)
