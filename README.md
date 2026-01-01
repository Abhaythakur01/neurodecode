# NeuroDecode: Real-Time Adaptive Brain-Computer Interface

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An expert-level Brain-Computer Interface system with adaptive neural decoding for motor control and clinical translation framework

## Overview

NeuroDecode is a comprehensive BCI system that combines cutting-edge machine learning algorithms with production-quality engineering to enable real-time neural decoding for motor control. The project demonstrates the full pipeline from raw neural data to deployed application with clinical translation considerations.

### Key Features

- **13+ Decoding Algorithms**: Classic (Kalman, Wiener, LDA, HMM), ML (SVM, RF, XGBoost, GP), and Deep Learning (LSTM, Transformer, TCN, VAE)
- **Novel Adaptive Meta-Learner**: Automatically selects and combines decoders based on brain state
- **Real-Time Performance**: <50ms latency requirement for closed-loop control
- **Production Web Application**: FastAPI backend + React frontend with live visualizations
- **Clinical Translation**: Comprehensive regulatory documentation and clinical protocols
- **Uncertainty Quantification**: Confidence intervals on predictions for safe operation

## Project Structure

```
neural_decoding_sys/
├── src/
│   ├── decoders/          # All decoding algorithms
│   │   ├── classic/       # Kalman, Wiener, LDA, HMM
│   │   ├── ml/            # SVM, Random Forest, XGBoost, Gaussian Process
│   │   ├── deep_learning/ # LSTM, Transformer, TCN, VAE
│   │   └── meta_learner/  # Novel adaptive hybrid decoder
│   ├── preprocessing/     # Signal processing pipeline
│   ├── features/          # Feature extraction
│   ├── backend/           # FastAPI web application
│   ├── evaluation/        # Benchmarking framework
│   └── utils/             # Shared utilities
├── frontend/              # React + TypeScript application
├── docs/
│   ├── clinical/          # Clinical translation documents
│   ├── papers/            # Literature review
│   ├── api/               # API documentation
│   └── architecture/      # System design
├── tests/                 # Unit & integration tests (80%+ coverage)
├── data/                  # Neural datasets
├── models/                # Trained models
└── notebooks/             # Jupyter notebooks for analysis
```

## Three Core Modules

### Module 1: Multi-Algorithm Neural Decoder Suite

Implements and benchmarks 13+ neural decoding algorithms:

**Classic Decoders:**
- Kalman Filter (industry standard)
- Wiener Filter (optimal linear decoder)
- Linear Discriminant Analysis (discrete classification)
- Hidden Markov Models (state-based decoding)

**Machine Learning Decoders:**
- Support Vector Machine (multiple kernels)
- Random Forest ensemble
- Gradient Boosting (XGBoost/LightGBM)
- Gaussian Process (with uncertainty)

**Deep Learning Decoders:**
- LSTM Networks (temporal dynamics)
- Transformer Architecture (state-of-the-art)
- Temporal Convolutional Networks (efficient)
- Variational Autoencoders (latent space)

**Novel Contribution - Adaptive Meta-Learner:**
- Automatically selects best algorithm for current brain state
- Combines decoders using confidence-weighted ensemble
- Adapts online as neural patterns change
- Handles electrode dropout gracefully
- Self-calibrates with minimal user intervention

### Module 2: Real-Time Neural Decoding Application

Production-quality web application with:

- Real-time neural signal processing
- Interactive 2D/3D cursor control
- Live visualization (raster plots, firing rates, trajectories)
- User calibration module (2-5 minute setup)
- Adaptive recalibration (automatic drift correction)
- Clinical dashboard for patient tracking
- Session recording and replay

**Technology Stack:**
- Backend: Python + FastAPI (async)
- Frontend: React + TypeScript
- ML Framework: PyTorch
- Visualization: Plotly, D3.js, Three.js
- Database: PostgreSQL
- Deployment: Docker + CI/CD
- Testing: pytest (>80% coverage)

### Module 3: Clinical Translation & Research Framework

Comprehensive documentation for clinical deployment:

- Clinical protocol for target populations (SCI, ALS, stroke)
- FDA regulatory documentation (SaMD classification)
- Risk analysis (ISO 14971)
- Ethics white paper
- Health economics model (QALY, cost-effectiveness)
- Clinical trial protocol (IRB-ready)

## Performance Metrics

- **Decoding Accuracy**: R² correlation with intended movement
- **Latency**: <50ms end-to-end processing
- **Robustness**: Performance under signal degradation
- **Adaptability**: Learning curve and convergence speed
- **Efficiency**: Computational resources (embedded-ready)
- **Calibration**: Uncertainty quantification accuracy

## Datasets

The project uses multiple publicly available neural datasets:

1. **Neural Latents Benchmark (NLB)** - standardized reaching tasks
2. **CRCNS Reaching Datasets** - motor cortex recordings
3. **DANDI Archive** - diverse neurophysiology data
4. **BCI Competition Datasets** - historical benchmarks

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- PostgreSQL 13+
- Docker (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/neural_decoding_sys.git
cd neural_decoding_sys

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install frontend dependencies
cd frontend
npm install
cd ..

# Set up database
python scripts/setup_database.py

# Download datasets
python scripts/download_datasets.py
```

### Quick Start

```bash
# Run backend
uvicorn src.backend.main:app --reload

# Run frontend (in another terminal)
cd frontend
npm start

# Run tests
pytest tests/ -v --cov=src
```

## Usage

### Training a Decoder

```python
from src.decoders.classic import KalmanFilterDecoder
from src.preprocessing import NeuralDataLoader

# Load data
loader = NeuralDataLoader('data/datasets/nlb/')
X_train, y_train, X_test, y_test = loader.load_reaching_task()

# Train decoder
decoder = KalmanFilterDecoder()
decoder.fit(X_train, y_train)

# Evaluate
predictions = decoder.predict(X_test)
r2_score = decoder.evaluate(X_test, y_test)
print(f"R² Score: {r2_score:.3f}")
```

### Using the Meta-Learner

```python
from src.decoders.meta_learner import AdaptiveMetaLearner

# Initialize with multiple base decoders
meta_learner = AdaptiveMetaLearner(
    base_decoders=['kalman', 'lstm', 'transformer'],
    adaptation_rate=0.01
)

# Online learning
for neural_data, true_movement in real_time_stream():
    prediction, confidence = meta_learner.predict(neural_data)
    meta_learner.update(neural_data, true_movement)  # Adapt online
```

## Development Roadmap

### Month 1: Foundation (Current)
- [x] Project structure setup
- [ ] Literature review (50+ papers)
- [ ] Dataset acquisition and exploration
- [ ] Data preprocessing pipeline
- [ ] Baseline Kalman Filter implementation

### Month 2: Core Algorithms
- [ ] Classic decoders (Wiener, LDA, HMM)
- [ ] ML decoders (SVM, RF, XGBoost, GP)
- [ ] Benchmarking suite

### Month 3: Advanced System
- [ ] Deep learning decoders (LSTM, Transformer, TCN, VAE)
- [ ] Adaptive meta-learner
- [ ] Real-time optimization

### Month 4: Frontend & Integration
- [ ] React application
- [ ] Visualization components
- [ ] Integration testing

### Month 5: Clinical Framework
- [ ] Clinical protocol
- [ ] Regulatory documentation
- [ ] Ethics analysis

### Month 6: Polish & Launch
- [ ] Comprehensive testing
- [ ] Documentation completion
- [ ] Deployment
- [ ] Conference paper submission

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/decoders/

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Documentation

- [API Documentation](docs/api/README.md)
- [Architecture Design](docs/architecture/DESIGN.md)
- [Clinical Protocol](docs/clinical/PROTOCOL.md)
- [Research Papers](docs/papers/LITERATURE.md)

## Contributing

This is currently a solo research project. Contributions, suggestions, and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neurodecode2026,
  title={NeuroDecode: Real-Time Adaptive Brain-Computer Interface},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/neural_decoding_sys}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Neural Latents Benchmark team
- CRCNS and DANDI data providers
- Open source BCI community
- Reference implementations from Glaser Lab, Sussillo Lab

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Project Link**: https://github.com/yourusername/neural_decoding_sys
- **Blog**: https://yourblog.com/neurodecode

---

**Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: January 2026
