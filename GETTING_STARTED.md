# Getting Started with NeuroDecode

## Project Setup Complete!

Your NeuroDecode project structure has been successfully initialized. This document will guide you through the next steps.

## What's Been Set Up

### 1. Project Structure
```
✓ Complete directory hierarchy
✓ Source code modules (decoders, preprocessing, features, backend)
✓ Frontend directory (React application)
✓ Documentation structure
✓ Tests directory with sample tests
✓ Data and models directories
```

### 2. Configuration Files
```
✓ setup.py & pyproject.toml - Python package configuration
✓ requirements.txt - Production dependencies
✓ requirements-dev.txt - Development dependencies
✓ .gitignore - Git ignore patterns
✓ .env.example - Environment variables template
```

### 3. Development Tools
```
✓ Dockerfile - Container configuration
✓ docker-compose.yml - Multi-service orchestration
✓ Makefile - Common development commands
✓ pytest.ini - Test configuration
✓ .pre-commit-config.yaml - Pre-commit hooks
```

### 4. CI/CD
```
✓ GitHub Actions workflow (ci.yml)
✓ Automated testing pipeline
✓ Code quality checks
✓ Docker build automation
```

### 5. Documentation
```
✓ README.md - Project overview
✓ DESIGN.md - Architecture documentation
✓ LITERATURE.md - Research papers and references
```

## Next Steps

### Phase 1: Environment Setup (Week 1, Days 1-2)

#### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

#### 3. Set Up Environment Variables
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
# Minimal required for local development:
# - DATABASE_URL (if using PostgreSQL)
# - REDIS_URL (if using Redis)
```

#### 4. Verify Installation
```bash
# Run sample tests
pytest tests/unit/test_sample.py -v

# Expected: All tests should pass
```

### Phase 2: Literature Review (Week 1-2)

#### 1. Download Key Papers
- Review `docs/papers/LITERATURE.md`
- Start with the fundamental papers (1-7)
- Take notes on implementation details
- Create a `docs/papers/notes/` directory for summaries

#### 2. Study Code Repositories
```bash
# Clone reference implementations (in a separate directory)
git clone https://github.com/KordingLab/Neural_Decoding.git
git clone https://github.com/snel-repo/neural-data-transformers.git
```

#### 3. Create Literature Summary
```bash
# Create notes directory
mkdir -p docs/papers/notes

# Document your findings
# Focus on: algorithms, hyperparameters, evaluation metrics
```

### Phase 3: Dataset Acquisition (Week 2)

#### 1. Neural Latents Benchmark (NLB)
```bash
# Create download script
touch scripts/download_nlb.py

# Follow instructions at: https://neurallatents.github.io/
```

#### 2. CRCNS Data
- Visit: https://crcns.org/
- Download reaching datasets
- Store in `data/raw/crcns/`

#### 3. Data Organization
```
data/
├── raw/
│   ├── nlb/
│   ├── crcns/
│   └── dandi/
└── processed/
    └── [will be generated]
```

### Phase 4: Start Implementation (Week 3-4)

#### 1. Preprocessing Pipeline
```bash
# Create first module
touch src/preprocessing/signal_processing.py
touch src/preprocessing/spike_detection.py
touch tests/unit/test_preprocessing.py
```

#### 2. Feature Extraction
```bash
# Create feature extraction modules
touch src/features/firing_rates.py
touch src/features/spectral_features.py
touch tests/unit/test_features.py
```

#### 3. Baseline Decoder (Kalman Filter)
```bash
# Implement first decoder
touch src/decoders/classic/kalman_filter.py
touch tests/unit/decoders/test_kalman_filter.py
```

## Development Workflow

### Daily Workflow
```bash
# 1. Pull latest changes (if team project)
git pull

# 2. Create feature branch
git checkout -b feature/kalman-filter

# 3. Make changes
# ... code ...

# 4. Run tests
make test-unit

# 5. Check code quality
make lint

# 6. Format code
make format

# 7. Commit changes
git add .
git commit -m "Implement Kalman Filter decoder"

# 8. Push to remote
git push origin feature/kalman-filter
```

### Using Docker (Optional)
```bash
# Build and start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

### Running Jupyter Notebooks
```bash
# Start Jupyter (standalone)
jupyter lab

# Or use Docker service
docker-compose up jupyter

# Access at: http://localhost:8888
```

## Recommended Reading Order

### Week 1
1. Gilja et al. (2012) - Kalman Filter
2. Wu et al. (2006) - Neural Decoding basics
3. Glaser et al. (2020) - ML for Neural Decoding overview

### Week 2
4. Sussillo et al. (2016) - LFADS
5. Ye & Pandarinath (2021) - Transformers
6. Finn et al. (2017) - MAML

## Useful Commands

### Testing
```bash
make test              # Run all tests
make test-unit         # Run unit tests only
make test-coverage     # Generate coverage report
pytest -k "test_name"  # Run specific test
```

### Code Quality
```bash
make lint              # Check code quality
make format            # Format code
pre-commit run --all   # Run all pre-commit hooks
```

### Documentation
```bash
make docs              # Build documentation
```

## Getting Help

### Resources
- **Architecture**: See `docs/architecture/DESIGN.md`
- **Literature**: See `docs/papers/LITERATURE.md`
- **API Docs**: Will be generated in `docs/api/`

### Common Issues

#### Issue: Dependencies won't install
```bash
# Solution: Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Issue: Tests fail with import errors
```bash
# Solution: Install package in editable mode
pip install -e .
```

#### Issue: Docker services won't start
```bash
# Solution: Check if ports are already in use
docker-compose down
# Then try again
docker-compose up
```

## Development Timeline (Month 1)

### Week 1-2: Foundation
- [x] Project structure (DONE)
- [ ] Literature review (50+ papers)
- [ ] Dataset download and exploration
- [ ] Development environment setup
- [ ] Architecture design document

### Week 3-4: Initial Implementation
- [ ] Data preprocessing pipeline
- [ ] Feature extraction
- [ ] Baseline Kalman Filter
- [ ] Evaluation framework
- [ ] Unit tests for all components

## Milestone Checklist

After completing each milestone, update this checklist:

- [x] Project structure created
- [x] Git repository initialized
- [x] Documentation framework set up
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Sample tests pass
- [ ] Literature review complete
- [ ] Datasets downloaded
- [ ] Preprocessing pipeline implemented
- [ ] First decoder (Kalman) working
- [ ] Evaluation metrics implemented

## Next Document to Create

Once you're ready to start coding, create:
- `docs/architecture/API_DESIGN.md` - Detailed API specifications
- `docs/development/CODING_STANDARDS.md` - Coding conventions
- `notebooks/01_data_exploration.ipynb` - Initial data analysis

## Questions to Consider

As you begin implementation:
1. What datasets will you start with?
2. What bin size for neural activity? (20ms default)
3. What train/test split? (80/20 default)
4. What evaluation metrics to prioritize?
5. GPU availability for deep learning?

---

**Ready to begin?** Start with the environment setup and literature review!

**Last Updated**: January 2026
