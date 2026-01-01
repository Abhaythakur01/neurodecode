"""
NeuroDecode: Real-Time Adaptive Brain-Computer Interface
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith('#') and not line.startswith('-r')]

setup(
    name="neurodecode",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-Time Adaptive Brain-Computer Interface for Motor Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural_decoding_sys",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/neural_decoding_sys/issues",
        "Documentation": "https://neural-decode.readthedocs.io",
        "Source Code": "https://github.com/yourusername/neural_decoding_sys",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.25.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurodecode-train=src.cli.train:main",
            "neurodecode-eval=src.cli.evaluate:main",
            "neurodecode-serve=src.cli.serve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "brain-computer-interface",
        "bci",
        "neural-decoding",
        "machine-learning",
        "deep-learning",
        "neuroscience",
        "motor-control",
        "adaptive-learning",
        "meta-learning",
        "real-time-systems",
    ],
)
