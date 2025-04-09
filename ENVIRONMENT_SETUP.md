# TravelMind Development Environment Setup

**Project:** TravelMind - Transforming Travel Through Intelligent Reasoning
**Author:** Frank Chen
**Repository:** <https://github.com/1998frankchen/TravelMind>
**Version:** 1.0.0

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [UV Environment Setup](#uv-environment-setup)
4. [Alternative Setup Methods](#alternative-setup-methods)
5. [Configuration Files](#configuration-files)
6. [Troubleshooting](#troubleshooting)
7. [Development Tools](#development-tools)

---

## System Requirements

### Minimum Requirements

- **Operating System:** Ubuntu 20.04+ / macOS 10.15+ / Windows 10+
- **Python:** 3.8 or higher
- **RAM:** 16GB minimum
- **Storage:** 50GB available space
- **Network:** Stable internet connection for model downloads

### Recommended Requirements

- **Operating System:** Ubuntu 22.04 LTS
- **Python:** 3.11
- **RAM:** 32GB or more
- **GPU:** NVIDIA RTX 3090 or equivalent (24GB VRAM)
- **Storage:** 100GB+ SSD storage
- **CUDA:** 12.4 (for GPU acceleration)

---

## Quick Start

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/1998frankchen/TravelMind.git
cd TravelMind

# Run the automated setup script
./scripts/setup.sh

# Start the application
./scripts/start.sh
```

The automated setup script will:

1. Detect your system configuration
2. Install UV package manager (if not present)
3. Create and configure the virtual environment
4. Install all dependencies
5. Verify the installation

---

## UV Environment Setup

UV is a modern, fast Python package manager that provides reproducible builds and efficient dependency resolution.

### Installing UV

```bash
# Unix/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### Project Configuration

The project uses `pyproject.toml` for dependency management:

```toml
[project]
name = "travelmind"
version = "1.0.0"
requires-python = ">=3.8"

[tool.uv]
dev-dependencies = [
    "pytest>=6.0",
    "black",
    "isort",
    "mypy",
]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

### Installing Dependencies

```bash
# Sync all dependencies
uv sync

# Install with development dependencies
uv sync --dev

# Install with specific extras
uv sync --extra gpu --extra web
```

### Creating Lock File

```bash
# Generate lock file for reproducible builds
uv lock

# Update dependencies
uv lock --upgrade
```

---

## Alternative Setup Methods

### Traditional Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Conda Environment

```bash
# Create conda environment
conda create -n travelmind python=3.11

# Activate environment
conda activate travelmind

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Docker Container

```bash
# Build Docker image
docker build -t travelmind:latest .

# Run container with GPU support
docker run --gpus all -it travelmind:latest

# Mount local directory
docker run --gpus all -v $(pwd):/workspace -it travelmind:latest
```

---

## Configuration Files

### pyproject.toml Structure

```toml
[project]
name = "travelmind"
version = "1.0.0"
description = "AI-powered travel planning system with RLHF"
authors = [{name = "Frank Chen", email = "frank@example.com"}]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"

dependencies = [
    "torch==2.5.1",
    "transformers==4.49.0",
    "peft==0.14.0",
    "datasets==3.2.0",
    "accelerate==1.2.1",
    "deepspeed==0.16.2",
    "gradio==5.23.3",
    "langchain==0.3.23",
    "chromadb==0.6.3",
    # ... additional dependencies
]

[project.optional-dependencies]
gpu = [
    "bitsandbytes==0.45.0",
    "flash-attn==2.5.0",
]
web = [
    "gradio",
    "streamlit",
    "fastapi",
    "uvicorn",
]
dev = [
    "pytest",
    "black",
    "isort",
    "mypy",
    "flake8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
python-version = "3.11"
python-downloads = true

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

### .python-version

```
3.11
```

### Environment Variables

Create a `.env` file for environment-specific configurations:

```bash
# API Keys (Optional)
OPENAI_API_KEY=your_openai_api_key
ZHIPU_API_KEY=your_zhipu_api_key

# Model Paths
MODEL_BASE_PATH=/path/to/models
DATA_PATH=/path/to/data

# Training Configuration
CUDA_VISIBLE_DEVICES=0,1
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=travelmind

# Application Settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

---

## Troubleshooting

### Common Issues and Solutions

#### UV Installation Failed

```bash
# Manual installation alternative
pip install uv

# Or use pipx
pipx install uv
```

#### CUDA Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
uv remove torch torchvision torchaudio
uv add torch torchvision torchaudio --index pytorch-cuda
```

#### Memory Issues

```bash
# Enable memory-efficient training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use gradient checkpointing
python main.py --gradient_checkpointing --batch_size 1
```

#### Dependency Conflicts

```bash
# Clear UV cache
uv cache clean

# Reinstall with fresh lock file
rm uv.lock
uv lock
uv sync --reinstall
```

#### Permission Errors

```bash
# Fix script permissions
chmod +x scripts/*.sh

# Fix Python permissions
find . -name "*.py" -exec chmod 644 {} \;
```

---

## Development Tools

### Code Quality Tools

```bash
# Format code with Black
black src/

# Sort imports with isort
isort src/

# Type checking with mypy
mypy src/

# Lint with flake8
flake8 src/

# Run all checks
make lint
```

### Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v tests/
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
make test

# Format and lint
make format
make lint

# Commit changes
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/new-feature
```

---

## Additional Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Project Documentation](./docs/)

---

**Note:** This environment setup guide is maintained as part of the TravelMind project. For issues or contributions, please visit the [GitHub repository](https://github.com/1998frankchen/TravelMind).
