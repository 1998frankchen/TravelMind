#!/bin/bash

# TravelMind Environment Setup Script
# Author: Frank Chen
# Description: Automated environment setup for TravelMind project

set -e  # Exit on error

# Configuration
PROJECT_NAME="TravelMind"
PYTHON_VERSION="3.11"
VENV_NAME=".venv"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions
print_message() {
    echo -e "${2}[${1}]${NC} ${3}"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_python_version() {
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    required_version="3.8"

    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
        print_message "OK" "$GREEN" "Python $python_version detected"
        return 0
    else
        print_message "ERROR" "$RED" "Python 3.8+ required (found $python_version)"
        return 1
    fi
}

check_gpu() {
    if check_command nvidia-smi; then
        print_message "OK" "$GREEN" "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    else
        print_message "WARNING" "$YELLOW" "No NVIDIA GPU detected - CPU mode only"
    fi
}

install_uv() {
    if check_command uv; then
        print_message "OK" "$GREEN" "UV already installed"
        return 0
    fi

    print_message "INFO" "$BLUE" "Installing UV package manager..."

    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        # Unix/Linux/macOS
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    if check_command uv; then
        print_message "OK" "$GREEN" "UV installed successfully"
        return 0
    else
        print_message "ERROR" "$RED" "UV installation failed"
        return 1
    fi
}

setup_environment() {
    local env_type=$1

    if [[ "$env_type" == "uv" ]]; then
        print_message "INFO" "$BLUE" "Setting up UV environment..."

        # Install UV if needed
        install_uv || return 1

        # Sync dependencies
        print_message "INFO" "$BLUE" "Installing dependencies..."
        uv sync

        # Install development dependencies
        if [[ "$2" == "--dev" ]]; then
            uv sync --dev
        fi

        print_message "OK" "$GREEN" "UV environment ready"

    else
        print_message "INFO" "$BLUE" "Setting up virtual environment..."

        # Create virtual environment
        if [[ ! -d "$VENV_NAME" ]]; then
            python3 -m venv "$VENV_NAME"
        fi

        # Activate and install
        source "$VENV_NAME/bin/activate"
        pip install --upgrade pip
        pip install -r requirements.txt

        if [[ "$2" == "--dev" ]]; then
            pip install -r requirements-dev.txt 2>/dev/null || true
        fi

        print_message "OK" "$GREEN" "Virtual environment ready"
    fi
}

verify_installation() {
    print_message "INFO" "$BLUE" "Verifying installation..."

    python3 -c "
import sys
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✓ CUDA available: {torch.cuda.device_count()} GPU(s)')
except ImportError:
    print('✗ PyTorch not installed')
    sys.exit(1)

try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except ImportError:
    print('✗ Transformers not installed')
    sys.exit(1)
" || {
    print_message "WARNING" "$YELLOW" "Some components not installed"
    return 1
}

    print_message "OK" "$GREEN" "Installation verified"
}

# Main execution
main() {
    echo
    print_message "WELCOME" "$GREEN" "$PROJECT_NAME Environment Setup"
    echo

    # Parse arguments
    ENV_TYPE="auto"
    DEV_MODE=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --uv)
                ENV_TYPE="uv"
                shift
                ;;
            --venv)
                ENV_TYPE="venv"
                shift
                ;;
            --dev)
                DEV_MODE="--dev"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --uv     Use UV package manager"
                echo "  --venv   Use traditional virtual environment"
                echo "  --dev    Install development dependencies"
                echo "  --help   Show this help message"
                exit 0
                ;;
            *)
                print_message "ERROR" "$RED" "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Check prerequisites
    print_message "INFO" "$BLUE" "Checking system requirements..."

    check_python_version || exit 1
    check_gpu

    # Setup environment
    if [[ "$ENV_TYPE" == "auto" ]]; then
        if check_command uv; then
            ENV_TYPE="uv"
        else
            print_message "INFO" "$BLUE" "UV not found, trying to install..."
            if install_uv; then
                ENV_TYPE="uv"
            else
                ENV_TYPE="venv"
            fi
        fi
    fi

    setup_environment "$ENV_TYPE" "$DEV_MODE" || exit 1

    # Verify
    verify_installation

    # Success message
    echo
    print_message "SUCCESS" "$GREEN" "Environment setup complete!"
    echo
    echo "To start using TravelMind:"

    if [[ "$ENV_TYPE" == "uv" ]]; then
        echo "  uv run python main.py"
    else
        echo "  source $VENV_NAME/bin/activate"
        echo "  python main.py"
    fi

    echo
}

# Run main function
main "$@"