#!/bin/bash

# TravelMind Application Launcher
# Author: Frank Chen
# Description: Start TravelMind with various configurations

set -e

# Configuration
PROJECT_NAME="TravelMind"
DEFAULT_FUNCTION="use_rag_web_demo"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Functions
print_message() {
    echo -e "${2}[${1}]${NC} ${3}"
}

check_environment() {
    # Check if UV is available
    if command -v uv >/dev/null 2>&1; then
        return 0  # UV environment
    elif [[ -d ".venv" ]] && [[ -f ".venv/bin/activate" ]]; then
        return 1  # Virtual environment
    else
        return 2  # No environment
    fi
}

show_menu() {
    echo
    print_message "MENU" "$CYAN" "Available Functions:"
    echo
    echo "  1. use_rag           - Basic RAG system"
    echo "  2. use_rag_web_demo  - Web interface (default)"
    echo "  3. rag_dispatcher    - Advanced RAG dispatcher"
    echo "  4. train             - Model training"
    echo "  5. use_agent         - Agent system"
    echo "  6. inference         - Run inference"
    echo "  7. use_city_rag      - City-specific RAG"
    echo
    echo "  0. Exit"
    echo
}

get_function_name() {
    case $1 in
        1) echo "use_rag" ;;
        2) echo "use_rag_web_demo" ;;
        3) echo "rag_dispatcher" ;;
        4) echo "train" ;;
        5) echo "use_agent" ;;
        6) echo "inference" ;;
        7) echo "use_city_rag" ;;
        *) echo "" ;;
    esac
}

run_with_uv() {
    local function=$1
    shift
    print_message "INFO" "$BLUE" "Starting with UV: $function"
    uv run python main.py --function "$function" "$@"
}

run_with_venv() {
    local function=$1
    shift
    print_message "INFO" "$BLUE" "Starting with venv: $function"
    source .venv/bin/activate
    python main.py --function "$function" "$@"
}

run_direct() {
    local function=$1
    shift
    print_message "INFO" "$BLUE" "Starting directly: $function"
    python3 main.py --function "$function" "$@"
}

# Main execution
main() {
    echo
    print_message "WELCOME" "$GREEN" "$PROJECT_NAME Launcher"
    echo

    # Parse arguments
    FUNCTION=""
    INTERACTIVE=true
    EXTRA_ARGS=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --function|-f)
                FUNCTION="$2"
                INTERACTIVE=false
                shift 2
                ;;
            --rag-type)
                EXTRA_ARGS="$EXTRA_ARGS --rag_type $2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  -f, --function NAME  Run specific function"
                echo "  --rag-type TYPE      RAG type for dispatcher"
                echo "  -h, --help           Show this help message"
                echo
                echo "Functions:"
                echo "  use_rag, use_rag_web_demo, rag_dispatcher,"
                echo "  train, use_agent, inference, use_city_rag"
                exit 0
                ;;
            *)
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
                ;;
        esac
    done

    # Check environment
    check_environment
    ENV_STATUS=$?

    if [[ $ENV_STATUS -eq 2 ]]; then
        print_message "ERROR" "$RED" "No environment detected!"
        echo "Please run ./scripts/setup.sh first"
        exit 1
    fi

    # Interactive mode
    if [[ "$INTERACTIVE" == true ]]; then
        show_menu
        read -p "Select function (0-7): " choice

        if [[ "$choice" == "0" ]]; then
            print_message "INFO" "$BLUE" "Exiting..."
            exit 0
        fi

        FUNCTION=$(get_function_name "$choice")

        if [[ -z "$FUNCTION" ]]; then
            print_message "ERROR" "$RED" "Invalid selection"
            exit 1
        fi

        # Additional options for specific functions
        if [[ "$FUNCTION" == "rag_dispatcher" ]]; then
            echo
            echo "RAG Types:"
            echo "  1. self_rag (default)"
            echo "  2. rag"
            echo "  3. mem_walker"
            echo
            read -p "Select RAG type (1-3): " rag_choice

            case $rag_choice in
                1) EXTRA_ARGS="$EXTRA_ARGS --rag_type self_rag" ;;
                2) EXTRA_ARGS="$EXTRA_ARGS --rag_type rag" ;;
                3) EXTRA_ARGS="$EXTRA_ARGS --rag_type mem_walker" ;;
            esac
        fi
    fi

    # Use default if not specified
    if [[ -z "$FUNCTION" ]]; then
        FUNCTION="$DEFAULT_FUNCTION"
    fi

    # Run the application
    echo
    print_message "START" "$GREEN" "Launching $FUNCTION..."
    echo

    case $ENV_STATUS in
        0)  # UV environment
            run_with_uv "$FUNCTION" $EXTRA_ARGS
            ;;
        1)  # Virtual environment
            run_with_venv "$FUNCTION" $EXTRA_ARGS
            ;;
        *)  # Fallback
            run_direct "$FUNCTION" $EXTRA_ARGS
            ;;
    esac
}

# Handle interrupts
trap 'print_message "INTERRUPTED" "$YELLOW" "Application stopped by user"; exit 1' INT TERM

# Run main function
main "$@"