#!/bin/bash

# helpers/setup.sh

# Check if Python 3.11 is installed
PYTHON_COMMAND=$(command -v python3)
if [[ -z "$PYTHON_COMMAND" || ! "$("$PYTHON_COMMAND" --version 2>&1)" =~ "Python 3.11" ]]; then
    echo "Python 3.11 is not installed or not the default python. Please install Python 3.11 or set it as default."
    exit 1
fi

# Check if the virtual environment exists
if [ -d "env" ]; then
    read -p "Virtual environment 'env' already exists. Do you want to recreate it? (yes/no): " yn
    case $yn in
        [Yy]* ) rm -rf env;;
        [Nn]* ) echo "Using existing virtual environment."; break;;
        * ) echo "Please answer yes or no."; exit 1;;
    esac
fi

# Create virtual environment
python3 -m venv env
echo "Virtual environment created."
source env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Validate if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment activation failed."
    exit 1
else
    echo "Virtual environment activated successfully."
fi

# Run make commands
pip install build
python -m build
pip install --upgrade -e .[devel]

# Determine the operating system
OS_TYPE=$(uname -s)
echo "Operating System: $OS_TYPE"

if [[ "$OS_TYPE" == "MINGW"* || "$OS_TYPE" == "CYGWIN"* || "$OS_TYPE" == "MSYS_NT"* ]]; then
    # Windows-specific FAISS setup
    # Check if 'faiss' folder exists
    if [ ! -d "faiss" ]; then
        git clone "https://github.com/facebookresearch/faiss.git" faiss
    fi

    # Navigate to 'faiss' directory
    cd faiss

    # Find the MKL library path dynamically
    MKL_ROOT="/c/Program Files (x86)/Intel/oneAPI/mkl/"
    MKL_VERSION=$(ls "$MKL_ROOT" | grep -E '^[0-9]+\.[0-9]+$' | sort -V | tail -n 1)
    if [ -z "$MKL_VERSION" ]; then
        echo "MKL version not found in $MKL_ROOT."
        exit 1
    fi
    MKL_PATH="$MKL_ROOT/$MKL_VERSION/lib"
    if [ ! -d "$MKL_PATH" ]; then
        echo "MKL library path not found: $MKL_PATH"
        exit 1
    fi
    echo "MKL library path: $MKL_PATH"

    # Ensure SWIG is installed
    if ! command -v swig &> /dev/null; then
        echo "SWIG could not be found. Please install SWIG."
        exit 1
    fi

    # remove build folder from faiss for clean start
    rm -rf build

    # Run cmake to build FAISS
    PYTHON_EXECUTABLE=$(which python)
    echo "Using Python executable: $PYTHON_EXECUTABLE"
    cmake -B build -DFAISS_ENABLE_GPU=OFF -DBLA_VENDOR=Intel10_64_dyn -DBLAS_LIBRARIES="$MKL_PATH/mkl_intel_lp64.lib;$MKL_PATH/mkl_sequential.lib;$MKL_PATH/mkl_core.lib" -DLAPACK_LIBRARIES="$MKL_PATH/mkl_intel_lp64.lib;$MKL_PATH/mkl_sequential.lib;$MKL_PATH/mkl_core.lib" -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" -DMKL_LIBRARIES="$MKL_PATH" .

    # Execute make commands
    cmake --build build --config Release -j
    (cd build/faiss/python && python setup.py install)
else
    # Non-Windows FAISS installation
    pip install faiss-cpu
fi

# End of script
echo "Process has ended."
