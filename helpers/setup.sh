#!/bin/bash

# helpers/setup.sh

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

# Activate virtual environment based on the operating system
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == "MINGW"* || "$OS_TYPE" == "CYGWIN"* || "$OS_TYPE" == "MSYS_NT"* ]]; then
    # Windows activation
    source env/Scripts/activate
else
    # Linux or MacOS activation
    source env/bin/activate
fi

# Upgrade pip
python -m pip install --upgrade pip

# Validate if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment activation failed."
    exit 1
else
    echo "Virtual environment activated successfully."
fi

# Install required dependencies for building PyTorch
pip install ninja setuptools cmake pyyaml build typing_extensions

# Clone PyTorch repository
if [ ! -d "pytorch" ]; then
    git clone --recursive https://github.com/pytorch/pytorch.git
    echo "Cloned PyTorch repository."
else
    echo "PyTorch repository already exists. Updating..."
    cd pytorch && git submodule sync && git submodule update --init --recursive && cd ..
fi

cd pytorch

# Check operating system and configure the build
if [[ "$OS_TYPE" == "Darwin" ]]; then
    # MacOS build configuration
    pip install torch
elif [[ "$OS_TYPE" == "MINGW"* || "$OS_TYPE" == "CYGWIN"* || "$OS_TYPE" == "MSYS_NT"* ]]; then
    # Windows build configuration
    export USE_CUDA=0
    export USE_DISTRIBUTED=0
    export USE_MKLDNN=0
    export BUILD_TEST=0
    export USE_NCCL=0
    export USE_QNNPACK=0
    export USE_TENSORPIPE=0
    export MAX_JOBS=4
    export USE_OPENMP=0
    python setup.py clean
    python setup.py install
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

cd ..

python -m build
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade -e .[devel]
pip install openvino transformers

# End of script
echo "PyTorch has been built and installed in the virtual environment."
echo "Process has ended, you are now ready to use the virtual environment."
