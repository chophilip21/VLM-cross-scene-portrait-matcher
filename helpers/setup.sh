#!/bin/bash 

# helpers/setup.sh

# 1) Check if the virtual environment exists
if [ -d "env" ]; then

    # If a virtual environment might be active, deactivate it first
    if [[ -n "$VIRTUAL_ENV" && "$VIRTUAL_ENV" == *"/env" ]]; then
        echo "Deactivating currently active virtual environment..."
        deactivate
    fi

    read -p "Virtual environment 'env' already exists. Do you want to recreate it? (yes/no): " yn
    case $yn in
        [Yy]* ) 
            echo "Removing existing 'env' directory..."
            rm -rf env
            ;;
        [Nn]* ) 
            echo "Using existing virtual environment."
            ;;
        * ) 
            echo "Please answer yes or no."
            exit 1
            ;;
    esac
fi

# 2) Create the virtual environment if it's missing
if [ ! -d "env" ]; then
    python -m venv env
    echo "Virtual environment created."
fi

# 3) Detect OS and activate the virtual environment
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == "MINGW"* || "$OS_TYPE" == "CYGWIN"* || "$OS_TYPE" == "MSYS_NT"* ]]; then
    # Windows activation
    source env/Scripts/activate
else
    # Linux or macOS activation
    source env/bin/activate
fi

# 4) Confirm the virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment activation failed."
    exit 1
else
    echo "Virtual environment activated successfully."
fi

# 5) Upgrade pip
python -m pip install --upgrade pip

# 6) PyTorch installation logic based on OS
if [[ "$OS_TYPE" == "MINGW"* || "$OS_TYPE" == "CYGWIN"* || "$OS_TYPE" == "MSYS_NT"* ]]; then
    # On Windows, ask if user wants to build from source
    read -p "Do you want to build PyTorch from source? (yes/no): " build_source

    if [[ "$build_source" == "yes" ]]; then
        echo "User chose to build PyTorch from source."
        rm -rf pytorch

        # Install prerequisites needed for building from source (including NumPy!)
        pip install ninja setuptools cmake pyyaml build typing_extensions numpy

        echo -e "\e[1;33mInstall Visual Studio with the 'Desktop development with C++' workload.\e[0m"
        echo -e "\e[1;34mDownload here: https://visualstudio.microsoft.com/visual-cpp-build-tools/\e[0m"

        # Clone or update PyTorch repo
        if [ ! -d "pytorch" ]; then
            git clone --recursive https://github.com/pytorch/pytorch.git
            echo "Cloned PyTorch repository."
        else
            echo "PyTorch repository already exists. Updating..."
            cd pytorch
            git submodule sync
            git submodule update --init --recursive
            cd ..
        fi

        # Build from source
        cd pytorch
        # Example CPU-only build flags
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
        cd ..
    else
        echo "User chose to install PyTorch via pip (CPU-only)."
        pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
    fi
else
    # On macOS or Linux, skip the build and install directly
    echo "Non-Windows OS detected. Installing PyTorch via pip (CPU-only)."
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
fi

# 7) Continue with the rest of the packages
python -m build
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade -e .[devel]
pip install openvino transformers

# End of script
echo "PyTorch setup complete (either built from source or installed CPU-only)."
echo "OpenVINO, transformers, and other packages have been installed."
echo "You are now ready to use the virtual environment."
