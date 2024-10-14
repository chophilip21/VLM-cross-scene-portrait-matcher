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

# Run make commands
pip install build
python -m build
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade -e .[devel]


# End of script
echo "Process has ended."
