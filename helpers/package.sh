#!/bin/bash

# helpers/package.sh

# Check if the virtual environment exists
if [ ! -d "env" ]; then
    echo "Virtual environment 'env' does not exist. Please create it first. by running make setup command."
    exit 1
fi

# Check if launch.spec exists in the project root folder
if [ ! -f "launch.spec" ]; then
    echo "launch.spec does not exist in the project root folder. Please ensure it is present."
    exit 1
fi

# Remove previous build directory
rm -rf dist/
echo "Previous build artifacts removed."

# Activate virtual environment
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source env/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source env/Scripts/activate
else
    echo "Unsupported OS type: $OSTYPE"
    exit 1
fi

# Validate if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment activation failed."
    exit 1
else
    echo "Virtual environment activated successfully."
fi

# Run PyInstaller with launch.spec
pyinstaller launch.spec

# End of script
echo "Packaging process has ended."