#!/bin/bash

# helpers/package.sh

# Check if the virtual environment exists
if [ ! -d "env" ]; then
    echo "Virtual environment 'env' does not exist. Please create it first by running the make setup command."
    exit 1
fi

# Remove previous build directories
rm -rf main.build main.dist __nuitka_build__
echo "Previous build artifacts removed."

# Activate virtual environment
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source env/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source env/Scripts/activate
elif [[ "$OSTYPE" == "win32" ]]; then
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

cxfreeze --script launch.py

# End of script
echo "Packaging process has ended."
