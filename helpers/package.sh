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

# Find the Qt plugins path
QT_PLUGIN_PATH=$(find env/ -name "qwindows.dll" | xargs dirname | xargs dirname)

# Check if the QT_PLUGIN_PATH is found
if [[ -z "$QT_PLUGIN_PATH" ]]; then
    echo "Could not find Qt plugins. Please ensure PySide6 is installed in the virtual environment."
    exit 1
fi

# Run Nuitka with automatic "Yes" to all prompts and include the Qt plugins directory, enabling the pyside6 plugin
yes | python -m nuitka --standalone --follow-imports --include-plugin-directory="$QT_PLUGIN_PATH" --enable-plugin=pyside6 --include-data-file=config.ini=config.ini --include-data-dir=assets=assets main.py

# End of script
echo "Packaging process has ended."
