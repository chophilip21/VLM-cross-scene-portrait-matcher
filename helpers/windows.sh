#!/bin/bash

# Step 1: Check if windows.iss exists
if [[ ! -f "windows.iss" ]]; then
  echo "Error: windows.iss file not found!"
  exit 1
fi
echo "Step 1: windows.iss file found."

# Step 2: Prompt for the real Dropbox API key
read -p "Enter the real Dropbox API key (must be 143 characters): " real_api_key

# Step 3: Check if the API key length is correct
if [[ ${#real_api_key} -ne 143 ]]; then
  echo "Error: API key must be exactly 143 characters long!"
  exit 1
fi
echo "Step 2: Real Dropbox API key received."

# Step 4: Copy contents of windows.iss and replace the fake key with the real one
fake_key="ADS02323SD123"
cp windows.iss compile.iss
sed -i "s/$fake_key/$real_api_key/g" compile.iss
echo "Step 3: Fake key replaced with the real key in compile.iss."

# Step 5: Confirm success
if [[ -f "compile.iss" ]]; then
  echo "Step 4: compile.iss file created successfully."
else
  echo "Error: Failed to create compile.iss."
  exit 1
fi
