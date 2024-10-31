#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Load Conda into the current shell session
echo "Loading Conda..."
if [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    source "$(conda info --base)/shell/condabin/conda-hook.ps1"
else
    # Fallback when conda.sh is not found
    if [[ ! -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        echo "Conda initialization script not found."
    else
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
fi

ENV_NAME="kantodex-classifier"

# Check if the environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating..."
    conda env update -n ${ENV_NAME} -f environment.yml --prune
else
    echo "Creating environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

# Activate the environment
echo "Activating environment '${ENV_NAME}'..."
conda activate ${ENV_NAME}

# Install additional Python packages if requirements.txt exists
if [[ -f requirements.txt ]]; then
    echo "Installing additional packages from requirements.txt..."
    pip install --no-cache-dir -r requirements.txt
fi

echo "Environment setup complete."
