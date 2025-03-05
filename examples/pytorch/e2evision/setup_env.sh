#!/bin/bash
set -e  # Exit immediately if any command fails

# Set the environment name and Python version
ENV_NAME="lightning-e2e"
PYTHON_VERSION="3.11"

# Function to check if conda environment exists
env_exists() {
    conda env list | grep -q "^${ENV_NAME}\s"
}

# Check if environment exists
if env_exists; then
    echo "Environment ${ENV_NAME} already exists. Activating and updating dependencies..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
else
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python=${PYTHON_VERSION} -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

# Create a requirements.txt file (adjust dependencies as needed for your project)
echo "Generating requirements.txt file..."
cat > requirements.txt << 'EOF'
scipy
matplotlib
seaborn
scikit-learn
torchvision
lightning
opencv-python
tqdm
wandb
EOF

# Install/update dependencies using pip
echo "Installing/updating dependencies from requirements.txt..."
pip install -r requirements.txt --upgrade

echo "Environment ${ENV_NAME} is ready and dependencies are installed/updated."
