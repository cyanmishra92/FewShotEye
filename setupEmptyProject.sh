#!/bin/bash

# Step 1: Create project directory structure
mkdir -p ./{data,models,notebooks,src,tests}

# Step 2: Create empty Python files and other necessary files
echo "import torch" > src/__init__.py
touch src/main.py src/model.py src/data_loader.py src/train.py src/test.py
touch tests/__init__.py


# Step 4: Set up a Conda environment and activate it
conda create --name FewShot python=3.8 -y
conda activate FewShot

# Step 5: Install required Python packages
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy pandas matplotlib -c conda-forge

# Step 6: Initialize Git repository and make the first commit
git init
git add .
git commit -m "Initial project setup with basic directory structure and files"

# Step 7: Link your GitHub repository and push
echo "Push to GitHub:"
git pull
git push

# Complete setup message
echo "Project setup is complete and pushed to GitHub."

