#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating data directory..."
mkdir data

echo "Downloading models..."
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P data/

echo "Cloning submodules..."
git submodule init
git submodule update

echo "Cloning deepsvg..."
git clone https://github.com/alexandre01/deepsvg.git