#!/usr/bin/env bash

# Install Miniconda
MINICONDA=Miniconda3-latest-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/$MINICONDA
bash $MINICONDA -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Initialize conda
conda init bash
source ~/.bashrc

# Create and activate environment
conda env create -f conda_env.yaml
conda activate graph_molec  # Replace with your env name

