#!/bin/bash

# run this with 'source install.sh'
# remove the virtual env
# conda remove -n fusion4landslide --all -y

echo "---------------------------------------------"
echo
echo "----------- Env Installation -----------"
echo
echo "---------------------------------------------"

PROJECT_NAME=fusion4landslide
PYTHON=3.8
#CUDA_VERSION=11.8

# create and activate a new virtual environment
echo "Creating and activate a new virtual environment '${PROJECT_NAME}'"
conda create -n ${PROJECT_NAME} python=${PYTHON} -y
conda activate ${PROJECT_NAME}

# CUDA 11.8, from https://pytorch.org/get-started/previous-versions/
CUDA_VERSION=11.8
echo "Install CUDA ${CUDA_VERSION}"
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y

# install necessary libraries
echo "Install necessary libraries"
pip install open3d
pip install easydict
pip install coloredlogs
# may only used for f2s3 project
pip install hnswlib
pip install pytorch-lightning


# install cpp_core
cd cpp_core/pcd_tiling
source generate_wraper.sh
cd ..
cd supervoxel_segmentation
source generate_wraper.sh
cd ..

