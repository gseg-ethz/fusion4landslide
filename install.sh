#!/bin/bash

# Stop script immediately if any command fails
set -e

echo "---------------------------------------------"
echo "----------- Env Installation -----------"
echo "---------------------------------------------"

PROJECT_NAME=fusion4landslide
PYTHON=3.8

# >>> make conda activate work in script
if ! command -v conda &> /dev/null; then
  echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi
eval "$(conda shell.bash hook)"
# <<<

echo "Creating conda env: ${PROJECT_NAME} (python=${PYTHON})"
conda create -n "${PROJECT_NAME}" python="${PYTHON}" -y
conda activate "${PROJECT_NAME}"

# -------------------------
# 1) Install PyTorch + CUDA
# -------------------------
# Match: torch 2.4.1 + cu124
TORCH_VERSION=2.4.1
TV_VERSION=0.19.1
TA_VERSION=2.4.1
CUDA_VERSION=12.4

echo "Install PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}"
conda install pytorch=="${TORCH_VERSION}" torchvision=="${TV_VERSION}" torchaudio=="${TA_VERSION}" \
  pytorch-cuda="${CUDA_VERSION}" cuda-toolkit="${CUDA_VERSION}" cuda-nvcc="${CUDA_VERSION}" cuda-compiler="${CUDA_VERSION}" gcc_linux-64=13 gxx_linux-64=13 cuda-cudart-dev="${CUDA_VERSION}" cuda-cccl="${CUDA_VERSION}" -c pytorch -c nvidia -c conda-forge -y

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'is_available:', torch.cuda.is_available())"

# -------------------------
# 2) Install PyG stack (torch-geometric + extensions)
# -------------------------
# PyG wheels index for torch 2.4.1 + cu124
PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu124.html"

echo "Install PyTorch Geometric stack from: ${PYG_WHL_URL}"
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f "${PYG_WHL_URL}"
pip install torch-geometric==2.3.0

python -c "import torch_geometric; print('torch_geometric:', torch_geometric.__version__)"

# -------------------------
# 3) Install python deps
# -------------------------
echo "Install python requirements"
pip install -r requirements.txt

# -------------------------
# 4) Build your cpp_core wrappers
# -------------------------
echo "Build cpp_core wrappers"

# Force C++ extensions to strictly use the Conda environment's CUDA toolkit
export CUDA_HOME=$CONDA_PREFIX

# Install missing build dependencies, forcing PCL 1.13+ and VTK
conda install cmake make swig "pcl>=1.13" vtk libboost-python -c conda-forge -y

pushd cpp_core/pcd_tiling
sed -i 's/boost_python311/boost_python38/g' CMakeLists.txt
bash generate_wraper.sh
popd

# install it when using supervoxel_seegmentation for partitioning
pushd cpp_core/supervoxel_segmentation
sed -i 's/boost_python311/boost_python38/g' CMakeLists.txt
bash generate_wraper.sh
popd

# -------------------------
# 5) FRNN (as in your script)
# -------------------------
echo "⭐ Installing FRNN"
# Remove any existing FRNN artifacts if the script failed previously
rm -rf superpoint_transformer/src/dependencies/FRNN

git clone https://github.com/zhaoyiww/superpoint_transformer.git || true
cd superpoint_transformer
mkdir -p src/dependencies
if [ ! -d "src/dependencies/FRNN" ]; then
  git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN
fi

# Force the compiler to target known, stable GPU architectures and allow GCC 13
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
export RELAXED_CUDA_VERSION_CHECK=1

# Unset the flags that broke standard GCC just in case they are lingering
unset CFLAGS
unset CXXFLAGS

pushd src/dependencies/FRNN/external/prefix_sum
python setup.py install
popd

pushd src/dependencies/FRNN
python setup.py install
popd

# -------------------------
# 6) point_geometric_features
# -------------------------
echo "⭐ Installing Point Geometric Features"
conda install -c conda-forge libstdcxx-ng -y
pip install git+https://github.com/drprojects/point_geometric_features.git

# -------------------------
# 7) parallel-cut-pursuit deps
# -------------------------
echo "⭐ Installing Parallel Cut-Pursuit"
if [ ! -d "src/dependencies/parallel_cut_pursuit" ]; then
  git clone https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git src/dependencies/parallel_cut_pursuit
fi
if [ ! -d "src/dependencies/grid_graph" ]; then
  git clone https://gitlab.com/1a7r0ch3/grid-graph.git src/dependencies/grid_graph
fi

python scripts/setup_dependencies.py build_ext

cd ..
echo "✅ Done."