mkdir build
cd build

#numpy.get_include()
numpy_include=$(python -c "import numpy; print(numpy.get_include())")
echo $numpy_include

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I\ $numpy_include ..
make -j 8
swig -c++ -python ../supervoxel.i
python -c "import supervoxel"

#numpy_include=$(python -c "import numpy; print(numpy.get_include())")
#
#cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I\ $numpy_include .
#make -j 8
#swig -c++ -python supervoxel.i
#python -c "import supervoxel"
