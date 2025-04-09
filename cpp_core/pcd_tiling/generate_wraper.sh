mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# -j option specifies the number of jobs (parallel processes), 8 indicates the number of parallel jobs to speed up
make -j 8
swig -c++ -python ../pcd_tiling.i
python -c "import pcd_tiling"

#cmake -DCMAKE_BUILD_TYPE=Release .
## -j option specifies the number of jobs (parallel processes), 8 indicates the number of parallel jobs to speed up
#make -j 8
#swig -c++ -python pcd_tiling.i
#python -c "import pcd_tiling"