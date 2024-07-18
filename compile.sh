
export CXXFLAGS="-Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays"
export CUDAFLAGS="-std=c++20 -O3 -diag-suppress 186"


#g++ tools/generateMatrix.cpp -Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays -o generateMatrix
g++ src/*.cpp $CXXFLAGS -o bin/classifier
#nvcc src/*.cu -O3 -diag-suppress 186 -gencode=arch=compute_86,code=sm_86 -o bin/a.out