
export CXXFLAGS="-Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays"
export CUDAFLAGS="-std=c++20 -O3 -diag-suppress 186"

#tools
#g++ tools/generateMatrix.cpp $CXXFLAGS -o generateMatrix

#main script
echo "Compiling CPU..."
g++ src/cpu/*.cpp $CXXFLAGS -o bin/classifier_baseline
echo "Compiling GPU..."
nvcc src/gpu/*.cu -O3 -diag-suppress 186 -gencode=arch=compute_86,code=sm_86 -o bin/classifier