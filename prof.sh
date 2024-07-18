
export CXXFLAGS="-Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays"
g++ src/*.cpp -g --std=c++23 $CXXFLAGS -o bin/hot
perf record --call-graph dwarf -F 500 ./bin/hot
hotspot perf.data