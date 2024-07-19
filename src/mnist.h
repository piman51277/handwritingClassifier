#pragma once
#include <fstream>
#include <cinttypes>
#include "net.h"
#include "vector.h"

TrainingDataSet get_MNIST(const char *images, const char *labels);