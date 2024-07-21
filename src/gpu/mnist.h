#pragma once
#include <fstream>
#include <cinttypes>
#include "net.h"
#include "matrix.h"

TrainingDataSet get_MNIST(const char *images, const char *labels, uint32_t batchSize);