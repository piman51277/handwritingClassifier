#pragma once
#include <cinttypes>
#include <fstream>
#include <memory>
#include <vector>
#include "matrix.h"

struct TrainingData
{
  // these are col-wise
  Matrix input;
  Matrix expected;
};

struct TrainingDataSet
{
  uint32_t numData;
  TrainingData *data;

  TrainingDataSet() : numData(0), data(nullptr) {}
  ~TrainingDataSet()
  {
    if (data != nullptr)
    {
      cudaFree(data);
    }
  }
};

struct Layer
{
  uint32_t dim;
  Matrix weights;
  Vector biases;
};

struct TrainConfig
{
  uint32_t numEpochs;
  uint32_t batchSize;
  double eta_weight;
  double eta_bias;
  double momentum;
};

class Net
{
public:
  uint32_t numLayers;
  Layer *layers;
  void evaluateMonit(Matrix &input, Matrix *layerInputs);

public:
  Net(std::vector<uint32_t> &layerSizes);
  ~Net();
  void save(const char *filename);
  static Net load(const char *filename);

  void initializeWeights();
  Vector evaluate(Vector &input);
  Matrix evaluate(Matrix &input);
  double error(TrainingDataSet &dataSet);
  double error_percent(TrainingDataSet &dataSet);
  void train(TrainingDataSet &dataSet, TrainConfig &config);
};
