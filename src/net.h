#pragma once
#include <cinttypes>
#include <memory>
#include "matrix.h"

struct TrainingData
{
  Vector input;
  Vector output;
};

struct TrainingDataSet
{
  uint32_t numData;
  TrainingData *data;

  TrainingDataSet() : numData(0), data(nullptr) {}
  ~TrainingDataSet()
  {
    if (data != nullptr)
      delete[] data;
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
  void evaluateMonit(Vector &input, Vector *layerInputs);

public:
  Net(uint32_t numLayers, uint32_t *layerSizes);
  ~Net();
  void initializeWeights();
  Vector evaluate(Vector &input);
  double error(TrainingDataSet &dataSet);
  double error_verbose(TrainingDataSet &dataSet);
  void train(TrainingDataSet &dataSet, TrainConfig &config);
};
