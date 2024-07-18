#include "net.h"
#include <iostream>

void activation(Vector &v)
{
  // sigmoid
  for (uint32_t i = 0; i < v.dim; i++)
  {
    v.vec.get()[i] = 1 / (1 + exp(-v.vec.get()[i]));
  }
}

Vector _activation(Vector &v)
{
  Vector result = copyVector(v);
  activation(result);
  return result;
}

Vector activation_inverse(Vector &v)
{
  Vector result = copyVector(v);
  for (uint32_t i = 0; i < v.dim; i++)
  {
    double sig = 1 / (1 + exp(-v.vec.get()[i]));
    result.vec.get()[i] = sig * (1 - sig);
  }
  return result;
}

double MSE(Vector &actual, Vector &expected)
{
  double sum = 0;
  for (uint32_t i = 0; i < actual.dim; i++)
  {
    sum += (actual.vec.get()[i] - expected.vec.get()[i]) * (actual.vec.get()[i] - expected.vec.get()[i]);
  }
  return sum / actual.dim;
}

Net::Net(uint32_t numLayers, uint32_t *layerSizes)
{
  this->numLayers = numLayers;
  this->layers = new Layer[numLayers];

  // initialize layers
  for (uint32_t i = 0; i < numLayers; i++)
  {
    this->layers[i].dim = layerSizes[i];
    if (i > 0)
    {
      this->layers[i].weights = createMatrix(layerSizes[i], layerSizes[i - 1]);
      this->layers[i].biases = createVector(layerSizes[i]);
    }
  }
}

Net::~Net()
{
  delete[] this->layers;
}

void Net::initializeWeights()
{
  std::mt19937 mt{std::random_device{}()};
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    Matrix &weights = this->layers[i].weights;
    Vector &biases = this->layers[i].biases;

    for (uint32_t j = 0; j < weights.dim1 * weights.dim2; j++)
    {
      weights.mat.get()[j] = dist(mt);
    }

    for (uint32_t j = 0; j < biases.dim; j++)
    {
      biases.vec.get()[j] = dist(mt);
    }
  }
}

Vector Net::evaluate(Vector &input)
{
  uint32_t inputSize = this->layers[0].dim;
  if (input.dim != inputSize)
  {
    throw std::runtime_error("Input size does not match network input size");
  }

  Vector result = copyVector(input);
  for (uint32_t i = 1; i < this->numLayers; i++)
  {

    result = mulVector(this->layers[i].weights, result);

    result = addVector(result, this->layers[i].biases);
    activation(result);
  }
  return result;
}

double Net::error(TrainingDataSet &dataSet)
{
  double sum = 0;
  for (uint32_t i = 0; i < dataSet.numData; i++)
  {
    TrainingData &data = dataSet.data[i];
    Vector result = evaluate(data.input);
    double error = MSE(result, data.output);
    sum += error;
  }
  return sum / dataSet.numData;
}
double Net::error_verbose(TrainingDataSet &dataSet)
{
  double sum = 0;
  for (uint32_t i = 0; i < dataSet.numData; i++)
  {
    TrainingData &data = dataSet.data[i];
    Vector result = evaluate(data.input);
    double error = MSE(result, data.output);
    std::cout << "Error: " << error << std::endl;
    std::cout << "Expected: ";
    printVector(data.output);
    std::cout << "Actual: ";
    printVector(result);
    sum += error;
  }
  return sum / dataSet.numData;
}

void Net::evaluateMonit(Vector &input, Vector *layerInputs)
{
  uint32_t inputSize = this->layers[0].dim;
  if (input.dim != inputSize)
  {
    throw std::runtime_error("Input size does not match network input size");
  }

  layerInputs[0] = copyVector(input);
  Vector result = copyVector(input);
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    result = mulVector(this->layers[i].weights, result);
    addVectorSelf(result, this->layers[i].biases);
    layerInputs[i] = copyVector(result);
    activation(result);
  }
}

void Net::train(TrainingDataSet &dataSet, TrainConfig &config)
{

  Matrix *gradientSums = new Matrix[this->numLayers];
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    gradientSums[i] = createMatrix(this->layers[i].weights.dim1, this->layers[i].weights.dim2);
  }

  // for momentum
  Matrix *lastGradients = new Matrix[this->numLayers];
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    lastGradients[i] = createMatrix(this->layers[i].weights.dim1, this->layers[i].weights.dim2);
  }

  Vector *biasSums = new Vector[this->numLayers];
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    biasSums[i] = createVector(this->layers[i].dim);
  }

  // prevent reallocation
  Vector *layerInputs = new Vector[this->numLayers];

  // the 0th index is unused, but kept for easier indexing
  Vector *deltas = new Vector[this->numLayers];

  for (uint32_t i = 0; i < this->numLayers; i++)
  {
    deltas[i] = createVector(this->layers[i].dim);
  }

  for (uint32_t epoch = 0; epoch < config.numEpochs; epoch++)
  {
    // reset sums
    for (uint32_t i = 1; i < this->numLayers; i++)
    {
      rstMatrix(gradientSums[i]);
    }

    // reset biases
    for (uint32_t i = 1; i < this->numLayers; i++)
    {
      rstVector(biasSums[i]);
    }

    const uint32_t batchSize = config.batchSize;
    uint32_t ptr = 0;
    while (ptr < dataSet.numData)
    {
      uint32_t end = std::max(ptr + batchSize, dataSet.numData);
      for (uint32_t caseNum = ptr; caseNum < end; caseNum++)
      {
        TrainingData &data = dataSet.data[caseNum];
        evaluateMonit(data.input, layerInputs);

        // work backwards
        for (uint32_t layer = this->numLayers - 1; layer >= 1; layer--)
        {
          Vector &layerInputSum = layerInputs[layer];
          Vector layerOutput = _activation(layerInputSum);
          Vector net_j_rInv = activation_inverse(layerInputSum);
          Vector &expected = data.output;
          Vector &delta = deltas[layer];
          // calculate delta
          if (layer == this->numLayers - 1)
          {

            // output layer
            for (uint32_t i = 0; i < delta.dim; i++)
            {
              delta.vec.get()[i] = (layerOutput.vec.get()[i] - expected.vec.get()[i]) * net_j_rInv.vec.get()[i];
            }
          }
          else
          {
            // inner layers
            Vector &lastdelta = deltas[layer + 1];
            Matrix &lastWeights = this->layers[layer + 1].weights;
            for (uint32_t i = 0; i < delta.dim; i++)
            {
              double sum = 0;
              for (uint32_t j = 0; j < lastdelta.dim; j++)
              {
                // each row is an output, each column is an input
                sum += lastdelta.vec.get()[j] * lastWeights.mat.get()[j * layerInputs[layer + 1].dim + i];
              }
              delta.vec.get()[i] = sum * net_j_rInv.vec.get()[i];
            }
          }

          // calculate and sum gradients
          Matrix &grad = gradientSums[layer];
          for (uint32_t i = 0; i < grad.dim1; i++) // rows (outputs)
          {
            for (uint32_t j = 0; j < grad.dim2; j++) // columns (inputs)
            {
              grad.mat.get()[i * grad.dim2 + j] += delta.vec.get()[i] * layerInputs[layer - 1].vec.get()[j];
            }
          }

          // calculate and sum biases
          Vector &biasSum = biasSums[layer];
          for (uint32_t i = 0; i < biasSum.dim; i++)
          {
            biasSum.vec.get()[i] += delta.vec.get()[i];
          }
        }
      }

      // learning rate / numData (to simplify eta * sum / numData * -1)
      const double adjScalarWeight = config.eta_weight / batchSize * -1.0;
      const double adjScalarBias = config.eta_bias / batchSize * -1.0;

      // compute update factors and update weights and biases
      for (uint32_t i = 1; i < this->numLayers; i++)
      {
        Matrix &grad = gradientSums[i];
        Vector &biasSum = biasSums[i];
        Matrix &lastGrad = lastGradients[i];

        // momentum
        mulMatrix(lastGrad, -1.0 * config.momentum, lastGrad);
        addMatrixSelf(this->layers[i].weights, lastGrad);

        // copy gradients for next iteration
        cpyMatrix(grad, lastGrad);

        mulMatrix(grad, adjScalarWeight, grad);
        mulVector(biasSum, adjScalarBias, biasSum);

        addMatrixSelf(this->layers[i].weights, grad);
        addVectorSelf(this->layers[i].biases, biasSum);
      }

      ptr += batchSize;
    }
  }
};