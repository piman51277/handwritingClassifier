#include "matrix.h"
#include "net.h"
#include "activation.h"
#include "constants.h"
#include "fused.h"

// small helper to estimate training time completion
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

Net::Net(std::vector<uint32_t> &layers)
{
  this->numLayers = layers.size();
  this->layers = new Layer[layers.size()];

  for (uint32_t i = 0; i < layers.size(); i++)
  {
    this->layers[i].dim = layers[i];
    if (i != 0)
    {
      this->layers[i].weights = Matrix::create(layers[i], layers[i - 1]);
      this->layers[i].biases = Vector::create(layers[i]);
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
    double *w;
    cudaMallocHost(&w, weights.dim1 * weights.dim2 * sizeof(double));

    for (uint32_t j = 0; j < weights.dim1; j++)
    {
      for (uint32_t k = 0; k < weights.dim2; k++)
      {
        w[j * weights.dim2 + k] = dist(mt);
      }
    }
    cudaMemcpy(weights.mat.get(), w, weights.dim1 * weights.dim2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaFreeHost(w);

    Vector &biases = this->layers[i].biases;
    double *b;
    cudaMallocHost(&b, biases.dim * sizeof(double));

    for (uint32_t j = 0; j < biases.dim; j++)
    {
      b[j] = dist(mt);
    }
    cudaMemcpy(biases.vec.get(), b, biases.dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaFreeHost(b);
  }
}

void Net::save(const char *filename)
{
  std::ofstream file(filename, std::ios::binary);

  // write the number of layers
  file.write((char *)&this->numLayers, sizeof(uint32_t));

  // write the dimensions of each layer
  for (uint32_t i = 0; i < this->numLayers; i++)
  {
    file.write((char *)&this->layers[i].dim, sizeof(uint32_t));
  }

  // write the weights and biases
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    Matrix &weights = this->layers[i].weights;
    Vector &biases = this->layers[i].biases;

    double *w;
    cudaMallocHost(&w, weights.dim1 * weights.dim2 * sizeof(double));
    cudaMemcpy(w, weights.mat.get(), weights.dim1 * weights.dim2 * sizeof(double), cudaMemcpyDeviceToHost);

    double *b;
    cudaMallocHost(&b, biases.dim * sizeof(double));
    cudaMemcpy(b, biases.vec.get(), biases.dim * sizeof(double), cudaMemcpyDeviceToHost);

    file.write((char *)w, weights.dim1 * weights.dim2 * sizeof(double));
    file.write((char *)b, biases.dim * sizeof(double));

    cudaFreeHost(w);
    cudaFreeHost(b);
  }
}

Net Net::load(const char *filename)
{
  std::ifstream file(filename, std::ios::binary);

  uint32_t numLayers;
  file.read((char *)&numLayers, sizeof(uint32_t));

  std::vector<uint32_t> layers;
  for (uint32_t i = 0; i < numLayers; i++)
  {
    uint32_t dim;
    file.read((char *)&dim, sizeof(uint32_t));
    layers.push_back(dim);
  }

  Net net(layers);

  for (uint32_t i = 1; i < numLayers; i++)
  {
    Matrix &weights = net.layers[i].weights;
    Vector &biases = net.layers[i].biases;

    double *w;
    cudaMallocHost(&w, weights.dim1 * weights.dim2 * sizeof(double));
    file.read((char *)w, weights.dim1 * weights.dim2 * sizeof(double));
    cudaMemcpy(weights.mat.get(), w, weights.dim1 * weights.dim2 * sizeof(double), cudaMemcpyHostToDevice);

    double *b;
    cudaMallocHost(&b, biases.dim * sizeof(double));
    file.read((char *)b, biases.dim * sizeof(double));
    cudaMemcpy(biases.vec.get(), b, biases.dim * sizeof(double), cudaMemcpyHostToDevice);

    cudaFreeHost(w);
    cudaFreeHost(b);
  }

  return net;
}

Vector Net::evaluate(Vector &input)
{
  Vector result = Vector::copy(input);
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    result = Matrix::mul(this->layers[i].weights, result);
    result.add(this->layers[i].biases);
    ActivationFunction(result);
  }
  return result;
}

Matrix Net::evaluate(Matrix &input)
{
  Matrix result = Matrix::copy(input);
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    result = Matrix::mul(this->layers[i].weights, result);
    result.addColwise(this->layers[i].biases);
    ActivationFunction(result);
  }
  return result;
}

void Net::evaluateMonit(Matrix &input, Matrix *layerInputs)
{
  layerInputs[0] = Matrix::copy(input);
  Matrix result = Matrix::copy(input);
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    // result = Matrix::mul(this->layers[i].weights, result);
    // result.addColwise(this->layers[i].biases);
    result = fused_mul_colwise(this->layers[i].weights, result, this->layers[i].biases);
    layerInputs[i] = Matrix::copy(result);
    ActivationFunction(result);
  }
}

double Net::error(TrainingDataSet &dataSet)
{
  double sum = 0;
  uint32_t count = 0;
  for (uint32_t i = 0; i < dataSet.numData; i++)
  {
    Matrix &input = dataSet.data[i].input;
    Matrix &expected = dataSet.data[i].expected;
    Matrix result = evaluate(input);
    sum += MSE(result, expected);
    count += expected.dim2;
  }

  return sum / count;
}

double Net::error_percent(TrainingDataSet &dataSet)
{
  uint32_t correct = 0;
  uint32_t count = 0;
  for (uint32_t i = 0; i < dataSet.numData; i++)
  {
    Matrix &input = dataSet.data[i].input;
    Matrix &expected = dataSet.data[i].expected;
    Matrix result = evaluate(input);

    // copt the expected and result matrix to host
    double *expected_host;
    double *result_host;
    cudaMallocHost(&expected_host, expected.dim1 * expected.dim2 * sizeof(double));
    cudaMallocHost(&result_host, result.dim1 * result.dim2 * sizeof(double));
    cudaMemcpy(expected_host, expected.mat.get(), expected.dim1 * expected.dim2 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_host, result.mat.get(), result.dim1 * result.dim2 * sizeof(double), cudaMemcpyDeviceToHost);

    uint32_t numRow = expected.dim1;
    uint32_t numCol = expected.dim2;

    for (uint32_t col = 0; col < numCol; col++)
    {
      // find the index of the max value in the result matrix
      uint32_t maxIndex = 0;
      for (uint32_t row = 1; row < numRow; row++)
      {
        if (result_host[col + numCol * row] > result_host[col + numCol * maxIndex])
        {
          maxIndex = row;
        }
      }

      // check if the corresponding expected value is 1 (bigger than 0.99)
      if (expected_host[col + numCol * maxIndex] > 0.99)
      {
        correct++;
      }
    }

    count += expected.dim2;
    cudaFreeHost(expected_host);
    cudaFreeHost(result_host);
  }

  return (double)correct / count;
}

// deltas is dim1 x cases
// inputs is dim2 x cases
__global__ void compGradKern(double *deltas, double *inputs, double *output, uint32_t dim1, uint32_t dim2, uint32_t cases)
{
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = index; i < dim1 * dim2; i += stride)
  {
    // relative to gradient matrix
    const uint32_t row = i / dim2;
    const uint32_t col = i % dim2;
    const uint32_t dOff = row * cases;
    const uint32_t iOff = col * cases;

    double sum = 0.0;
#pragma unroll
    for (uint32_t c = 0; c < cases; c++)
    {
      sum += deltas[dOff + c] * inputs[iOff + c];
    }
    output[i] = sum;
  }
}

__global__ void compBiasGradKern(double *deltas, double *output, uint32_t dim1, uint32_t cases)
{
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = index; i < dim1; i += stride)
  {
    double sum = 0.0;
#pragma unroll
    for (uint32_t c = 0; c < cases; c++)
    {
      sum += deltas[i * cases + c];
    }
    output[i] = sum;
  }
}

void Net::train(TrainingDataSet &dataSet, TrainConfig &config)
{
  // Pre-allocate memory to avoid allocation in the loop
  Matrix *gradients = new Matrix[this->numLayers];
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    gradients[i] = Matrix::create(this->layers[i].weights.dim1, this->layers[i].weights.dim2);
  }

  Vector *biasGradients = new Vector[this->numLayers];
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    biasGradients[i] = Vector::create(this->layers[i].biases.dim);
  }

  Matrix *lastGradients = new Matrix[this->numLayers];
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    lastGradients[i] = Matrix::create(this->layers[i].weights.dim1, this->layers[i].weights.dim2, 0.0);
  }

  Matrix *layerInputs = new Matrix[this->numLayers];
  for (uint32_t i = 0; i < this->numLayers; i++)
  {
    layerInputs[i] = Matrix::create(this->layers[i].dim, config.batchSize);
  }

  Matrix *deltas = new Matrix[this->numLayers];
  for (uint32_t i = 1; i < this->numLayers; i++)
  {
    deltas[i] = Matrix::create(this->layers[i].dim, config.batchSize);
  }

  auto t1 = Clock::now();

  for (uint32_t epoch = 0; epoch < config.numEpochs; epoch++)
  {
    auto t2 = Clock::now();
    uint32_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    uint32_t timeRemaining = (uint32_t)((config.numEpochs - epoch) * ms);
    uint32_t timeRemainingSec = timeRemaining / 1000;
    uint32_t timeRemainingMin = timeRemainingSec / 60;
    timeRemainingSec = timeRemainingSec % 60;
    t1 = t2;

    // reset the cursor to the beginning of the line
    std::cout << "\r";
    std::cout << "Started Epoch " << epoch + 1 << "/" << config.numEpochs << " -- Dur " << ms << "ms -- Est " << timeRemainingMin << "m " << timeRemainingSec << "s " << std::flush;

    // the cases are already pre-batched
    for (uint32_t tCase = 0; tCase < dataSet.numData; tCase++)
    {
      TrainingData &data = dataSet.data[tCase];
      evaluateMonit(data.input, layerInputs);

      // backprop
      for (uint32_t layer = this->numLayers - 1; layer >= 1; layer--)
      {
        Matrix &layerInput = layerInputs[layer];
        Matrix layerOutput = ActivationFunctionCopy(layerInput);
        Matrix netj_rhoP = ActivationFunctionPrime(layerOutput);
        Matrix &expected = data.expected;
        Matrix &delta = deltas[layer];

        // compute delta

        // if last layer
        if (layer == this->numLayers - 1)
        {
          // Matrix::sub(layerOutput, expected, delta);
          // Matrix::dotmul(delta, netj_rhoP, delta);
          fused_sub_dotmul(layerOutput, expected, netj_rhoP, delta);
        }

        // inner layers
        else
        {
          Matrix &lastDelta = deltas[layer + 1];
          Matrix &lastWeights = this->layers[layer + 1].weights;
          // Matrix::mulTrans(lastWeights, lastDelta, delta);
          // delta.dotmul(netj_rhoP);
          fused_mulT_dotmul(lastWeights, lastDelta, netj_rhoP, delta);
        }

        // compute gradient
        Matrix &gradient = gradients[layer];
        Matrix nextlayerOutput = layerInputs[layer - 1];
        compGradKern<<<SM_COUNT, SM_THREADS>>>(delta.mat.get(), nextlayerOutput.mat.get(), gradient.mat.get(), gradient.dim1, gradient.dim2, config.batchSize);

        // compute bias gradient
        compBiasGradKern<<<SM_COUNT, SM_THREADS>>>(delta.mat.get(), biasGradients[layer].vec.get(), biasGradients[layer].dim, config.batchSize);
        cudaDeviceSynchronize();
      }

      // learning rate / numData (to simplify eta * sum / numData * -1)
      const double adjScalarWeight = config.eta_weight / config.batchSize * -1;
      const double adjScalarBias = config.eta_bias / config.batchSize * -1;

      // compute update factors and update weights and biases
      for (uint32_t i = 1; i < this->numLayers; i++)
      {
        Vector &biasGrad = biasGradients[i];
        Matrix &grad = gradients[i];
        Matrix &lastGrad = lastGradients[i];

        // momentum
        lastGrad.mul(config.momentum);
        this->layers[i].weights.add(lastGrad);

        grad.mul(adjScalarWeight);

        // copy gradient for next iteration
        Matrix::copy(grad, lastGrad);

        this->layers[i].weights.add(grad);

        biasGrad.mul(adjScalarBias);
        this->layers[i].biases.add(biasGrad);
      }
    }
  }

  std::cout << std::endl;
}