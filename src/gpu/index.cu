#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <random>
#include <vector>
#include "matrix.h"
#include "mnist.h"
#include "net.h"

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

void visualizeMNIST(TrainingData &data, uint32_t index)
{
  Matrix &input = data.input;
  Matrix &expected = data.expected;

  // copy data to host
  double *inputData;
  double *expectedData;
  cudaMallocHost(&inputData, input.dim1 * input.dim2 * sizeof(double));
  cudaMallocHost(&expectedData, expected.dim1 * expected.dim2 * sizeof(double));
  cudaMemcpy(inputData, input.mat.get(), input.dim1 * input.dim2 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(expectedData, expected.mat.get(), expected.dim1 * expected.dim2 * sizeof(double), cudaMemcpyDeviceToHost);

  // images are 28x28, and we are looking at the index-th column
  for (uint32_t i = 0; i < 28; i++)
  {
    for (uint32_t j = 0; j < 28; j++)
    {
      if (inputData[index + (i * 28 + j) * input.dim2] > 0.1)
      {
        std::cout << "X";
      }
      else
      {
        std::cout << "-";
      }
    }
    std::cout << std::endl;
  }

  // print expected label
  for (uint32_t i = 0; i < 10; i++)
  {
    if (expectedData[index + i * expected.dim2] > 0.1)
    {
      std::cout << i << std::endl;
    }
  }
}

std::vector<uint32_t> getSettings(const char *filename)
{
  std::ifstream file(filename, std::ios::binary);
  int layers;
  file >> layers;
  std::cout << "Layers: " << layers << std::endl;
  std::vector<uint32_t> settings(layers);
  for (int i = 0; i < layers; i++)
  {
    std::cout << "Layer " << i << ": ";
    file >> settings[i];
    std::cout << settings[i] << std::endl;
  }
  return settings;
}

void train(const char *filename)
{
  cudaDeviceReset();

  std::cout << "Loading MNIST data..." << std::endl;
  TrainingDataSet MNIST = get_MNIST("./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte", 4000);
  TrainingDataSet MNIST_test = get_MNIST("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte", 2000);
  std::cout << "MNIST data loaded." << std::endl;

  // create network
  std::vector<uint32_t> layers = getSettings("config.txt");
  Net net(layers);
  net.initializeWeights();
  std::cout << "Network created." << std::endl;

  // test network
  std::cout << "Testing network..." << std::endl;
  double error = net.error(MNIST_test);
  std::cout << "Initial error (MSE): " << error << std::endl;
  error = net.error_percent(MNIST_test);
  std::cout << "Initial accuracy (percent): " << error << std::endl;

  // train network
  std::cout << "Training network..." << std::endl;
  TrainConfig config{1000, 4000, 0.2, 0.2, 0.95};

  auto t1 = Clock::now();
  net.train(MNIST, config);
  auto t2 = Clock::now();

  std::cout << "Training complete." << std::endl;
  std::cout << "Training time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

  // test network
  std::cout << "Testing network..." << std::endl;
  error = net.error(MNIST_test);
  std::cout << "Final error (MSE): " << error << std::endl;
  error = net.error_percent(MNIST_test);
  std::cout << "Final error (percent): " << error << std::endl;

  // save the net to disk
  net.save(filename);
}

int main()
{
  train("network.bin");

  TrainingDataSet MNIST_test = get_MNIST("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte", 1000);

  // load the net from disk
  std::cout << "Loading saved network..." << std::endl;
  Net net2 = Net::load("network.bin");

  // test network
  std::cout << "Testing network..." << std::endl;
  double error = net2.error(MNIST_test);
  std::cout << "Final error (MSE): " << error << std::endl;
  error = net2.error_percent(MNIST_test);
  std::cout << "Final error (percent): " << error << std::endl;
}