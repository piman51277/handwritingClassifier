#include "matrix.h"
#include "net.h"
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

TrainingDataSet createDataSet(double **inputs, int inputSize, double **outputs, int outputSize, int numData)
{
  TrainingDataSet dataSet;
  dataSet.numData = numData;
  dataSet.data = new TrainingData[numData];

  for (int i = 0; i < numData; i++)
  {
    dataSet.data[i].input = createVector(inputSize);
    dataSet.data[i].output = createVector(outputSize);

    for (int j = 0; j < inputSize; j++)
    {
      dataSet.data[i].input.vec.get()[j] = inputs[i][j];
    }

    for (int j = 0; j < outputSize; j++)
    {
      dataSet.data[i].output.vec.get()[j] = outputs[i][j];
    }
  }

  return dataSet;
}

int main_alt()
{
  // XOR test
  double **inputs = new double *[4];
  double **outputs = new double *[4];
  inputs[0] = new double[2]{0, 0};
  inputs[1] = new double[2]{0, 1};
  inputs[2] = new double[2]{1, 0};
  inputs[3] = new double[2]{1, 1};
  outputs[0] = new double[1]{0};
  outputs[1] = new double[1]{1};
  outputs[2] = new double[1]{1};
  outputs[3] = new double[1]{0};

  TrainingDataSet dataSet = createDataSet(inputs, 2, outputs, 1, 4);
  delete[] inputs;
  delete[] outputs;

  // use a 2 3 1 network
  uint32_t layerSizes[4] = {2, 3, 3, 1};

  Net net(3, layerSizes);
  net.initializeWeights();

  double error = net.error(dataSet);

  std::cout << "Initial error: " << error << std::endl;

  TrainConfig config{100000, 2, 0.2, 0.2, 0.1};

  auto start = Clock::now();

  net.train(dataSet, config);

  auto end = Clock::now();

  std::cout << "Training took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << "ms" << std::endl;

  error = net.error(dataSet);

  std::cout << "Final error: " << error << std::endl;

  return 0;
}

int main()
{
  Matrix A = readMatrix("A.bin");
  Matrix B = readMatrix("B.bin");

  auto start = Clock::now();
  Matrix C = mulMatrix(A, B);
  auto end = Clock::now();

  std::cout << "Multiplication took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << "ms" << std::endl;

  writeMatrix("C.bin", C);
}