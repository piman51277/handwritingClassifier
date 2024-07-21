# handwritingClassifier
A from-scratch implementation of a feed-forward neural network and backpropagation used to classify handwritten digits from the MNIST dataset. As of now, the best model trained using this implementation has achieved **96.44%** accuracy using a 5-layer 784-400-400-200-10 configuration.

This includes two implementations:

- An implementation in C++ (`/src/cpu`) used to establish a baseline for correctness
- An implementation in CUDA C++ (`/src/gpu`) used to train the model in a reasonable timeframe

## Installation
1.
```
git clone git@github.com:piman51277/handwritingClassifier.git
cd handwritingClassifier
./compile.sh
```

2. Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and extract the files to `data/` in the project directory.


## Running
- (CPU) `./bin/classifier_baseline`
- (GPU) `./bin/classifier`

## Implementation Details
The mathematical basis for this implementation comes from the [the Wikipedia article on Backpropagation](https://en.wikipedia.org/wiki/Backpropagation#Derivation). Variables are named to roughly correspond to the notation used in the article.

### Activation & Error Functions
This implementation uses sigmoid activation and MSE error functions. 

### Additional Features
- **Momentum**: This implementation uses a momentum factor to help the model converge faster.
- **Batch Training**: This implementation uses batch training to speed up the training process.
- **Bias**: This implementation uses bias nodes in each layer

### Key Differences Between CPU & GPU Implementations
- **Batch Training**: The CPU implementation batches during the training process, while the GPU implementation expects data to be pre-batched.
- **Training Process**: The CPU implementation evalutes only one test case at a time using Vector x Matrix multiplication, while the GPU implementation evaluates an entire batch at once using Matrix x Matrix multiplication.


## Custom Datasets
Although it is currently being used to train against the MNIST dataset, this implementation is general enough to work with any dataset. See `src/gpu/mnist.cu` for an example of how to prepare data for training.

> This section provides instructions on using the **CUDA C++** implementation. The C++ implementation uses a similar but incompatible API.

### Preparing Data
1. Preprocess your data so that each testcase can be represented by a pair of vectors.
2. Pick a batch size. For performance reasons, this model expects data to be pre-batched.
3. For each batch: Create two column-wise matricies, one for the input data and one for the expected output data. Each column in either matrix should represent a single testcase. Thier dimensions should be `input_size x batch_size` and `output_size x batch_size` respectively.
4. Pack each batch into a `TrainingData` object, and pack these into a `TrainingDataSet` object.

### Initializing the Model
You have two options for initializing the model:
1. Load a model from a file
```c++
#include "net.h"
Net net = Net::load("path/to/model");
```
2. Create a new model with random weights & biases
```c++
#include <vector>
#include "net.h"
std::vector<uint32_t> layers = {784, 100, 10};
Net net(layers);
net.initializeWeights();
```

### Training the Model
Training will use a `TrainingDataSet` object as data and a `TrainConfig` object to configure the training process. The  structure of `TrainConfig` is as follows:
```c++
struct TrainConfig
{
  uint32_t numEpochs; // Number of epochs to train for
  uint32_t batchSize; // Number of testcases per batch (this must match your chosen batch size)
  double eta_weight; // Learning rate for weights
  double eta_bias; // Learning rate for biases
  double momentum; // Momentum factor
};
```

For training against the MNIST dataset, the configuration `{1000, 4000, 0.2, 0.2, 0.95}` was used.
However, it is recommended to try different configurations to see what yields the best results in terms of performance and accuracy.

### Checking the Model
You have two options for checking the accruacy of the model:
1. `double Net::error(TrainingDataSet &dataSet)`
    - Returns the Mean Squared Error of the model.
2. `double Net::error_percent(TrainingDataSet &dataSet)`
    - Returns the accuracy of the model. Designed for the classification problems, checks how many cases where the highest output correspond with the expected answer. (Note, this returns the porportion of **correct** classifications)

### Tuning Training Performance
Tweak the parameters set in `src/gpu/constants.h` and `compile.sh` to match the specifications of your GPU. The current settings are optimized for a RTX 3060 12GB.

- `SM_COUNT`: The number of SMs on your GPU
- `BLOCK_SIZE`: The number of threads per block (you have to find an optimal value through trial and error, but a good starting point is 512)

- `compile.sh`: Set the `-gencode=arch=compute_86,code=sm_86` flag to match the compute capability of your GPU. This is set to 8.6 for the RTX 3060.