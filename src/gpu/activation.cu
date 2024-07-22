#include "constants.h"
#include "activation.h"

__global__ void activationKernel(double *__restrict__ mat, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    mat[i] = 1 / (1 + exp(-mat[i]));
  }
}

__global__ void _activationPrimeKernel(double *__restrict__ mat, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    double val = mat[i];
    mat[i] = val * (1 - val);
  }
}

void ActivationFunction(Matrix &mat)
{
  activationKernel<<<SM_COUNT, SM_THREADS>>>(mat.mat.get(), mat.dim1, mat.dim2);
  cudaDeviceSynchronize();
}

void ActivationFunction(Vector &vec)
{
  activationKernel<<<SM_COUNT, SM_THREADS>>>(vec.vec.get(), vec.dim, 1);
  cudaDeviceSynchronize();
}

Matrix ActivationFunctionCopy(Matrix &mat)
{
  Matrix copy = Matrix::copy(mat);
  activationKernel<<<SM_COUNT, SM_THREADS>>>(copy.mat.get(), copy.dim1, copy.dim2);
  cudaDeviceSynchronize();
  return copy;
}

Matrix ActivationFunctionPrime(Matrix &mat)
{
  Matrix copy = Matrix::copy(mat);
  _activationPrimeKernel<<<SM_COUNT, SM_THREADS>>>(copy.mat.get(), copy.dim1, copy.dim2);
  cudaDeviceSynchronize();
  return copy;
}

__global__ void MSEKernel(double *__restrict__ output, double *__restrict__ target, double *__restrict__ result, uint32_t dim)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  __shared__ double sdata[SM_THREADS];
  sdata[threadIdx.x] = 0.0;

  for (int i = index; i < dim; i += stride)
  {
    sdata[threadIdx.x] += (output[i] - target[i]) * (output[i] - target[i]);
  }

  __syncthreads();

  // have each block come up with a sum
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    *result = sdata[0];
  }
}

double MSE(Vector &output, Vector &target)
{
  double *result;
  cudaMallocManaged(&result, sizeof(double));
  *result = 0.0;

  MSEKernel<<<SM_COUNT, SM_THREADS>>>(output.vec.get(), target.vec.get(), result, output.dim);
  cudaDeviceSynchronize();

  double res = *result;
  cudaFree(result);
  return res;
}

double MSE(Matrix &output, Matrix &target)
{
  double *result;
  cudaMallocManaged(&result, sizeof(double));
  *result = 0.0;

  MSEKernel<<<SM_COUNT, SM_THREADS>>>(output.mat.get(), target.mat.get(), result, output.dim1 * output.dim2);
  cudaDeviceSynchronize();

  double res = *result;
  cudaFree(result);
  return res;
}