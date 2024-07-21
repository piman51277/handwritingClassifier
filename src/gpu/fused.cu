#include "fused.h"
#include "constants.h"

__global__ void fused_mul_colwise_Kern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ v, double *__restrict__ C, uint32_t dim1, uint32_t dim2, uint32_t dim3)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < dim1 * dim3; i += stride)
  {
    const int row = i / dim3;
    const int col = i % dim3;
    double sum = 0;
#pragma unroll
    for (int j = 0; j < dim2; j++)
    {
      sum += A[row * dim2 + j] * B[j * dim3 + col];
    }
    C[i] = sum + v[row];
  }
}

Matrix fused_mul_colwise(Matrix &A, Matrix &B, Vector &v)
{
  Matrix res = Matrix::create(A.dim1, B.dim2);
  fused_mul_colwise_Kern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), v.vec.get(), res.mat.get(), A.dim1, A.dim2, B.dim2);
  cudaDeviceSynchronize();
  return res;
}

__global__ void fused_sub_dotmul_Kern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, double *__restrict__ res, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    res[i] = (A[i] - B[i]) * C[i];
  }
}

void fused_sub_dotmul(Matrix &A, Matrix &B, Matrix &C, Matrix &res)
{
  fused_sub_dotmul_Kern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), C.mat.get(), res.mat.get(), A.dim1, A.dim2);
  cudaDeviceSynchronize();
}

__global__ void fused_mulT_dotmul_Kern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, double *__restrict__ res, uint32_t dim1, uint32_t dim2, uint32_t dim3)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < dim1 * dim3; i += stride)
  {
    const int row = i / dim3;
    const int col = i % dim3;
    double sum = 0;
#pragma unroll
    for (int j = 0; j < dim2; j++)
    {
      sum += A[j * dim1 + row] * B[j * dim3 + col];
    }
    res[i] = sum * C[i];
  }
}

void fused_mulT_dotmul(Matrix &A, Matrix &B, Matrix &C, Matrix &res)
{
  fused_mulT_dotmul_Kern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), C.mat.get(), res.mat.get(), A.dim2, A.dim1, B.dim2);
  cudaDeviceSynchronize();
}