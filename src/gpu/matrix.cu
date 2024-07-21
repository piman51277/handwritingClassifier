#include "matrix.h"
#include "constants.h"

__global__ void fillKern(double *__restrict__ ptr, uint32_t dim, double fill)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim; i += stride)
  {
    ptr[i] = fill;
  }
}

Matrix Matrix::create(uint32_t dim1, uint32_t dim2, double fill)
{
  Matrix m;
  m.dim1 = dim1;
  m.dim2 = dim2;
  double *tmp;
  cudaMalloc(&tmp, sizeof(double) * dim1 * dim2);
  fillKern<<<SM_COUNT, SM_THREADS>>>(tmp, dim1 * dim2, fill);
  cudaDeviceSynchronize();
  cuda_ptr ptr(tmp, CudaDeleter());
  m.mat = std::move(ptr);
  return m;
}

Matrix Matrix::create(uint32_t dim1, uint32_t dim2)
{
  Matrix m;
  m.dim1 = dim1;
  m.dim2 = dim2;
  double *tmp;
  cudaMalloc(&tmp, sizeof(double) * dim1 * dim2);
  cuda_ptr ptr(tmp, CudaDeleter());
  m.mat = std::move(ptr);
  return m;
}

__global__ void matColRippleKern(double *__restrict__ mat, double *__restrict__ vec, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    mat[i] = vec[i / dim2];
  }
}

__global__ void matRowRippleKern(double *__restrict__ mat, double *__restrict__ vec, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    mat[i] = vec[i % dim2];
  }
}

Matrix Matrix::create(Vector &v, uint32_t cols)
{
  Matrix m;
  m.dim1 = v.dim;
  m.dim2 = cols;
  double *tmp;
  cudaMalloc(&tmp, sizeof(double) * m.dim1 * m.dim2);
  matColRippleKern<<<SM_COUNT, SM_THREADS>>>(tmp, v.vec.get(), m.dim1, m.dim2);
  cudaDeviceSynchronize();
  cuda_ptr ptr(tmp, CudaDeleter());
  m.mat = std::move(ptr);
  return m;
}

Matrix Matrix::create(uint32_t rows, Vector &v)
{
  Matrix m;
  m.dim1 = rows;
  m.dim2 = v.dim;
  double *tmp;
  cudaMalloc(&tmp, sizeof(double) * m.dim1 * m.dim2);
  matRowRippleKern<<<SM_COUNT, SM_THREADS>>>(tmp, v.vec.get(), m.dim1, m.dim2);
  cudaDeviceSynchronize();
  cuda_ptr ptr(tmp, CudaDeleter());
  m.mat = std::move(ptr);
  return m;
}

void Matrix::copy(Matrix &m, Matrix &res)
{
  if (res.dim1 != m.dim1 || res.dim2 != m.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  cudaMemcpy(res.mat.get(), m.mat.get(), sizeof(double) * m.dim1 * m.dim2, cudaMemcpyDeviceToDevice);
}

Matrix Matrix::copy(Matrix &m)
{
  Matrix co = create(m.dim1, m.dim2);
  copy(m, co);
  return co;
}

Matrix Matrix::read(const char *filename)
{
  std::ifstream file(filename, std::ios_base::binary);
  Matrix matrix;
  file.read(reinterpret_cast<char *>(&matrix.dim1), sizeof(matrix.dim1));
  file.read(reinterpret_cast<char *>(&matrix.dim2), sizeof(matrix.dim2));
  double *tmp;
  cudaMalloc(&tmp, sizeof(double) * matrix.dim1 * matrix.dim2);
  double *mat = new double[matrix.dim1 * matrix.dim2];
  file.read(reinterpret_cast<char *>(mat), sizeof(double) * matrix.dim1 * matrix.dim2);
  cudaMemcpy(tmp, mat, sizeof(double) * matrix.dim1 * matrix.dim2, cudaMemcpyHostToDevice);
  file.close();
  cuda_ptr ptr(tmp, CudaDeleter());
  matrix.mat = std::move(ptr);
  delete[] mat;
  return matrix;
}

void Matrix::write(const char *filename, Matrix &matrix)
{
  std::ofstream file(filename);
  file.write(reinterpret_cast<char *>(&matrix.dim1), sizeof(matrix.dim1));
  file.write(reinterpret_cast<char *>(&matrix.dim2), sizeof(matrix.dim2));
  double *mat = new double[matrix.dim1 * matrix.dim2];
  cudaMemcpy(mat, matrix.mat.get(), sizeof(double) * matrix.dim1 * matrix.dim2, cudaMemcpyDeviceToHost);
  file.write(reinterpret_cast<char *>(mat), sizeof(double) * matrix.dim1 * matrix.dim2);
  file.close();
  delete[] mat;
}

void Matrix::write(const char *filename)
{
  write(filename, *this);
}

void Matrix::print()
{
  double *m;
  cudaMallocHost(&m, sizeof(double) * dim1 * dim2);
  cudaMemcpy(m, this->mat.get(), sizeof(double) * dim1 * dim2, cudaMemcpyDeviceToHost);

  for (int i = 0; i < dim1; i++)
  {
    for (int j = 0; j < dim2; j++)
    {
      std::cout << m[i * dim2 + j] << " ";
    }
    std::cout << std::endl;
  }
  cudaFree(m);
}

__global__ void matrixAddKern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    C[i] = A[i] + B[i];
  }
}

void Matrix::add(Matrix &A, Matrix &B, Matrix &res)
{
  if (A.dim1 != B.dim1 || A.dim2 != B.dim2 || A.dim1 != res.dim1 || A.dim2 != res.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixAddKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), res.mat.get(), A.dim1, A.dim2);
  cudaDeviceSynchronize();
}

Matrix Matrix::add(Matrix &A, Matrix &B)
{
  Matrix res = create(A.dim1, A.dim2);
  add(A, B, res);
  return res;
}

void Matrix::add(Matrix &A)
{
  add(*this, A, *this);
}

__global__ void matrixAddColwiseKern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    C[i] = A[i] + B[i / dim2];
  }
}

void Matrix::addColwise(Matrix &A, Vector &B, Matrix &res)
{
  if (A.dim1 != res.dim1 || A.dim2 != res.dim2 || B.dim != res.dim1)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixAddColwiseKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.vec.get(), res.mat.get(), A.dim1, A.dim2);
  cudaDeviceSynchronize();
}

Matrix Matrix::addColwise(Matrix &A, Vector &B)
{
  Matrix res = create(A.dim1, A.dim2);
  addColwise(A, B, res);
  return res;
}

void Matrix::addColwise(Vector &A)
{
  addColwise(*this, A, *this);
}

__global__ void matrixSubKern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = index; i < dim1 * dim2; i += stride)
  {
    C[i] = A[i] - B[i];
  }
}

void Matrix::sub(Matrix &A, Matrix &B, Matrix &res)
{
  if (A.dim1 != B.dim1 || A.dim2 != B.dim2 || A.dim1 != res.dim1 || A.dim2 != res.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixSubKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), res.mat.get(), A.dim1, A.dim2);
  cudaDeviceSynchronize();
}

Matrix Matrix::sub(Matrix &A, Matrix &B)
{
  Matrix res = create(A.dim1, A.dim2);
  sub(A, B, res);
  return res;
}

void Matrix::sub(Matrix &A)
{
  sub(*this, A, *this);
}

__global__ void matrixMulKern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, uint32_t dim1, uint32_t dim2, uint32_t dim3)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < dim1 * dim3; i += stride)
  {
    const int row = i / dim3;
    const int col = i % dim3;
    double sum = 0;
    for (int j = 0; j < dim2; j++)
    {
      sum += A[row * dim2 + j] * B[j * dim3 + col];
    }
    C[i] = sum;
  }
}

void Matrix::mul(Matrix &A, Matrix &B, Matrix &res)
{
  if (A.dim2 != B.dim1 || A.dim1 != res.dim1 || B.dim2 != res.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixMulKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), res.mat.get(), A.dim1, A.dim2, B.dim2);
  cudaDeviceSynchronize();
}

__global__ void matrixMulVecKern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < dim1; i += stride)
  {
    double sum = 0;
    for (int j = 0; j < dim2; j++)
    {
      sum += A[i * dim2 + j] * B[j];
    }
    C[i] = sum;
  }
}

void Matrix::mul(Matrix &A, Vector &B, Vector &res)
{
  if (A.dim2 != B.dim || A.dim1 != res.dim)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixMulVecKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.vec.get(), res.vec.get(), A.dim1, A.dim2);
  cudaDeviceSynchronize();
}

__global__ void matrixMulScalarKern(double *__restrict__ A, double scalar, double *__restrict__ C, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < dim1 * dim2; i += stride)
  {
    C[i] = A[i] * scalar;
  }
}

void Matrix::mul(Matrix &A, double scalar, Matrix &res)
{
  if (A.dim1 != res.dim1 || A.dim2 != res.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixMulScalarKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), scalar, res.mat.get(), A.dim1, A.dim2);
  cudaDeviceSynchronize();
}

Matrix Matrix::mul(Matrix &A, Matrix &B)
{
  Matrix res = create(A.dim1, B.dim2);
  mul(A, B, res);
  return res;
}

Matrix Matrix::mul(Matrix &A, double scalar)
{
  Matrix res = create(A.dim1, A.dim2);
  mul(A, scalar, res);
  return res;
}

Vector Matrix::mul(Matrix &A, Vector &B)
{
  Vector res = Vector::create(A.dim1);
  mul(A, B, res);
  return res;
}

void Matrix::mul(double scalar)
{
  mul(*this, scalar, *this);
}

// A is dim2 x dim1, B is dim2 x dim3 and C is dim1 x dim3
// 1 3 4
__global__ void matrixMulTransKern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, uint32_t dim1, uint32_t dim2, uint32_t dim3)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < dim1 * dim3; i += stride)
  {
    const int row = i / dim3;
    const int col = i % dim3;
    double sum = 0;
    for (int j = 0; j < dim2; j++)
    {
      sum += A[j * dim1 + row] * B[j * dim3 + col];
    }
    C[i] = sum;
  }
}

void Matrix::mulTrans(Matrix &A, Matrix &B, Matrix &res)
{
  if (A.dim1 != B.dim1 || A.dim2 != res.dim1 || B.dim2 != res.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixMulTransKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), res.mat.get(), A.dim2, A.dim1, B.dim2);
  cudaDeviceSynchronize();
}

Matrix Matrix::mulTrans(Matrix &A, Matrix &B)
{
  Matrix res = create(A.dim2, B.dim2);
  mulTrans(A, B, res);
  return res;
}

__global__ void matrixDotmulKern(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, uint32_t dim1, uint32_t dim2)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < dim1 * dim2; i += stride)
  {
    C[i] = A[i] * B[i];
  }
}

void Matrix::dotmul(Matrix &A, Matrix &B, Matrix &res)
{
  if (A.dim1 != B.dim1 || A.dim2 != B.dim2 || A.dim1 != res.dim1 || A.dim2 != res.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  matrixDotmulKern<<<SM_COUNT, SM_THREADS>>>(A.mat.get(), B.mat.get(), res.mat.get(), A.dim1, A.dim2);
  cudaDeviceSynchronize();
}

Matrix Matrix::dotmul(Matrix &A, Matrix &B)
{
  Matrix res = create(A.dim1, A.dim2);
  dotmul(A, B, res);
  return res;
}

void Matrix::dotmul(Matrix &A)
{
  dotmul(*this, A, *this);
}

Vector Vector::create(uint32_t dim, double fill)
{
  Vector v;
  v.dim = dim;

  double *tmp;
  cudaMalloc(&tmp, sizeof(double) * dim);
  fillKern<<<SM_COUNT, SM_THREADS>>>(tmp, dim, fill);
  cudaDeviceSynchronize();
  cuda_ptr ptr(tmp, CudaDeleter());
  v.vec = std::move(ptr);
  return v;
}

Vector Vector::create(uint32_t dim)
{
  Vector v;
  v.dim = dim;
  double *tmp;
  cudaMalloc(&tmp, sizeof(double) * dim);
  cuda_ptr ptr(tmp, CudaDeleter());
  v.vec = std::move(ptr);
  return v;
}

Vector Vector::copy(Vector &v)
{
  Vector copy = create(v.dim);
  cudaMemcpy(copy.vec.get(), v.vec.get(), sizeof(double) * v.dim, cudaMemcpyDeviceToDevice);
  return copy;
}

void Vector::print()
{
  double *vec;
  cudaMallocHost(&vec, sizeof(double) * dim);
  cudaMemcpy(vec, this->vec.get(), sizeof(double) * dim, cudaMemcpyDeviceToHost);
  for (int i = 0; i < dim; i++)
  {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
  cudaFree(vec);
}

void Vector::add(Vector &A, Vector &B, Vector &res)
{
  matrixAddKern<<<SM_COUNT, SM_THREADS>>>(A.vec.get(), B.vec.get(), res.vec.get(), A.dim, 1);
  cudaDeviceSynchronize();
}

Vector Vector::add(Vector &A, Vector &B)
{
  Vector res = Vector::create(A.dim);
  add(A, B, res);
  return res;
}

void Vector::add(Vector &A)
{
  add(*this, A, *this);
}

void Vector::sub(Vector &A, Vector &B, Vector &res)
{
  matrixSubKern<<<SM_COUNT, SM_THREADS>>>(A.vec.get(), B.vec.get(), res.vec.get(), A.dim, 1);
  cudaDeviceSynchronize();
}

Vector Vector::sub(Vector &A, Vector &B)
{
  Vector res = Vector::create(A.dim);
  sub(A, B, res);
  return res;
}

void Vector::sub(Vector &A)
{
  sub(*this, A, *this);
}

void Vector::mul(Vector &A, double scalar, Vector &res)
{
  matrixMulScalarKern<<<SM_COUNT, SM_THREADS>>>(A.vec.get(), scalar, res.vec.get(), A.dim, 1);
  cudaDeviceSynchronize();
}

Vector Vector::mul(Vector &A, double scalar)
{
  Vector res = Vector::create(A.dim);
  mul(A, scalar, res);
  return res;
}

void Vector::mul(double scalar)
{
  mul(*this, scalar, *this);
}