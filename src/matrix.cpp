#include "matrix.h"

Matrix createMatrix(uint32_t dim1, uint32_t dim2)
{
  Matrix matrix;
  matrix.dim1 = dim1;
  matrix.dim2 = dim2;
  matrix.mat.reset(new double[dim1 * dim2]);
  double *mat = matrix.mat.get();
  std::fill(mat, mat + dim1 * dim2, 0);
  return matrix;
}

Matrix readMatrix(const char *filename)
{
  std::ifstream file(filename, std::ios_base::binary);
  Matrix matrix;
  file.read(reinterpret_cast<char *>(&matrix.dim1), sizeof(matrix.dim1));
  file.read(reinterpret_cast<char *>(&matrix.dim2), sizeof(matrix.dim2));
  matrix.mat.reset(new double[matrix.dim1 * matrix.dim2]);
  file.read(reinterpret_cast<char *>(matrix.mat.get()), sizeof(double) * matrix.dim1 * matrix.dim2);
  file.close();
  return matrix;
}

void writeMatrix(const char *filename, Matrix &matrix)
{
  std::ofstream file(filename);
  file.write(reinterpret_cast<char *>(&matrix.dim1), sizeof(matrix.dim1));
  file.write(reinterpret_cast<char *>(&matrix.dim2), sizeof(matrix.dim2));
  file.write(reinterpret_cast<char *>(matrix.mat.get()), sizeof(double) * matrix.dim1 * matrix.dim2);
  file.close();
}

void printMatrix(Matrix &m)
{
  for (uint32_t i = 0; i < m.dim1; i++)
  {
    for (uint32_t j = 0; j < m.dim2; j++)
    {
      std::cout << m.mat.get()[i * m.dim2 + j] << " ";
    }
    std::cout << std::endl;
  }
}

// variant where memory is externally managed
void mulMatrix(Matrix &A, Matrix &B, Matrix &result)
{
  if (A.dim2 != B.dim1)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  double *cache = new double[A.dim2];
  for (uint32_t x = 0; x < result.dim2; x++)
  {
    for (uint32_t i = 0; i < B.dim1; i++)
    {
      cache[i] = B.mat.get()[i * result.dim2 + x];
    }

    for (uint32_t y = 0; y < A.dim1; y++)
    {
      double sum = 0;
      for (uint32_t j = 0; j < A.dim2; j++)
      {
        sum += A.mat.get()[y * A.dim2 + j] * cache[j];
      }
      result.mat.get()[y * result.dim2 + x] = sum;
    }
  }
}

Matrix mulMatrix(Matrix &A, Matrix &B)
{
  Matrix result = createMatrix(A.dim1, B.dim2);
  mulMatrix(A, B, result);
  return result;
}

// variant where memory is externally managed
void mulMatrix(Matrix &A, double scalar, Matrix &result)
{
  for (uint32_t i = 0; i < A.dim1 * A.dim2; i++)
  {
    result.mat.get()[i] = A.mat.get()[i] * scalar;
  }
}

Matrix mulMatrix(Matrix &A, double scalar)
{
  Matrix result = createMatrix(A.dim1, A.dim2);
  mulMatrix(A, scalar, result);
  return result;
}

void addMatrix(Matrix &A, Matrix &B, Matrix &result)
{
  if (A.dim1 != B.dim1 || A.dim2 != B.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  for (uint32_t i = 0; i < A.dim1 * A.dim2; i++)
  {
    result.mat.get()[i] = A.mat.get()[i] + B.mat.get()[i];
  }
}

Matrix addMatrix(Matrix &A, Matrix &B)
{
  Matrix result = createMatrix(A.dim1, A.dim2);
  addMatrix(A, B, result);
  return result;
}

void addMatrixSelf(Matrix &A, Matrix &B)
{
  addMatrix(A, B, A);
}

void cpyMatrix(Matrix &A, Matrix &result)
{
  if (A.dim1 != result.dim1 || A.dim2 != result.dim2)
  {
    throw std::runtime_error("Matrix dimensions do not match");
  }

  std::copy(A.mat.get(), A.mat.get() + A.dim1 * A.dim2, result.mat.get());
}

void rstMatrix(Matrix &A)
{
  std::fill(A.mat.get(), A.mat.get() + A.dim1 * A.dim2, 0);
}

bool matrixEquals(Matrix &A, Matrix &B, double epsilon)
{
  if (A.dim1 != B.dim1 || A.dim2 != B.dim2)
  {
    return false;
  }

  for (uint32_t i = 0; i < A.dim1 * A.dim2; i++)
  {
    if (std::abs(A.mat.get()[i] - B.mat.get()[i]) > epsilon)
    {
      return false;
    }
  }

  return true;
}

Vector createVector(uint32_t dim)
{
  Vector vector;
  vector.dim = dim;
  vector.vec.reset(new double[dim]);
  std::fill(vector.vec.get(), vector.vec.get() + dim, 0);
  return vector;
}

Vector copyVector(Vector &v)
{
  Vector result = createVector(v.dim);
  std::copy(v.vec.get(), v.vec.get() + v.dim, result.vec.get());
  return result;
}

void printVector(Vector &v)
{
  for (uint32_t i = 0; i < v.dim; i++)
  {
    std::cout << v.vec.get()[i] << " ";
  }
  std::cout << std::endl;
}

Vector mulVector(Matrix &A, Vector &v)
{
  Vector result = createVector(A.dim1);
  mulVector(A, v, result);
  return result;
}

// variant where memory is externally managed
void mulVector(Matrix &A, Vector &v, Vector &result)
{
  if (A.dim2 != v.dim)
  {
    throw std::runtime_error("Matrix and vector dimensions do not match");
  }

  for (uint32_t i = 0; i < A.dim1; i++)
  {
    double sum = 0;
    for (uint32_t j = 0; j < A.dim2; j++)
    {
      sum += A.mat.get()[i * A.dim2 + j] * v.vec.get()[j];
    }
    result.vec.get()[i] = sum;
  }
}

// variant where memory is externally managed
void mulVector(Vector &v, Matrix &A, Vector &result)
{
  if (A.dim1 != v.dim)
  {
    throw std::runtime_error("Matrix and vector dimensions do not match");
  }

  for (uint32_t i = 0; i < A.dim2; i++)
  {
    double sum = 0;
    for (uint32_t j = 0; j < A.dim1; j++)
    {
      sum += v.vec.get()[j] * A.mat.get()[j * A.dim2 + i];
    }
    result.vec.get()[i] = sum;
  }
}

Vector mulVector(Vector &v, Matrix &A)
{
  Vector result = createVector(A.dim2);
  mulVector(v, A, result);
  return result;
}

// variant where memory is externally managed
void mulVector(Vector &v, double scalar, Vector &result)
{
  double *vec = result.vec.get();
  for (uint32_t i = 0; i < v.dim; i++)
  {
    vec[i] = vec[i] * scalar;
  }
}

Vector mulVector(Vector &v, double &scalar)
{
  Vector result = createVector(v.dim);
  mulVector(v, scalar, result);
  return result;
}

void addVector(Vector &v1, Vector &v2, Vector &result)
{
  if (v1.dim != v2.dim)
  {
    throw std::runtime_error("Vector dimensions do not match");
  }

  for (uint32_t i = 0; i < v1.dim; i++)
  {
    result.vec.get()[i] = v1.vec.get()[i] + v2.vec.get()[i];
  }
}

Vector addVector(Vector &v1, Vector &v2)
{
  Vector result = createVector(v1.dim);
  addVector(v1, v2, result);
  return result;
}

void addVectorSelf(Vector &v1, Vector &v2)
{
  addVector(v1, v2, v1);
}

void cpyVector(Vector &v, Vector &result)
{
  if (v.dim != result.dim)
  {
    throw std::runtime_error("Vector dimensions do not match");
  }

  std::copy(v.vec.get(), v.vec.get() + v.dim, result.vec.get());
}

void rstVector(Vector &v)
{
  std::fill(v.vec.get(), v.vec.get() + v.dim, 0);
}

bool vectorEquals(Vector &v1, Vector &v2, double epsilon)
{
  if (v1.dim != v2.dim)
  {
    return false;
  }

  for (uint32_t i = 0; i < v1.dim; i++)
  {
    if (std::abs(v1.vec.get()[i] - v2.vec.get()[i]) > epsilon)
    {
      return false;
    }
  }

  return true;
}