#include "vector.h"
#include "matrix.h"

Matrix Matrix::create(uint32_t dim1, uint32_t dim2)
{
  Matrix matrix;
  matrix.dim1 = dim1;
  matrix.dim2 = dim2;
  matrix.mat.reset(new double[dim1 * dim2]);
  double *mat = matrix.mat.get();
  std::fill(mat, mat + dim1 * dim2, 0);
  return matrix;
}

void Matrix::print()
{
  double *mat = this->mat.get();
  for (uint32_t i = 0; i < dim1; i++)
  {
    for (uint32_t j = 0; j < dim2; j++)
    {
      std::cout << mat[i * dim2 + j] << " ";
    }
    std::cout << std::endl;
  }
}

Matrix Matrix::readFile(const char *filename)
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

void Matrix::writeFile(const char *filename, Matrix &matrix)
{
  std::ofstream file(filename);
  file.write(reinterpret_cast<char *>(&matrix.dim1), sizeof(matrix.dim1));
  file.write(reinterpret_cast<char *>(&matrix.dim2), sizeof(matrix.dim2));
  file.write(reinterpret_cast<char *>(matrix.mat.get()), sizeof(double) * matrix.dim1 * matrix.dim2);
  file.close();
}

void Matrix::writeFile(const char *filename)
{
  writeFile(filename, *this);
}

void Matrix::copy(Matrix &m1, Matrix &m2)
{
  if (m1.dim1 != m2.dim1 || m1.dim2 != m2.dim2)
  {
    throw std::invalid_argument("Matrix dimensions do not match");
  }

  std::copy(m1.mat.get(), m1.mat.get() + m1.dim1 * m1.dim2, m2.mat.get());
}

Matrix Matrix::copy(Matrix &m)
{
  Matrix result = create(m.dim1, m.dim2);
  std::copy(m.mat.get(), m.mat.get() + m.dim1 * m.dim2, result.mat.get());
  return result;
}

void Matrix::reset(Matrix &m)
{
  std::fill(m.mat.get(), m.mat.get() + m.dim1 * m.dim2, 0);
}

void Matrix::reset()
{
  reset(*this);
}

void Matrix::mul(Matrix &A, Matrix &B, Matrix &result)
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

void Matrix::mul(Matrix &A, double scalar, Matrix &result)
{
  double *mat = result.mat.get();
  for (uint32_t i = 0; i < A.dim1 * A.dim2; i++)
  {
    mat[i] = A.mat.get()[i] * scalar;
  }
}

void Matrix::mul(Matrix &A, Vector &v, Vector &result)
{
  return Vector::mul(A, v, result);
}

void Matrix::mul(Vector &v, Matrix &A, Vector &result)
{
  return Vector::mul(v, A, result);
}

Matrix Matrix::mul(Matrix &A, Matrix &B)
{
  Matrix result = create(A.dim1, B.dim2);
  mul(A, B, result);
  return result;
}

Matrix Matrix::mul(Matrix &A, double scalar)
{
  Matrix result = create(A.dim1, A.dim2);
  mul(A, scalar, result);
  return result;
}

Vector Matrix::mul(Matrix &A, Vector &v)
{
  Vector result = Vector::create(A.dim1);
  mul(A, v, result);
  return result;
}

Vector Matrix::mul(Vector &v, Matrix &A)
{
  Vector result = Vector::create(A.dim2);
  mul(v, A, result);
  return result;
}

void Matrix::mul(Matrix &A)
{
  Matrix result = create(dim1, A.dim2);
  mul(*this, A, result);
  this->dim1 = result.dim1;
  this->dim2 = result.dim2;
  this->mat = std::move(result.mat);
}

void Matrix::mul(double scalar)
{
  mul(*this, scalar, *this);
}

void Matrix::add(Matrix &A, Matrix &B, Matrix &result)
{
  if (A.dim1 != B.dim1 || A.dim2 != B.dim2)
  {
    throw std::invalid_argument("Matrix dimensions do not match");
  }

  double *mat = result.mat.get();
  for (uint32_t i = 0; i < A.dim1 * A.dim2; i++)
  {
    mat[i] = A.mat.get()[i] + B.mat.get()[i];
  }
}

Matrix Matrix::add(Matrix &A, Matrix &B)
{
  Matrix result = create(A.dim1, A.dim2);
  add(A, B, result);
  return result;
}

void Matrix::add(Matrix &A)
{
  add(*this, A, *this);
}

bool Matrix::equals(Matrix &A, Matrix &B, double epsilon)
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

bool Matrix::equals(Matrix &A, double epsilon)
{
  return equals(*this, A, epsilon);
}