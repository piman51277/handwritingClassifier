#include "vector.h"
#include "matrix.h"

Vector Vector::create(uint32_t dim)
{
  Vector vector;
  vector.dim = dim;
  vector.vec.reset(new double[dim]);
  std::fill(vector.vec.get(), vector.vec.get() + dim, 0);
  return vector;
}

void Vector::print()
{

  for (uint32_t i = 0; i < dim; i++)
  {
    std::cout << vec.get()[i] << " ";
  }
  std::cout << std::endl;
}

void Vector::copy(Vector &v1, Vector &v2)
{
  if (v1.dim != v2.dim)
  {
    throw std::invalid_argument("Vector dimensions do not match");
  }

  std::copy(v1.vec.get(), v1.vec.get() + v1.dim, v2.vec.get());
}

Vector Vector::copy(Vector &v)
{
  Vector result = create(v.dim);
  copy(v, result);
  return result;
}

void Vector::reset(Vector &v)
{
  std::fill(v.vec.get(), v.vec.get() + v.dim, 0);
}

void Vector::reset()
{
  Vector::reset(*this);
}

void Vector::mul(Vector &v, double scalar, Vector &result)
{
  for (uint32_t i = 0; i < v.dim; i++)
  {
    result.vec.get()[i] = v.vec.get()[i] * scalar;
  }
}

void Vector::mul(Matrix &A, Vector &v, Vector &result)
{
  if (A.dim2 != v.dim)
  {
    throw std::invalid_argument("Matrix and vector dimensions do not match");
  }

  for (uint32_t i = 0; i < A.dim1; i++)
  {
    result.vec.get()[i] = 0;
    for (uint32_t j = 0; j < A.dim2; j++)
    {
      result.vec.get()[i] += A.mat.get()[i * A.dim2 + j] * v.vec.get()[j];
    }
  }
}

void Vector::mul(Vector &v, Matrix &A, Vector &result)
{
  if (A.dim1 != v.dim)
  {
    throw std::invalid_argument("Matrix and vector dimensions do not match");
  }

  for (uint32_t i = 0; i < A.dim2; i++)
  {
    result.vec.get()[i] = 0;
    for (uint32_t j = 0; j < A.dim1; j++)
    {
      result.vec.get()[i] += A.mat.get()[j * A.dim2 + i] * v.vec.get()[j];
    }
  }
}

Vector Vector::mul(Matrix &A, Vector &v)
{
  Vector result = create(A.dim1);
  mul(A, v, result);
  return result;
}

Vector Vector::mul(Vector &v, double scalar)
{
  Vector result = create(v.dim);
  mul(v, scalar, result);
  return result;
}

Vector Vector::mul(Vector &v, Matrix &A)
{
  Vector result = create(A.dim2);
  mul(v, A, result);
  return result;
}

void Vector::mul(double scalar)
{
  mul(*this, scalar, *this);
}

void Vector::mul(Matrix &A)
{
  mul(*this, A, *this);
}

void Vector::add(Vector &v1, Vector &v2, Vector &result)
{
  if (v1.dim != v2.dim)
  {
    throw std::invalid_argument("Vector dimensions do not match");
  }

  for (uint32_t i = 0; i < v1.dim; i++)
  {
    result.vec.get()[i] = v1.vec.get()[i] + v2.vec.get()[i];
  }
}

Vector Vector::add(Vector &v1, Vector &v2)
{
  Vector result = create(v1.dim);
  add(v1, v2, result);
  return result;
}

void Vector::add(Vector &v)
{
  add(*this, v, *this);
}

bool Vector::equals(Vector &v1, Vector &v2, double epsilon)
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

bool Vector::equals(Vector &v, double epsilon)
{
  return equals(*this, v, epsilon);
}