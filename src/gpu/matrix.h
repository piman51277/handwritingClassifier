#pragma once
#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <random>
class Vector;

struct CudaDeleter
{
  void operator()(double *ptr)
  {
    cudaFree(ptr);
  }
};

typedef std::shared_ptr<double> cuda_ptr;

class Matrix
{
public:
  uint32_t dim1;
  uint32_t dim2;
  cuda_ptr mat;

  static Matrix create(uint32_t dim1, uint32_t dim2, double fill);
  static Matrix create(uint32_t dim1, uint32_t dim2);
  static Matrix create(Vector &v, uint32_t cols);
  static Matrix create(uint32_t rows, Vector &v);

  static void copy(Matrix &m1, Matrix &m2);
  static Matrix copy(Matrix &m);

  static Matrix read(const char *filename);
  static void write(const char *filename, Matrix &m);
  void write(const char *filename);

  void print();

  static void add(Matrix &m1, Matrix &m2, Matrix &res);
  static Matrix add(Matrix &m1, Matrix &m2);
  void add(Matrix &m);

  static void addColwise(Matrix &m, Vector &v, Matrix &res);
  static Matrix addColwise(Matrix &m, Vector &v);
  void addColwise(Vector &v);

  static void sub(Matrix &m1, Matrix &m2, Matrix &res);
  static Matrix sub(Matrix &m1, Matrix &m2);
  void sub(Matrix &m);

  static void mul(Matrix &m1, Matrix &m2, Matrix &res);
  static void mul(Matrix &m1, Vector &v, Vector &res);
  static void mul(Matrix &m, double scalar, Matrix &res);
  static Matrix mul(Matrix &m1, Matrix &m2);
  static Vector mul(Matrix &m, Vector &v);
  static Matrix mul(Matrix &m, double scalar);
  void mul(double scalar);

  static void mulTrans(Matrix &m1, Matrix &m2, Matrix &res);
  static Matrix mulTrans(Matrix &m1, Matrix &m2);

  static void dotmul(Matrix &m1, Matrix &m2, Matrix &res);
  static Matrix dotmul(Matrix &m1, Matrix &m2);
  void dotmul(Matrix &m);
};

class Vector
{
public:
  uint32_t dim;
  cuda_ptr vec;

  static Vector create(uint32_t dim, double fill);
  static Vector create(uint32_t dim);

  static Vector copy(Vector &v);

  void print();

  static void add(Vector &v1, Vector &v2, Vector &res);
  static Vector add(Vector &v1, Vector &v2);
  void add(Vector &v);

  static void sub(Vector &v1, Vector &v2, Vector &res);
  static Vector sub(Vector &v1, Vector &v2);
  void sub(Vector &v);

  static void mul(Vector &v, double scalar, Vector &res);
  static Vector mul(Vector &v, double scalar);
  void mul(double scalar);
};