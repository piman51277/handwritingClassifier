#pragma once
#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <random>
class Matrix;

class Vector
{
public:
  uint32_t dim;
  std::unique_ptr<double> vec;

  static Vector create(uint32_t dim);

  void print();

  static void copy(Vector &v1, Vector &v2);
  static Vector copy(Vector &v);

  static void reset(Vector &v);
  void reset();

  static void mul(Vector &v, double scalar, Vector &result);
  static void mul(Matrix &A, Vector &v, Vector &result);
  static void mul(Vector &v, Matrix &A, Vector &result);
  static Vector mul(Matrix &A, Vector &v);
  static Vector mul(Vector &v, double scalar);
  static Vector mul(Vector &v, Matrix &A);

  void mul(double scalar);
  void mul(Matrix &A); // performs vA. See Matrix.mul for Av

  static void add(Vector &v1, Vector &v2, Vector &result);
  static Vector add(Vector &v1, Vector &v2);
  void add(Vector &v);

  static bool equals(Vector &v1, Vector &v2, double epsilon);
  bool equals(Vector &v, double epsilon);
};