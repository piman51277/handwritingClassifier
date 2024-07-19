#pragma once
#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <random>
class Vector;

class Matrix
{
public:
  uint32_t dim1;
  uint32_t dim2;
  std::unique_ptr<double> mat;

  static Matrix create(uint32_t dim1, uint32_t dim2);

  void print();

  static Matrix readFile(const char *filename);
  static void writeFile(const char *filename, Matrix &matrix);
  void writeFile(const char *filename);

  static void copy(Matrix &m1, Matrix &m2);
  static Matrix copy(Matrix &m);

  static void reset(Matrix &m);
  void reset();

  static void mul(Matrix &A, Matrix &B, Matrix &result);
  static void mul(Matrix &A, double scalar, Matrix &result);
  static void mul(Matrix &A, Vector &v, Vector &result);
  static void mul(Vector &v, Matrix &A, Vector &result);
  static Matrix mul(Matrix &A, Matrix &B);
  static Matrix mul(Matrix &A, double scalar);
  static Vector mul(Matrix &A, Vector &v);
  static Vector mul(Vector &v, Matrix &A);

  void mul(Matrix &A);
  void mul(double scalar);

  static void add(Matrix &A, Matrix &B, Matrix &result);
  static Matrix add(Matrix &A, Matrix &B);
  void add(Matrix &A);

  static bool equals(Matrix &A, Matrix &B, double epsilon);
  bool equals(Matrix &A, double epsilon);
};