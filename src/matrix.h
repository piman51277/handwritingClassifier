#pragma once
#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <random>

struct Vector
{
  uint32_t dim;
  std::unique_ptr<double> vec;
};

struct Matrix
{
  uint32_t dim1;
  uint32_t dim2;
  std::unique_ptr<double> mat;
};

struct MatrixPair
{
  uint32_t dim1;
  uint32_t dim2;
  uint32_t dim3;
  std::unique_ptr<double> mat1;
  std::unique_ptr<double> mat2;
};

Matrix createMatrix(uint32_t dim1, uint32_t dim2);

Matrix readMatrix(const char *filename);
void writeMatrix(const char *filename, Matrix &matrix);

void printMatrix(Matrix &m);

void mulMatrix(Matrix &A, Matrix &B, Matrix &result);
void mulMatrix(Matrix &A, double scalar, Matrix &result);
Matrix mulMatrix(Matrix &A, Matrix &B);
Matrix mulMatrix(Matrix &A, double scalar);

void addMatrix(Matrix &A, Matrix &B, Matrix &result);
void addMatrixSelf(Matrix &A, Matrix &B);
Matrix addMatrix(Matrix &A, Matrix &B);

void cpyMatrix(Matrix &A, Matrix &result);
void rstMatrix(Matrix &A);

bool matrixEquals(Matrix &A, Matrix &B, double epsilon);

Vector createVector(uint32_t dim);
Vector copyVector(Vector &v);

void printVector(Vector &v);

void mulVector(Vector &v, double scalar, Vector &result);
void mulVector(Matrix &A, Vector &v, Vector &result);
void mulVector(Vector &v, double scalar, Vector &result);
Vector mulVector(Matrix &A, Vector &v);
Vector mulVector(Vector &v, double scalar);
Vector mulVector(Vector &v, double scalar);

void addVector(Vector &v1, Vector &v2, Vector &result);
void addVectorSelf(Vector &v1, Vector &v2);
Vector addVector(Vector &v1, Vector &v2);

void cpyVector(Vector &v, Vector &result);
void rstVector(Vector &v);

bool vectorEquals(Vector &v1, Vector &v2, double epsilon);
