#pragma once
#include <cinttypes>
#include "matrix.h"

__global__ void fused_mul_colwise_Kern(double *A, double *B, double *v, double *C, uint32_t dim1, uint32_t dim2, uint32_t dim3);

Matrix fused_mul_colwise(Matrix &A, Matrix &B, Vector &v);

__global__ void fused_sub_dotmul_Kern(double *A, double *B, double *C, double *res, uint32_t dim1, uint32_t dim2);

void fused_sub_dotmul(Matrix &A, Matrix &B, Matrix &C, Matrix &res);

__global__ void fused_mulT_dotmul_Kern(double *A, double *B, double *C, double *res, uint32_t dim1, uint32_t dim2, uint32_t dim3);

void fused_mulT_dotmul(Matrix &A, Matrix &B, Matrix &C, Matrix &res);