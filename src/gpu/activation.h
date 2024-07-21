#pragma once
#include "matrix.h"

void ActivationFunction(Matrix &A);
void ActivationFunction(Vector &A);
Matrix ActivationFunctionCopy(Matrix &A);
// shortcut because sigmoid requires recalculation otherwise
Matrix ActivationFunctionPrime(Matrix &Activated);

double MSE(Vector &actual, Vector &expected);
double MSE(Matrix &actual, Matrix &expected);