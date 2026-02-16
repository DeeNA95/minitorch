#pragma once
#include "matrix.cuh"

namespace minitorch {

void mat_add(const Matrix &A, const Matrix &B, Matrix &C);
void mat_sub(const Matrix &A, const Matrix &B, Matrix &C);
void mat_elem_mul(const Matrix &A, const Matrix &B, Matrix &C);
void mat_scalar_mul(const Matrix &A, float B, Matrix &C); // no doubles for now
Matrix mat_transpose(Matrix &A);
Matrix mat_matmul(const Matrix &A, const Matrix &B);
void b_add(const Matrix &, const Matrix &B, Matrix &C);
} // namespace minitorch
