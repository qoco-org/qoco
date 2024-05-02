/**
 * @file linalg.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides various linear algebra operations.
 */

#ifndef LINALG_H
#define LINALG_H
#include "definitions.h"
#include "structs.h"

/**
 * @brief Allocates a new csc matrix and copies A to it.
 *
 * @param A Matrix to copy.
 * @return Pointer to new constructed matrix.
 */
QCOSCscMatrix* new_qcos_csc_matrix(QCOSCscMatrix* A);

/**
 * @brief Copies array of QCOSFloats from x to array y.
 *
 * @param x Source array.
 * @param y Destination array.
 * @param n Length of arrays.
 */
void copy_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n);

/**
 * @brief Copies and negates array of QCOSFloats from x to array y.
 *
 * @param x Source array.
 * @param y Destination array.
 * @param n Length of arrays.
 */
void copy_and_negate_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n);

/**
 * @brief Copies array of QCOSInts from x to array y.
 *
 * @param x Source array.
 * @param y Destination array.
 * @param n Length of arrays.
 */
void copy_arrayi(const QCOSInt* x, QCOSInt* y, QCOSInt n);

/**
 * @brief Computes dot product of u and v.
 *
 * @param u Input vector.
 * @param v Input vector.
 * @param n Length of vectors.
 * @return Dot product of u and v.
 */
QCOSFloat dot(QCOSFloat* u, QCOSFloat* v, QCOSInt n);

/**
 * @brief Computes maximum element of array of QCOSInts.
 *
 * @param x Input array.
 * @param n Length of array.
 * @return Maximum element of x.
 */
QCOSInt max_arrayi(QCOSInt* x, QCOSInt n);

/**
 * @brief Scales array x by s and stores result in y.
 * y = s * x
 *
 * @param x Input array.
 * @param x Output array.
 * @param s Scaling factor.
 * @param n Length of arrays.
 */
void scale_arrayf(QCOSFloat* x, QCOSFloat* y, QCOSFloat s, QCOSInt n);

/**
 * @brief Computes z = a * x + y.
 *
 * @param x Input vector.
 * @param y Input vector.
 * @param z Result vector.
 * @param a Scaling factor.
 * @param n Length of vectors.
 */
void axpy(QCOSFloat* x, QCOSFloat* y, QCOSFloat* z, QCOSFloat a, QCOSInt n);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices where M is
 * symmetric and only the upper triangular part is given. Computes r = M * v
 *
 * @param M Upper triangular part of M in CSC form.
 * @param v Vector.
 * @param r Result.
 */
void USpMv(QCOSCscMatrix* M, QCOSFloat* v, QCOSFloat* r);

#endif /* #ifndef LINALG_H*/
