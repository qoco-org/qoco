/**
 * @file qoco_linalg.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides various linear algebra operations.
 */

#ifndef QOCO_LINALG_H
#define QOCO_LINALG_H
#include "definitions.h"

typedef struct QOCOMatrix_ QOCOMatrix;
typedef struct QOCOVectorf_ QOCOVectorf;
typedef struct QOCOVectori_ QOCOVectori;

/**
 * @brief Compressed sparse column format matrices.
 *
 */
typedef struct {
  /** Number of rows. */
  QOCOInt m;

  /** Number of columns. */
  QOCOInt n;

  /** Number of nonzero elements. */
  QOCOInt nnz;

  /** Row indices (length: nnz). */
  QOCOInt* i;

  /** Column pointers (length: n+1). */
  QOCOInt* p;

  /** Data (length: nnz). */
  QOCOFloat* x;

} QOCOCscMatrix;

/**
 * @brief Allocates a new csc matrix and copies A to it.
 *
 * @param A Matrix to copy.
 * @return Pointer to new constructed matrix.
 */
QOCOCscMatrix* new_qoco_csc_matrix(const QOCOCscMatrix* A);

/**
 * @brief Frees all the internal arrays and the pointer to the QOCOCscMatrix.
 * Should only be used if QOCOCscMatrix and all internal arrays were malloc'ed.
 *
 * @param A Pointer to QOCOCscMatrix.
 */
void free_qoco_csc_matrix(QOCOCscMatrix* A);

/**
 * @brief Copies array of QOCOFloats from x to array y.
 *
 * @param x Source array.
 * @param y Destination array.
 * @param n Length of arrays.
 */
void copy_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOInt n);

/**
 * @brief Copies and negates array of QOCOFloats from x to array y.
 *
 * @param x Source array.
 * @param y Destination array.
 * @param n Length of arrays.
 */
void copy_and_negate_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOInt n);

/**
 * @brief Copies array of QOCOInts from x to array y.
 *
 * @param x Source array.
 * @param y Destination array.
 * @param n Length of arrays.
 */
void copy_arrayi(const QOCOInt* x, QOCOInt* y, QOCOInt n);

/**
 * @brief Computes dot product of u and v.
 *
 * @param u Input vector.
 * @param v Input vector.
 * @param n Length of vectors.
 * @return Dot product of u and v.
 */
QOCOFloat qoco_dot(const QOCOFloat* u, const QOCOFloat* v, QOCOInt n);

/**
 * @brief Computes maximum element of array of QOCOInts.
 *
 * @param x Input array.
 * @param n Length of array.
 * @return Maximum element of x.
 */
QOCOInt max_arrayi(const QOCOInt* x, QOCOInt n);

/**
 * @brief Scales array x by s and stores result in y.
 * y = s * x
 *
 * @param x Input array.
 * @param y Output array.
 * @param s Scaling factor.
 * @param n Length of arrays.
 */
void scale_arrayf(const QOCOFloat* x, QOCOFloat* y, QOCOFloat s, QOCOInt n);

/**
 * @brief Computes z = a * x + y.
 *
 * @param x Input vector.
 * @param y Input vector.
 * @param z Result vector.
 * @param a Scaling factor.
 * @param n Length of vectors.
 */
void qoco_axpy(const QOCOFloat* x, const QOCOFloat* y, QOCOFloat* z,
               QOCOFloat a, QOCOInt n);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices where M is
 * symmetric and only the upper triangular part is given. Computes r = M * v
 *
 * @param M Upper triangular part of M in CSC form.
 * @param v Vector.
 * @param r Result.
 */
void USpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices. Computes r = M *
 * v.
 *
 * @param M Matrix in CSC form.
 * @param v Vector.
 * @param r Result.
 */
void SpMv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices where M is first
 * transposed. Computes r = M^T * v.
 *
 * @param M Matrix in CSC form.
 * @param v Vector.
 * @param r Result.
 */
void SpMtv(const QOCOCscMatrix* M, const QOCOFloat* v, QOCOFloat* r);

/**
 * @brief Computes the infinity norm of x.
 *
 * @param x Input vector.
 * @param n Length of input vector.
 * @return Infinity norm of x.
 */
QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n);

#endif /* #ifndef QOCO_LINALG_H*/
