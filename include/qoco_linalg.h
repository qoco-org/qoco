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
typedef struct QOCOVectori_ QOCOVectorf_;

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
 * @brief Allocates a new QOCOMatrix and copies A to it.
 *
 * @param A Matrix to copy.
 * @return Pointer to new constructed matrix.
 */
QOCOMatrix* new_qoco_matrix(const QOCOCscMatrix* A);

/**
 * @brief Frees QOCOMatrix.
 *
 * @param A Matrix to free.
 */
void free_qoco_matrix(QOCOMatrix* A);

/**
 * @brief Allocates a new QOCOVectorf and copies A to it.
 *
 * @param x vector to copy.
 * @param n length of vector.
 * @return Pointer to new constructed vector.
 */
QOCOVectorf* new_qoco_vectorf(const QOCOFloat* x, QOCOInt n);

/**
 * @brief Frees QOCOVectorf.
 *
 * @param x Vector to free.
 */
void free_qoco_vectorf(QOCOVectorf* x);

/**
 * @brief Allocates a new QOCOMatrix and At to it.
 *
 * @param A Matrix to transpose.
 * @param AtoAt Mapping from indices of A to At.
 * @return Pointer to transposed matrix.
 */
QOCOMatrix* create_transposed_matrix(const QOCOMatrix* A, QOCOInt* AtoAt);

/**
 * @brief Adds reg * I to a QOCOMatrix. Called on P prior to construction
 * of KKT system in qoco_setup().
 *
 * @param num_diagP Number of new element added to diagonal.
 * @param P Matrix to be regularized.
 * @param reg Regularization factor.
 * @param nzadded_idx Indices of elements of M->x that are added.
 * @return P + reg * I.
 */
QOCOMatrix* regularize_P(QOCOInt num_diagP, QOCOMatrix* P, QOCOFloat reg,
                         QOCOInt* nzadded_idx);

/**
 * @brief Subtracts lambda * I to a QOCOMatrix. Called on P when updating
 * matrix data in update_matrix_data(). This function does not allocate and must
 * be called after regularize.
 *
 * @param M Matrix.
 * @param lambda Regularization.
 */
void unregularize(QOCOMatrix* M, QOCOFloat lambda);

/**
 * @brief Allocates a new QOCOMatrix that is lambda * I.
 *
 * @param n Size of identity matrix.
 * @param lambda Scaling factor for identity.
 * @return Pointer to new constructed matrix.
 */
QOCOMatrix* construct_identity(QOCOInt n, QOCOFloat lambda);

/**
 * @brief Computes M = a * M.
 *
 * @param a Scaling factor.
 * @param M QOCOMatrix to scale.
 */
void scale_matrix(QOCOFloat a, QOCOMatrix* M);

/**
 * @brief Scales the rows of M by E and columns of M by D.
 * M = diag(E) * M * diag(S)
 *
 * @param M An m by n QOCOMatrix matrix.
 * @param E Vector of length m.
 * @param D Vector of length m.
 */
void row_col_scale(QOCOMatrix* M, QOCOVectorf* E, QOCOVectorf* D);

/**
 * @brief Updates all the nonzero elements of M to Mnew.
 *
 * @param M Matrix to update.
 * @param Mnew Vector of new data.
 */
void update_matrix(QOCOMatrix* M, QOCOFloat* Mnew);

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
 * @brief Computes elementwise product z = x .* y
 *
 * @param x Input vector.
 * @param y Input vector.
 * @param z Output vector.
 */
void ew_product(QOCOVectorf* x, QOCOVectorf* y, QOCOVectorf* z);

/**
 * @brief Computes elementwise product z = x .* y
 *
 * @param x Input vector.
 * @param y Input array.
 * @param z Output vector.
 */
void ew_product_vec_array(QOCOVectorf* x, QOCOFloat* y, QOCOVectorf* z);

void scale_vectorf(QOCOFloat a, QOCOVectorf* u);

void copy_vectorf(QOCOVectorf* src, QOCOFloat* dest, QOCOInt dest_idx,
                  QOCOInt negate);

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
