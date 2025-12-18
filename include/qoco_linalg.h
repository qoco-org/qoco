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
 * @brief Sets the CPU mode flag (CUDA backend only).
 * During scaling statistics computation, get_data_vectorf returns host pointers.
 * This allows CPU access to vector data for statistics computation.
 *
 * @param active 1 if CPU mode is active, 0 otherwise.
 */
void set_cpu_mode(int active);

/**
 * @brief Allocates a new csc matrix and copies A to it.
 *
 * @param A Matrix to copy.
 * @return Pointer to new constructed matrix.
 */
QOCOCscMatrix* new_qoco_csc_matrix(const QOCOCscMatrix* A);

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
 * @brief Allocates a new QOCOVectorf and copies x to it.
 *
 * @param x vector to copy.
 * @param n length of vector.
 * @return Pointer to new constructed vector.
 */
QOCOVectorf* new_qoco_vectorf(const QOCOFloat* x, QOCOInt n);

/**
 * @brief Allocates a new QOCOVectori and copies x to it.
 *
 * @param x vector to copy.
 * @param n length of vector.
 * @return Pointer to new constructed vector.
 */
QOCOVectori* new_qoco_vectori(const QOCOInt* x, QOCOInt n);

/**
 * @brief Frees QOCOVectorf.
 *
 * @param x Vector to free.
 */
void free_qoco_vectorf(QOCOVectorf* x);

/**
 * @brief Frees QOCOVectori.
 *
 * @param x Vector to free.
 */
void free_qoco_vectori(QOCOVectori* x);


/**
 * @brief Returns the number of nonzeros in a QOCOMatrix.
 *
 * @param A Input matrix.
 * @return Number of nonzeros.
 */
QOCOInt get_nnz(const QOCOMatrix* A);

/**
 * @brief Computes elementwise product z = x .* y
 *
 * @param x Input array.
 * @param y Input array.
 * @param z Output array.
 * @param n Length of arrays.
 */
void ew_product(QOCOFloat* x, const QOCOFloat* y, QOCOFloat* z, QOCOInt n);

/**
 * @brief Returns x[idx].
 *
 * @param x Input vector.
 * @param idx Index.
 * @return x[idx].
 */
QOCOFloat get_element_vectorf(const QOCOVectorf* x, QOCOInt idx);

/**
 * @brief Returns x[idx].
 *
 * @param x Input vector.
 * @param idx Index.
 * @return x[idx].
 */
QOCOInt get_element_vectori(const QOCOVectori* x, QOCOInt idx);

/**
 * @brief Performs x[idx] = data.
 *
 * @param x Input vector.
 * @param idx Index.
 * @param idx data.
 */
void set_element_vectorf(QOCOVectorf* x, QOCOInt idx, QOCOFloat data);

/**
 * @brief Performs output .= 1.0 ./ input.
 *
 * @param input Input vector.
 * @param output Input vector.
 */
void reciprocal_vectorf(const QOCOVectorf* input, QOCOVectorf* output);

/**
 * @brief Returns &x[idx].
 *
 * @param x Input vector.
 * @param idx Index.
 * @return &x[idx].
 */
QOCOFloat* get_pointer_vectorf(const QOCOVectorf* x, QOCOInt idx);

/**
 * @brief Returns the underlying data array of a QOCOVectorf.
 *
 * @param x Input vector.
 * @return Pointer to underlying data array.
 */
QOCOFloat* get_data_vectorf(const QOCOVectorf* x);

/**
 * @brief Returns the underlying data array of a QOCOVectori.
 *
 * @param x Input vector.
 * @return Pointer to underlying data array.
 */
QOCOInt* get_data_vectori(const QOCOVectori* x);

/**
 * @brief Syncs vector data from host to device if needed (CUDA backend only).
 * This is a no-op for non-CUDA backends.
 * Note: This should NOT be called during qoco_solve to avoid CPU-GPU copies.
 *
 * @param v Input vector.
 */
void sync_vector_to_host(QOCOVectorf* v);

/**
 * @brief Syncs vector data from device to host if needed (CUDA backend only).
 * This is a no-op for non-CUDA backends.
 * Note: This should NOT be called during qoco_solve to avoid CPU-GPU copies.
 *
 * @param v Input vector.
 */
void sync_vector_to_device(QOCOVectorf* v);

/**
 * @brief Returns the length of a QOCOVectorf.
 *
 * @param x Input vector.
 * @return Length of vector.
 */
QOCOInt get_length_vectorf(const QOCOVectorf* x);

/**
 * @brief Returns the underlying CSC matrix from a QOCOMatrix.
 * This function provides access to implementation-specific data.
 * For builtin implementation, returns the internal CSC matrix.
 *
 * @param M Input matrix.
 * @return Pointer to underlying CSC matrix.
 */
QOCOCscMatrix* get_csc_matrix(const QOCOMatrix* M);

/**
 * @brief Computes the infinity norm of each column (or equivalently row) of a
 * symmetric sparse matrix M where only the upper triangular portion of M is
 * given. Works with QOCOMatrix*.
 *
 * @param M Upper triangular part of sparse symmetric matrix.
 * @param norm Result vector of length n.
 */
void col_inf_norm_USymm_matrix(const QOCOMatrix* M, QOCOFloat* norm);

/**
 * @brief Computes the infinity norm of each column of M and stores in norm.
 * Works with QOCOMatrix*.
 *
 * @param M An m by n sparse matrix.
 * @param norm Result vector of length n.
 */
void col_inf_norm_matrix(const QOCOMatrix* M, QOCOFloat* norm);

/**
 * @brief Computes the infinity norm of each row of M and stores in norm.
 * Works with QOCOMatrix*.
 *
 * @param M An m by n sparse matrix.
 * @param norm Result vector of length m.
 */
void row_inf_norm_matrix(const QOCOMatrix* M, QOCOFloat* norm);

/**
 * @brief Scales the rows of M by E and columns of M by D.
 * M = diag(E) * M * diag(D). Works with QOCOMatrix*.
 *
 * @param M An m by n sparse matrix.
 * @param E Vector of length m.
 * @param D Vector of length n.
 */
void row_col_scale_matrix(QOCOMatrix* M, const QOCOFloat* E, const QOCOFloat* D);

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
 * @param M Upper triangular part of M.
 * @param v Vector.
 * @param r Result.
 */
void USpMv(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices. Computes r = M *
 * v.
 *
 * @param M Matrix.
 * @param v Vector.
 * @param r Result.
 */
void SpMv(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices where M is first
 * transposed. Computes r = M^T * v.
 *
 * @param M Matrix.
 * @param v Vector.
 * @param r Result.
 */
void SpMtv(const QOCOMatrix* M, const QOCOFloat* v, QOCOFloat* r);

/**
 * @brief Computes the infinity norm of x.
 *
 * @param x Input vector.
 * @param n Length of input vector.
 * @return Infinity norm of x.
 */
QOCOFloat inf_norm(const QOCOFloat* x, QOCOInt n);

/**
 * @brief Computes the minimum absolute value of x.
 *
 * @param x Input vector.
 * @param n Length of input vector.
 * @return Minimum absolute value of x.
 */
QOCOFloat min_abs_val(const QOCOFloat* x, QOCOInt n);

#endif /* #ifndef QOCO_LINALG_H*/
