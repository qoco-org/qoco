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
#include "structs.h"

/**
 * @brief Allocates a new csc matrix and copies A to it.
 *
 * @param A Matrix to copy.
 * @return Pointer to new constructed matrix.
 */
QOCOCscMatrix* new_qoco_csc_matrix(const QOCOCscMatrix* A);

/**
 * @brief Allocates a new csc matrix that is lambda * I.
 *
 * @param n Size of identity matrix.
 * @param lambda Scaling factor for identity.
 * @return Pointer to new constructed matrix.
 */
QOCOCscMatrix* construct_identity(QOCOInt n, QOCOFloat lambda);

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

/**
 * @brief Adds lambda * I to a CSC matrix. Called on P prior to construction of
 * KKT system in qoco_setup(). This function calls realloc() when adding new
 * nonzeros.
 *
 * @param M Matrix to be regularized.
 * @param lambda Regularization factor.
 * @param nzadded_idx Indices of elements of M->x that are added.
 * @return Number of nonzeros added to M->x.
 */
QOCOInt regularize(QOCOCscMatrix* M, QOCOFloat lambda, QOCOInt* nzadded_idx);

/**
 * @brief Subtracts lambda * I to a CSC matrix. Called on P when updating
 * matrix data in update_matrix_data(). This function does not allocate and must
 * be called after regularize.
 *
 * @param M Matrix.
 * @param lambda Regularization.
 */
void unregularize(QOCOCscMatrix* M, QOCOFloat lambda);

/**
 * @brief Computes the infinity norm of each column (or equivalently row) of a
 * symmetric sparse matrix M where only the upper triangular portion of M is
 * given.
 *
 * @param M Upper triangular part of sparse symmetric matrix.
 * @param norm Result vector of length n.
 */
void col_inf_norm_USymm(const QOCOCscMatrix* M, QOCOFloat* norm);

/**
 * @brief Computes the infinity norm of each row of M and stores in norm.
 *
 * @param M An m by n sparse matrix.
 * @param norm Result vector of length m.
 */
void row_inf_norm(const QOCOCscMatrix* M, QOCOFloat* norm);

/**
 * @brief Allocates and computes A^T.
 *
 * @param A Input matrix.
 * @param AtoAt Mapping from A to At.
 */
QOCOCscMatrix* create_transposed_matrix(const QOCOCscMatrix* A, QOCOInt* AtoAt);

/**
 * @brief Scales the rows of M by E and columns of M by D.
 * M = diag(E) * M * diag(S)
 *
 * @param M An m by n sparse matrix.
 * @param E Vector of length m.
 * @param D Vector of length m.
 */
void row_col_scale(const QOCOCscMatrix* M, QOCOFloat* E, QOCOFloat* D);

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
 * @brief Inverts permutation vector p and stores inverse in pinv.
 *
 * @param p Input permutation vector.
 * @param pinv Inverse of permutation vector.
 * @param n Length of vectors.
 */
void invert_permutation(const QOCOInt* p, QOCOInt* pinv, QOCOInt n);

/**
 * @brief Computes cumulative sum of c.
 * @return Cumulative sum of c.
 */
QOCOInt cumsum(QOCOInt* p, QOCOInt* c, QOCOInt n);

/**
 * @brief C = A(p,p) = PAP' where A and C are symmetric and the upper triangular
 * part is stored.
 *
 * @param A
 * @param pinv
 * @param AtoC
 * @return QOCOCscMatrix*
 */
QOCOCscMatrix* csc_symperm(const QOCOCscMatrix* A, const QOCOInt* pinv,
                           QOCOInt* AtoC);

#endif /* #ifndef QOCO_LINALG_H*/
