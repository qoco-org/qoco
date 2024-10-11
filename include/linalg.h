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
QCOSCscMatrix* new_qcos_csc_matrix(const QCOSCscMatrix* A);

/**
 * @brief Allocates a new csc matrix that is lambda * I.
 *
 * @param n Size of identity matrix.
 * @param lambda Scaling factor for identity.
 * @return Pointer to new constructed matrix.
 */
QCOSCscMatrix* construct_identity(QCOSInt n, QCOSFloat lambda);

/**
 * @brief Frees all the internal arrays and the pointer to the QCOSCscMatrix.
 * Should only be used if QCOSCscMatrix and all internal arrays were malloc'ed.
 *
 * @param A Pointer to QCOSCscMatrix.
 */
void free_qcos_csc_matrix(QCOSCscMatrix* A);

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
QCOSFloat dot(const QCOSFloat* u, const QCOSFloat* v, QCOSInt n);

/**
 * @brief Computes maximum element of array of QCOSInts.
 *
 * @param x Input array.
 * @param n Length of array.
 * @return Maximum element of x.
 */
QCOSInt max_arrayi(const QCOSInt* x, QCOSInt n);

/**
 * @brief Scales array x by s and stores result in y.
 * y = s * x
 *
 * @param x Input array.
 * @param y Output array.
 * @param s Scaling factor.
 * @param n Length of arrays.
 */
void scale_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSFloat s, QCOSInt n);

/**
 * @brief Computes z = a * x + y.
 *
 * @param x Input vector.
 * @param y Input vector.
 * @param z Result vector.
 * @param a Scaling factor.
 * @param n Length of vectors.
 */
void axpy(const QCOSFloat* x, const QCOSFloat* y, QCOSFloat* z, QCOSFloat a,
          QCOSInt n);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices where M is
 * symmetric and only the upper triangular part is given. Computes r = M * v
 *
 * @param M Upper triangular part of M in CSC form.
 * @param v Vector.
 * @param r Result.
 */
void USpMv(const QCOSCscMatrix* M, const QCOSFloat* v, QCOSFloat* r);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices. Computes r = M *
 * v.
 *
 * @param M Matrix in CSC form.
 * @param v Vector.
 * @param r Result.
 */
void SpMv(const QCOSCscMatrix* M, const QCOSFloat* v, QCOSFloat* r);

/**
 * @brief Sparse matrix vector multiplication for CSC matrices where M is first
 * transposed. Computes r = M^T * v.
 *
 * @param M Matrix in CSC form.
 * @param v Vector.
 * @param r Result.
 */
void SpMtv(const QCOSCscMatrix* M, const QCOSFloat* v, QCOSFloat* r);

/**
 * @brief Computes the infinity norm of x.
 *
 * @param x Input vector.
 * @param n Length of input vector.
 * @return Infinity norm of x.
 */
QCOSFloat inf_norm(const QCOSFloat* x, QCOSInt n);

/**
 * @brief Adds lambda * I to a CSC matrix. Called on P prior to construction of
 * KKT system in qcos_setup(). This function calls realloc() when adding new
 * nonzeros.
 *
 * @param M Matrix to be regularized.
 * @param lambda Regularization factor.
 * @param nzadded Indices of elements of M->x that are added.
 * @return Number of nonzeros added to M->x.
 */
QCOSInt regularize(QCOSCscMatrix* M, QCOSFloat lambda, QCOSInt* nzadded_idx);

/**
 * @brief Subtracts lambda * I to a CSC matrix. Called on P when updating
 * matrix data in update_matrix_data(). This function does not allocate and must
 * be called after regularize.
 *
 * @param M Matrix.
 * @param lambda Regularization.
 */
void unregularize(QCOSCscMatrix* M, QCOSFloat lambda);

/**
 * @brief Computes the infinity norm of each column (or equivalently row) of a
 * symmetric sparse matrix M where only the upper triangular portion of M is
 * given.
 *
 * @param M Upper triangular part of sparse symmetric matrix.
 * @param norm Result vector of length n.
 */
void col_inf_norm_USymm(const QCOSCscMatrix* M, QCOSFloat* norm);

/**
 * @brief Computes the infinity norm of each row of M and stores in norm.
 *
 * @param M An m by n sparse matrix.
 * @param norm Result vector of length m.
 */
void row_inf_norm(const QCOSCscMatrix* M, QCOSFloat* norm);

/**
 * @brief Allocates and computes A^T.
 *
 * @param A Input matrix.
 */
QCOSCscMatrix* create_transposed_matrix(const QCOSCscMatrix* A);

/**
 * @brief Scales the rows of M by E and columns of M by D.
 * M = diag(E) * M * diag(S)
 *
 * @param M An m by n sparse matrix.
 * @param E Vector of length m.
 * @param D Vector of length m.
 */
void row_col_scale(const QCOSCscMatrix* M, QCOSFloat* E, QCOSFloat* D);

/**
 * @brief Computes elementwise product z = x .* y
 *
 * @param x Input array.
 * @param y Input array.
 * @param n Length of arrays.
 */
void ew_product(QCOSFloat* x, const QCOSFloat* y, QCOSFloat* z, QCOSInt n);

/**
 * @brief Inverts permutation vector p and stores inverse in pinv.
 *
 * @param p Input permutation vector.
 * @param pinv Inverse of permutation vector.
 * @param n Length of vectors.
 */
void invert_permutation(const QCOSInt* p, QCOSInt* pinv, QCOSInt n);

/**
 * @brief Computes cumulative sum of c.
 * @return Cumulative sum of c.
 */
QCOSInt cumsum(QCOSInt* p, QCOSInt* c, QCOSInt n);

/**
 * @brief C = A(p,p) = PAP' where A and C are symmetric and the upper triangular
 * part is stored.
 *
 * @param A
 * @param pinv
 * @param AtoC
 * @param values
 * @return QCOSCscMatrix*
 */
QCOSCscMatrix* csc_symperm(const QCOSCscMatrix* A, const QCOSInt* pinv,
                           QCOSInt* AtoC);

#endif /* #ifndef LINALG_H*/
