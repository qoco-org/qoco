/**
 * @file common_linalg.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2025, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Exposes the API for QOCO.
 */

#ifndef COMMON_LINALG_H
#define COMMON_LINALG_H
#include "definitions.h"
#include "qoco_linalg.h"

/**
 * @brief Frees all the internal arrays and the pointer to the QOCOCscMatrix.
 * Should only be used if QOCOCscMatrix and all internal arrays were malloc'ed.
 *
 * @param A Pointer to QOCOCscMatrix.
 */
void free_qoco_csc_matrix(QOCOCscMatrix* A);

/**
 * @brief Allocates a new csc matrix that is lambda * I.
 *
 * @param n Size of identity matrix.
 * @param lambda Scaling factor for identity.
 * @return Pointer to new constructed matrix.
 */
QOCOCscMatrix* construct_identity_csc(QOCOInt n, QOCOFloat lambda);

/**
 * @brief Counts the number of diagonal elements in upper triangular CSC matrix
 * M.
 *
 * @param M Input matrix.
 * @return Number of nonzeros on the diagonal of M.
 */
QOCOInt count_diag(QOCOCscMatrix* M);

/**
 * @brief Adds reg * I to a CSC matrix. Called on P prior to construction
 * of KKT system in qoco_setup(). This function calls realloc() when adding
 * new nonzeros. Matrix P is freed using free_qoco_csc_matrix
 *
 * @param num_diagP Number of new element added to diagonal.
 * @param P Matrix to be regularized.
 * @param reg Regularization factor.
 * @param nzadded_idx Indices of elements of M->x that are added.
 * @return P + reg * I.
 */
QOCOCscMatrix* regularize_P_csc(QOCOInt num_diagP, QOCOCscMatrix* P, QOCOFloat reg,
                            QOCOInt* nzadded_idx);

/**
 * @brief Subtracts lambda * I to a CSC matrix. Called on P when updating
 * matrix data in update_matrix_data(). This function does not allocate and must
 * be called after regularize.
 *
 * @param M Matrix.
 * @param lambda Regularization.
 */
void unregularize_csc(QOCOCscMatrix* M, QOCOFloat lambda);

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
QOCOCscMatrix* create_transposed_csc_matrix(const QOCOCscMatrix* A, QOCOInt* AtoAt);

/**
 * @brief Scales the rows of M by E and columns of M by D.
 * M = diag(E) * M * diag(S)
 *
 * @param M An m by n QOCOCscMatrix matrix.
 * @param E Vector of length m.
 * @param D Vector of length m.
 */
void row_col_scale_csc(const QOCOCscMatrix* M, QOCOFloat* E, QOCOFloat* D);

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
 * @brief Computes elementwise product z = x .* y
 *
 * @param x Input array.
 * @param y Input array.
 * @param z Output array.
 * @param n Length of arrays.
 */
void ew_product_arrayf(QOCOFloat* x, QOCOFloat* y, QOCOFloat* z, QOCOInt n);

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
void scale_arrayf(QOCOFloat* x, QOCOFloat* y, QOCOFloat s, QOCOInt n);

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

#endif /* #ifndef COMMON_LINALG_H */