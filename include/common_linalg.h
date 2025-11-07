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
 * @brief Allocates a new csc matrix that is lambda * I.
 *
 * @param n Size of identity matrix.
 * @param lambda Scaling factor for identity.
 * @return Pointer to new constructed matrix.
 */
QOCOCscMatrix* construct_identity(QOCOInt n, QOCOFloat lambda);

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
QOCOCscMatrix* regularize_P(QOCOInt num_diagP, QOCOCscMatrix* P, QOCOFloat reg,
                            QOCOInt* nzadded_idx);

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

#endif /* #ifndef COMMON_LINALG_H */