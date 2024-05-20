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
 * @param x Output array.
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
QCOSFloat norm_inf(const QCOSFloat* x, QCOSInt n);

/**
 * @brief Adds lambda * I to a CSC matrix. Called on P prior to construction of
 * KKT system in qcos_setup(). This function calls realloc() when adding new
 * nonzeros.
 *
 * @param M
 * @param lambda
 */
void regularize(QCOSCscMatrix* M, QCOSFloat lambda);

#endif /* #ifndef LINALG_H*/
