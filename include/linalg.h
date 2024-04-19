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

#endif /* #ifndef LINALG_H*/
