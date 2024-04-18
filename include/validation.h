/**
 * @file validation.h
 * @author Govind Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Includes functions that validate any user-provided data.
 */

#ifndef VALIDATION_H
#define VALIDATION_H

#include "structs.h"

/**
 * @brief Validates solver settings.
 *
 * @param settings Pointer to settings struct
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QCOSInt qcos_validate_settings(QCOSSettings* settings);

/**
 * @brief Validate problem data.
 *
 * @param P Upper triangular part of quadratic cost Hessian in CSC form
 * @param c Linear cost vector
 * @param A Affine equality constraint matrix in CSC form
 * @param b Affine equality constraint offset vector
 * @param G Conic constraint matrix in CSC form
 * @param h Conic constraint offset vector
 * @param l Dimension of non-negative orthant
 * @param ncones Number of second-order cones
 * @param q Dimension of each second-order cone
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QCOSInt qcos_validate_data(const QCOSCscMatrix* P, const QCOSFloat* c,
                           const QCOSCscMatrix* A, const QCOSFloat* b,
                           const QCOSCscMatrix* G, const QCOSFloat* h,
                           const QCOSInt l, const QCOSInt ncones,
                           const QCOSInt* q);

#endif /* #ifndef VALIDATION_H */