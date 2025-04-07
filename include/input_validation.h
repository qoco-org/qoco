/**
 * @file input_validation.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Includes functions that validate any user-provided data.
 */

#ifndef QOCO_INPUT_VALIDATION_H
#define QOCO_INPUT_VALIDATION_H

#include "enums.h"
#include "qoco_status.h"
#include "structs.h"
#include <stdio.h>

/**
 * @brief Validates solver settings.
 *
 * @param settings Pointer to settings struct
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QOCOInt qoco_validate_settings(const QOCOSettings* settings);

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
 * @param nsoc Number of second-order cones
 * @param q Dimension of each second-order cone
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QOCOInt qoco_validate_data(const QOCOCscMatrix* P, const QOCOFloat* c,
                           const QOCOCscMatrix* A, const QOCOFloat* b,
                           const QOCOCscMatrix* G, const QOCOFloat* h,
                           const QOCOInt l, const QOCOInt nsoc,
                           const QOCOInt* q);

#endif /* #ifndef QOCO_INPUT_VALIDATION_H */