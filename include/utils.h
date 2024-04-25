/**
 * @file utils.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides useful utility functions.
 */

#ifndef UTILS_H
#define UTILS_H

#include "structs.h"
#include <stdio.h>

/**
 * @brief Prints QCOSCscMatrix.
 *
 * @param M Pointer to QCOSCscMatrix that will be printed.
 */
void print_qcos_csc_matrix(QCOSCscMatrix* M);

/**
 * @brief Prints array of QCOSFloats.
 *
 * @param x Pointer to array.
 * @param n Number of elements in array.
 */
void print_arrayf(QCOSFloat* x, QCOSInt n);

/**
 * @brief Prints array of QCOSInts.
 *
 * @param x Pointer to array.
 * @param n Number of elements in array.
 */
void print_arrayi(QCOSInt* x, QCOSInt n);

/**
 * @brief Prints qcos header.
 *
 */
void print_header();

/**
 * @brief Checks stopping criteria.
 *
 * @param solver
 * @return 1 if stopping criteria met and 0 otherwise.
 */
unsigned char check_stopping(QCOSSolver* solver);

#endif /* #ifndef UTILS_H */