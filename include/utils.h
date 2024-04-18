/**
 * @file utils.h
 * @author Govind Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind Chari
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
 * @brief Prints QCOSVector.
 *
 * @param V Pointer to QCOSVector that will be printed.
 */
void print_qcos_vector(QCOSVector* V);

/**
 * @brief Prints qcos header.
 *
 */
void print_header();

#endif /* #ifndef UTILS_H */