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
 * Provides various utility functions.
 */

#ifndef UTILS_H
#define UTILS_H

#include "enums.h"
#include "linalg.h"
#include "structs.h"
#include <stdio.h>

/**
 * @brief Prints dimensions, number of nonzero elements, data, column pointers
 * and row indices for a sparse matrix in CSC form.
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
 * @brief Prints QCOS header.
 *
 * @param solver Pointer to solver.
 */
void print_header(QCOSSolver* solver);

/**
 * @brief Print solver progress.
 *
 * @param solver Pointer to solver.
 */
void log_iter(QCOSSolver* solver);

/**
 * @brief Prints QCOS footer.
 *
 * @param solution Pointer to solution struct.
 * @param status Solve status.
 */
void print_footer(QCOSSolution* solution, enum qcos_solve_status status);

/**
 * @brief Checks stopping criteria.
 * Before calling this function, work->kkt->rhs must contain the
 * residual of the KKT conditions as computed by compute_kkt_residual().
 * @param solver Pointer to solver.
 * @return 1 if stopping criteria met and 0 otherwise.
 */
unsigned char check_stopping(QCOSSolver* solver);

/**
 * @brief Copies data to QCOSSolution struct when solver terminates.
 *
 * @param solver Pointer to solver.
 */
void copy_solution(QCOSSolver* solver);

/**
 * @brief Allocates and returns a copy of the input settings struct.
 *
 * @param settings Input struct.
 * @return Pointer to constructed and copies settings struct.
 */
QCOSSettings* copy_settings(QCOSSettings* settings);

#endif /* #ifndef UTILS_H */