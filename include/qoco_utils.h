/**
 * @file qoco_utils.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides various utility functions.
 */

#ifndef QOCO_UTILS_H
#define QOCO_UTILS_H

#include "enums.h"
#include "qoco_linalg.h"
#include "qoco_status.h"
#include "structs.h"
#include <stdio.h>

/**
 * @brief Prints dimensions, number of nonzero elements, data, column pointers
 * and row indices for a sparse matrix in CSC form.
 *
 * @param M Pointer to QOCOCscMatrix that will be printed.
 */
void print_qoco_csc_matrix(QOCOCscMatrix* M);

/**
 * @brief Prints array of QOCOFloats.
 *
 * @param x Pointer to array.
 * @param n Number of elements in array.
 */
void print_arrayf(QOCOFloat* x, QOCOInt n);

/**
 * @brief Prints array of QOCOInts.
 *
 * @param x Pointer to array.
 * @param n Number of elements in array.
 */
void print_arrayi(QOCOInt* x, QOCOInt n);

/**
 * @brief Prints QOCO header.
 *
 * @param solver Pointer to solver.
 */
void print_header(QOCOSolver* solver);

/**
 * @brief Print solver progress.
 *
 * @param solver Pointer to solver.
 */
void log_iter(QOCOSolver* solver);

/**
 * @brief Prints QOCO footer.
 *
 * @param solution Pointer to solution struct.
 * @param status Solve status.
 */
void print_footer(QOCOSolution* solution, enum qoco_solve_status status);

/**
 * @brief Checks stopping criteria.
 * Before calling this function, work->kkt->rhs must contain the
 * residual of the KKT conditions as computed by compute_kkt_residual().
 * @param solver Pointer to solver.
 * @return 1 if stopping criteria met and 0 otherwise.
 */
unsigned char check_stopping(QOCOSolver* solver);

/**
 * @brief Copies data to QOCOSolution struct when solver terminates.
 *
 * @param solver Pointer to solver.
 */
void copy_solution(QOCOSolver* solver);

/**
 * @brief Allocates and returns a copy of the input settings struct.
 *
 * @param settings Input struct.
 * @return Pointer to constructed and copies settings struct.
 */
QOCOSettings* copy_settings(QOCOSettings* settings);

#endif /* #ifndef QOCO_UTILS_H */