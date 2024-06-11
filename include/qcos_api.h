/**
 * @file qcos_api.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Exposes the API for QCOS.
 */

#ifndef QCOS_API_H
#define QCOS_API_H

#include "definitions.h"
#include "enums.h"
#include "equilibration.h"
#include "input_validation.h"
#include "kkt.h"
#include "linalg.h"
#include "qcos_error.h"
#include "structs.h"
#include "utils.h"

/*
 QCOS solves second-order cone problems of the following form
 minimize   (1/2)x'Px + c'x
 subject to Gx \leq_C h <==> h - Gx \in C
            Ax = b

 where C is the cartesian product of the non-negative orthant and second-order
 cones
*/

/**
 * @brief Allocates all memory needed for QCOS to solve the SOCP.
 *
 * @param solver Pointer to solver.
 * @param n Number of optimization variables.
 * @param m Number of conic constraints.
 * @param p Number of affine equality constraints.
 * @param P Upper triangular part of quadratic cost Hessian in CSC form.
 * @param c Linear cost vector.
 * @param A Affine equality constraint matrix in CSC form.
 * @param b Affine equality constraint offset vector.
 * @param G Conic constraint matrix in CSC form.
 * @param h Conic constraint offset vector.
 * @param l Dimension of non-negative orthant.
 * @param nsoc Number of second-order cones.
 * @param q Dimension of each second-order cone.
 * @param settings Settings struct.
 * @return error code.
 */
QCOSInt qcos_setup(QCOSSolver* solver, QCOSInt n, QCOSInt m, QCOSInt p,
                   QCOSCscMatrix* P, QCOSFloat* c, QCOSCscMatrix* A,
                   QCOSFloat* b, QCOSCscMatrix* G, QCOSFloat* h, QCOSInt l,
                   QCOSInt nsoc, QCOSInt* q, QCOSSettings* settings);

/**
 * @brief Sets the data for a compressed sparse column matrix.
 *
 * @param A Pointer to the CSC matrix
 * @param m Number of rows in the matrix
 * @param n Number of columns in the matrix
 * @param Annz Number of nonzero elements in the matrix
 * @param Ax Array of data for the matrix
 * @param Ap Array of column pointers for the data
 * @param Ai Array of row indices for data
 */
void qcos_set_csc(QCOSCscMatrix* A, QCOSInt m, QCOSInt n, QCOSInt Annz,
                  QCOSFloat* Ax, QCOSInt* Ap, QCOSInt* Ai);

/**
 * @brief Set the default settings struct
 *
 * @param settings Pointer to settings struct
 */
void set_default_settings(QCOSSettings* settings);

/**
 * @brief Solves SOCP.
 *
 * @param solver Pointer to solver.
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QCOSInt qcos_solve(QCOSSolver* solver);

/**
 * @brief Frees all allocated memory.
 *
 * @param solver Pointer to solver.
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QCOSInt qcos_cleanup(QCOSSolver* solver);

#endif /* #ifndef QCOS_API_H */