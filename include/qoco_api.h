/**
 * @file qoco_api.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Exposes the API for QOCO.
 */

#ifndef QOCO_API_H
#define QOCO_API_H

#include "definitions.h"
#include "enums.h"
#include "equilibration.h"
#include "input_validation.h"
#include "kkt.h"
#include "qoco_linalg.h"
#include "qoco_status.h"
#include "qoco_utils.h"
#include "structs.h"

/*
 QOCO solves second-order cone problems of the following form
 minimize   (1/2)x'Px + c'x
 subject to Gx \leq_C h <==> h - Gx \in C
            Ax = b

 where C is the cartesian product of the non-negative orthant and second-order
 cones
*/

/**
 * @brief Allocates all memory needed for QOCO to solve the SOCP.
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
 * @return 0 if no error or flag containing error code.
 */
QOCOInt qoco_setup(QOCOSolver* solver, QOCOInt n, QOCOInt m, QOCOInt p,
                   QOCOCscMatrix* P, QOCOFloat* c, QOCOCscMatrix* A,
                   QOCOFloat* b, QOCOCscMatrix* G, QOCOFloat* h, QOCOInt l,
                   QOCOInt nsoc, QOCOInt* q, QOCOSettings* settings);

/**
 * @brief Sets the data for a compressed sparse column matrix.
 *
 * @param A Pointer to the CSC matrix.
 * @param m Number of rows in the matrix.
 * @param n Number of columns in the matrix.
 * @param Annz Number of nonzero elements in the matrix.
 * @param Ax Array of data for the matrix.
 * @param Ap Array of column pointers for the data.
 * @param Ai Array of row indices for data.
 */
void qoco_set_csc(QOCOCscMatrix* A, QOCOInt m, QOCOInt n, QOCOInt Annz,
                  QOCOFloat* Ax, QOCOInt* Ap, QOCOInt* Ai);

/**
 * @brief Set the default settings struct.
 *
 * @param settings Pointer to settings struct.
 */
void set_default_settings(QOCOSettings* settings);

/**
 * @brief Updates settings struct.
 *
 * @param solver Pointer to solver.
 * @param new_settings New settings struct.
 * @return 0 if update is successful.
 */
QOCOInt qoco_update_settings(QOCOSolver* solver,
                             const QOCOSettings* new_settings);

/**
 * @brief Updates data vectors. NULL can be passed in for any vector if that
 * data will not be updated.
 *
 * @param solver Pointer to solver.
 * @param cnew New c vector.
 * @param bnew New b vector.
 * @param hnew New h vector.
 */
void update_vector_data(QOCOSolver* solver, QOCOFloat* cnew, QOCOFloat* bnew,
                        QOCOFloat* hnew);

/**
 * @brief Updates data matrices. NULL can be passed in for any matrix data
 * pointers if that matrix will not be updated. It is assumed that the new
 * matrix will have the same sparsity structure as the existing matrix.
 *
 * @param solver Pointer to solver.
 * @param Pxnew New data for P->x.
 * @param Axnew New data for A->x.
 * @param Gxnew New data for G->x.
 */
void update_matrix_data(QOCOSolver* solver, QOCOFloat* Pxnew, QOCOFloat* Axnew,
                        QOCOFloat* Gxnew);

/**
 * @brief Solves SOCP.
 *
 * @param solver Pointer to solver.
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QOCOInt qoco_solve(QOCOSolver* solver);

/**
 * @brief Frees all memory allocated by qoco_setup.
 *
 * @param solver Pointer to solver.
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QOCOInt qoco_cleanup(QOCOSolver* solver);

#endif /* #ifndef QOCO_API_H */