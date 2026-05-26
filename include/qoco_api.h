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

#include "common_linalg.h"
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
 * @brief Allocates a fixed-size batch of QOCO solvers for problems with
 * identical dimensions and sparsity.
 *
 * The CUDA backend solves batch items concurrently. The builtin backend returns
 * QOCO_UNSUPPORTED_ERROR.
 *
 * @param batch Pointer to batch solver.
 * @param batch_size Number of problem instances.
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
 * @return QOCO_NO_ERROR if setup succeeds.
 */
QOCOInt qoco_batch_setup(QOCOBatchSolver* batch, QOCOInt batch_size,
                         QOCOInt n, QOCOInt m, QOCOInt p, QOCOCscMatrix* P,
                         QOCOFloat* c, QOCOCscMatrix* A, QOCOFloat* b,
                         QOCOCscMatrix* G, QOCOFloat* h, QOCOInt l,
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
void qoco_update_vector_data(QOCOSolver* solver, QOCOFloat* cnew,
                             QOCOFloat* bnew, QOCOFloat* hnew);

/**
 * @brief Updates data vectors for one batch item.
 *
 * NULL can be passed in for any vector if that data will not be updated.
 *
 * @param batch Pointer to batch solver.
 * @param item Batch item index.
 * @param cnew New c vector.
 * @param bnew New b vector.
 * @param hnew New h vector.
 * @return QOCO_NO_ERROR if update succeeds.
 */
QOCOInt qoco_batch_update_vector_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* cnew, QOCOFloat* bnew,
                                      QOCOFloat* hnew);

/**
 * @brief Sets an optional primal starting point after qoco_setup. The x0 vector
 * is copied and should be supplied in the original, unequilibrated problem
 * scaling. Pass NULL to clear a previously set custom starting point.
 *
 * @param solver Pointer to solver.
 * @param x0 Primal starting point of length n, or NULL to use default
 * initialization.
 */
void qoco_set_x0(QOCOSolver* solver, const QOCOFloat* x0);

/**
 * @brief Sets an optional primal starting point for one batch item.
 *
 * @param batch Pointer to batch solver.
 * @param item Batch item index.
 * @param x0 Primal starting point of length n, or NULL to use default
 * initialization.
 * @return QOCO_NO_ERROR if update succeeds.
 */
QOCOInt qoco_batch_set_x0(QOCOBatchSolver* batch, QOCOInt item,
                          const QOCOFloat* x0);

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
void qoco_update_matrix_data(QOCOSolver* solver, QOCOFloat* Pxnew,
                             QOCOFloat* Axnew, QOCOFloat* Gxnew);

/**
 * @brief Updates matrix nonzero values for one batch item.
 *
 * The new values must match the sparsity structure supplied to batch setup.
 * NULL can be passed in for any matrix if that data will not be updated.
 *
 * @param batch Pointer to batch solver.
 * @param item Batch item index.
 * @param Pxnew New data for P->x.
 * @param Axnew New data for A->x.
 * @param Gxnew New data for G->x.
 * @return QOCO_NO_ERROR if update succeeds.
 */
QOCOInt qoco_batch_update_matrix_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* Pxnew, QOCOFloat* Axnew,
                                      QOCOFloat* Gxnew);

/**
 * @brief Solves SOCP.
 *
 * @param solver Pointer to solver.
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QOCOInt qoco_solve(QOCOSolver* solver);

/**
 * @brief Solves all batch items and stores per-item statuses in
 * batch->statuses.
 *
 * @param batch Pointer to batch solver.
 * @return QOCO_NO_ERROR if all items were attempted.
 */
QOCOInt qoco_batch_solve(QOCOBatchSolver* batch);

/**
 * @brief Returns the solution struct for one batch item.
 *
 * @param batch Pointer to batch solver.
 * @param item Batch item index.
 * @return Per-item solution, or NULL for an invalid item.
 */
QOCOSolution* qoco_batch_get_solution(QOCOBatchSolver* batch, QOCOInt item);

/**
 * @brief Frees all memory allocated by qoco_setup.
 *
 * @param solver Pointer to solver.
 * @return Exitflag to check (0 for success, failure otherwise)
 */
QOCOInt qoco_cleanup(QOCOSolver* solver);

/**
 * @brief Frees all memory allocated by qoco_batch_setup.
 *
 * The batch object itself is not freed.
 *
 * @param batch Pointer to batch solver.
 * @return QOCO_NO_ERROR if cleanup succeeds.
 */
QOCOInt qoco_batch_cleanup(QOCOBatchSolver* batch);

#endif /* #ifndef QOCO_API_H */
