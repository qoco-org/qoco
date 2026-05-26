/**
 * @file qoco_batch.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2026, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Exposes the batched QOCO API.
 */

#ifndef QOCO_BATCH_H
#define QOCO_BATCH_H

#include "definitions.h"
#include "qoco_api.h"
#include "structs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocates a batch of QOCO solvers with common dimensions and sparsity.
 *
 * @param batch Pointer to batch solver.
 * @param batch_count Number of solver instances.
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
QOCOInt qoco_batch_setup(QOCOBatchSolver* batch, QOCOInt batch_count, QOCOInt n,
                         QOCOInt m, QOCOInt p, QOCOCscMatrix* P, QOCOFloat* c,
                         QOCOCscMatrix* A, QOCOFloat* b, QOCOCscMatrix* G,
                         QOCOFloat* h, QOCOInt l, QOCOInt nsoc, QOCOInt* q,
                         QOCOSettings* settings);

/**
 * @brief Updates vector data for one item in a batch.
 */
QOCOInt qoco_batch_update_vector_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* cnew, QOCOFloat* bnew,
                                      QOCOFloat* hnew);

/**
 * @brief Updates matrix values for one item in a batch. Sparsity must match
 * the matrices supplied to qoco_batch_setup.
 */
QOCOInt qoco_batch_update_matrix_data(QOCOBatchSolver* batch, QOCOInt item,
                                      QOCOFloat* Pxnew, QOCOFloat* Axnew,
                                      QOCOFloat* Gxnew);

/**
 * @brief Solves all items in a batch.
 *
 * @return QOCO_NO_ERROR if dispatch succeeded. Per-item solve statuses are
 * stored in batch->statuses.
 */
QOCOInt qoco_batch_solve(QOCOBatchSolver* batch);

/**
 * @brief Returns the solution for one item in a batch, or NULL for an invalid
 * item.
 */
QOCOSolution* qoco_batch_get_solution(QOCOBatchSolver* batch, QOCOInt item);

/**
 * @brief Frees all memory owned by a batch solver.
 */
QOCOInt qoco_batch_cleanup(QOCOBatchSolver* batch);

#ifdef __cplusplus
}
#endif

#endif /* #ifndef QOCO_BATCH_H */
