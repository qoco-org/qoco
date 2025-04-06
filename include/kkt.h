/**
 * @file kkt.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides various functions for solving, constructing and updating KKT
 * systems.
 */

#ifndef QOCO_KKT_H
#define QOCO_KKT_H

#include "cone.h"
#include "qdldl.h"
#include "qoco_linalg.h"
#include "structs.h"

/**
 * @brief Allocate memory for KKT matrix.
 *
 * @param work Pointer to workspace.
 */
void allocate_kkt(QOCOWorkspace* work);

/**
 * @brief Constructs upper triangular part of KKT matrix with -I
 * for Nestrov-Todd scaling matrix (the (3,3) block)
 *
 * clang-format off
 *
 *     [ P   A^T   G^T ]
 * K = | A    0     0  |
 *     [ G    0    -I  ]
 *
 * clang-format on
 *
 * @param solver Pointer to solver
 */
void construct_kkt(QOCOSolver* solver);

/**
 * @brief Gets initial values for primal and dual variables such that (s,z) \in
 * C.
 *
 * @param solver Pointer to solver.
 */
void initialize_ipm(QOCOSolver* solver);

/**
 * @brief Set the Nesterov-Todd block to be zeros. Used prior to
 * compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void set_nt_block_zeros(QOCOWorkspace* work);

/**
 * @brief Updates and regularizes Nesterov-Todd scaling block of KKT matrix.
 *
 *     [ P   A^T       G^T      ]
 * K = | A    0         0       |
 *     [ G    0   -W'W - e * I  ]
 *
 * @param solver Pointer to solver.
 */
void update_nt_block(QOCOSolver* solver);

/**
 * @brief Computes residual of KKT conditions and stores in work->kkt->rhs.
 *
 * clang-format off
 *
 *       [ P   A^T   G^T ] [ x ]   [    c   ]
 * res = | A    0     0  | | y ] + |   -b   |
 *       [ G    0     0  ] [ z ]   [ -h + s ]
 *
 * clang-format on
 *
 * @param solver Pointer to solver.
 */
void compute_kkt_residual(QOCOSolver* solver);

/**
 * @brief Constructs rhs for the affine scaling KKT system.
 * Before calling this function, work->kkt->kktres must contain the
 * residual of the KKT conditions as computed by compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void construct_kkt_aff_rhs(QOCOWorkspace* work);

/**
 * @brief Constructs rhs for the combined direction KKT system.
 * Before calling this function, work->kkt->kktres must contain the
 * negative residual of the KKT conditions as computed by
 * compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void construct_kkt_comb_rhs(QOCOWorkspace* work);

/**
 * @brief Performs Mehrotra predictor-corrector step.
 *
 * @param solver Pointer to solver.
 */
void predictor_corrector(QOCOSolver* solver);

/**
 * @brief Solves Kx = b once K has been factored. Solves via triangular solves
 * and applies iterative refinement afterwards.
 *
 * @param solver Pointer to solver.
 * @param b Pointer to rhs of kkt system.
 * @param iters Number of iterations of iterative refinement performed.
 */
void kkt_solve(QOCOSolver* solver, QOCOFloat* b, QOCOInt iters);

/**
 * @brief Computes y = Kx where
 *     [ P   A^T       G^T      ]
 * K = | A    0         0       |
 *     [ G    0   -W'W - e * I  ]
 *
 * @param solver Pointer to solver.
 * @param x Pointer to input vector.
 * @param y Pointer to output vector.
 */
void kkt_multiply(QOCOSolver* solver, QOCOFloat* x, QOCOFloat* y);

#endif /* #ifndef QOCO_KKT_H */