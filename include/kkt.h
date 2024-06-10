/**
 * @file kkt.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides various functions for solving, constructing and updating KKT
 * systems.
 */

#ifndef QCOS_KKT_H
#define QCOS_KKT_H

#include "cone.h"
#include "linalg.h"
#include "qdldl.h"
#include "structs.h"

/**
 * @brief Allocate memory for KKT matrix.
 *
 * @param work Pointer to workspace.
 */
void allocate_kkt(QCOSWorkspace* work);

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
 * @param work Pointer to solver
 */
void construct_kkt(QCOSSolver* solver);

/**
 * @brief Gets initial values for primal and dual variables such that (s,z) \in
 * C.
 *
 * @param solver Pointer to solver.
 */
void initialize_ipm(QCOSSolver* solver);

/**
 * @brief Set the Nesterov-Todd block to be zeros. Used prior to
 * compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void set_nt_block_zeros(QCOSWorkspace* work);

/**
 * @brief Updates and regularizes Nesterov-Todd scaling block of KKT matrix.
 *
 *     [ P   A^T       G^T      ]
 * K = | A    0         0       |
 *     [ G    0   -W'W - e * I  ]
 *
 * @param work Pointer to solver.
 */
void update_nt_block(QCOSSolver* solver);

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
 * @param work Pointer to workspace.
 */
void compute_kkt_residual(QCOSSolver* solver);

/**
 * @brief Constructs rhs for the affine scaling KKT system.
 * Before calling this function, work->kkt->kktres must contain the
 * residual of the KKT conditions as computed by compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void construct_kkt_aff_rhs(QCOSWorkspace* work);

/**
 * @brief Constructs rhs for the combined direction KKT system.
 * Before calling this function, work->kkt->kktres must contain the
 * negative residual of the KKT conditions as computed by
 * compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void construct_kkt_comb_rhs(QCOSWorkspace* work);

/**
 * @brief Performs Mehrotra predictor-corrector step.
 *
 * @param solver Pointer to solver.
 */
void predictor_corrector(QCOSSolver* solver);

/**
 * @brief Solves Kx = b once K has been factored. Solves via traingular solves
 * and applies iterative refinement afterwards.
 *
 * @param kkt Pointer to kkt struct.
 * @param b Pointer to rhs of kkt system.
 * @param iters Number of iterations of iterative refinement performed.
 */
void kkt_solve(QCOSKKT* kkt, QCOSFloat* b, QCOSInt iters);

/**
 * @brief Applies modified ruiz equilibration to scale data matrices. Computes
 D, E, F, and k as shown below to make the row and column infinity norms equal
 for the scaled KKT matrix.
 *
 *  * clang-format off
 *
 *  [ D     ] [ kP   A^T   G^T ] [ D     ]
 *  |   E   | |  A    0     0  | |   E   |
 *  [     F ] [  G    0    -I  ] [     F ]
 *
 * clang-format on

 *
 * @param solver Pointer to solver.
 */
void ruiz_equilibration(QCOSSolver* solver);

/**
 * @brief Undo variable transformation induced by ruiz equilibration.
 *
 * @param work Pointer to workspace.
 */
void unscale_variables(QCOSWorkspace* work);

#endif /* #ifndef QCOS_KKT_H */