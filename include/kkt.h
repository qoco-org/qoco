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
 * Provides various functions for constructing and updating blocks for the KKT
 * matrix.
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
 * @param work Pointer to QCOSWorkspace
 */
void construct_kkt(QCOSWorkspace* work);

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
 * @brief Updates Nesterov-Todd scaling block of KKT matrix.
 *
 *     [ P   A^T   G^T  ]
 * K = | A    0     0   |
 *     [ G    0   -W'W  ]
 *
 * @param work Pointer to workspace.
 */
void update_nt_block(QCOSWorkspace* work);

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
void compute_kkt_residual(QCOSWorkspace* work);

/**
 * @brief Constructs rhs for the affine scaling KKT system.
 * Before calling this function, work->kkt->rhs must contain the
 * residual of the KKT conditions as computed by compute_kkt_residual().
 *
 * @param work Pointer to workspace.
 */
void construct_kkt_aff_rhs(QCOSWorkspace* work);

/**
 * @brief Performs Mehrotra predictor-corrector step.
 *
 * @param solver Pointer to solver.
 */
void predictor_corrector(QCOSSolver* solver);

#endif /* #ifndef QCOS_KKT_H */