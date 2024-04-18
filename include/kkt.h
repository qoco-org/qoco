/**
 * @file kkt.h
 * @author Govind Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides various functions for constructing and updating blocks for the KKT
 * matrix.
 */

#ifndef QCOS_KKT_H
#define QCOS_KKT_H

#include "structs.h"

/**
 * @brief Allocate memory for KKT matrix.
 *
 * @param data Pointer to problem data.
 * @return Pointer to initialized KKT matrix.
 */
QCOSCscMatrix* initialize_kkt(QCOSProblemData* data);

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

#endif /* #ifndef QCOS_KKT_H */