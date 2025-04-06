/**
 * @file equilibration.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides functions to equilibrate problem data and scale variables.
 */

#ifndef QOCO_EQUILIBRATION_H
#define QOCO_EQUILIBRATION_H

#include "qoco_linalg.h"
#include "structs.h"

/**
 * @brief Applies modified ruiz equilibration to scale data matrices. Computes
 D, E, F, and k as shown below to make the row and column infinity norms equal
 for the scaled KKT matrix.
 *
 *  * clang-format off
 *
 *  [ D     ] [ kP   A^T   G^T ] [ D     ]
 *  |   E   | |  A    0     0  | |   E   |
 *  [     F ] [  G    0     0  ] [     F ]
 *
 * clang-format on

 *
 * @param solver Pointer to solver.
 */
void ruiz_equilibration(QOCOSolver* solver);

/**
 * @brief Undo variable transformation induced by ruiz equilibration.
 *
 * @param work Pointer to workspace.
 */
void unscale_variables(QOCOWorkspace* work);

#endif /* #ifndef QOCO_EQUILIBRATION_H */