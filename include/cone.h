/**
 * @file cone.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Includes various functions necessary for cone operations.
 */

#ifndef CONE_H
#define CONE_H

#include "linalg.h"

/**
 * @brief Computes cone product u * v = p with respect to C.
 *
 * @param u Input vector.
 * @param v Input vector.
 * @param p Cone product of u and v.
 * @param data Pointer to problem data.
 */
void cone_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p,
                  QCOSProblemData* data);

/**
 * @brief Computes second-order cone product u * v = p.
 *
 * @param u Input vector.
 * @param v Input vector.
 * @param p Cone product of u and v.
 * @param n Length of vectors.
 */
void soc_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p, QCOSInt n);

/**
 * @brief Computes residual of vector u with respect to the second order cone of
 * dimension n.
 *
 * @param u u = (u0, u1) is a vector in second order cone of dimension n.
 * @param n Dimension of second order cone.
 * @return Residual: norm(u1) - u0. Negative if the vector is in the cone and
 * positive otherwise.
 */
QCOSFloat soc_residual(QCOSFloat* u, QCOSInt n);

/**
 * @brief Computes residual of vector u with respect to cone C.
 *
 * @param u Vector to be tested.
 * @param data Pointer to problem data.
 * @return Residual: Negative if the vector is in the cone and positive
 * otherwise.
 */
QCOSFloat cone_residual(QCOSFloat* u, QCOSProblemData* data);

/**
 * @brief Performs u = u + (1 + a) * e where e is the cannonical vector for each
 * cone LP Cone: e = ones(n), second-order cone: e = (1,0,0,...) and a is the
 * minimum scalar value such that u + (1 + a) * e is in cone C.
 *
 * @param u Vector to bring to cone.
 * @param data Pointer to problem data.
 */
void bring2cone(QCOSFloat* u, QCOSProblemData* data);

#endif