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
 * @brief Computes second-order cone product u * v = p.
 *
 * @param u u = (u0, u1) is a vector in second-order cone of dimension n.
 * @param v v = (v0, v1) is a vector in second-order cone of dimension n.
 * @param p Cone product of u and v.
 * @param n Dimension of second-order cone.
 */
void soc_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p, QCOSInt n);

/**
 * @brief Commpues second-order cone division lambda # v = d
 *
 * @param lam lam = (lam0, lam1) is a vector in second-order cone of dimension
 * n.
 * @param v v = (v0, v1) is a vector in second-order cone of dimension n.
 * @param d Cone divisin of lam and v.
 * @param n Dimension of second-order cone.
 */
void soc_division(QCOSFloat* lam, QCOSFloat* v, QCOSFloat* d, QCOSInt n);

/**
 * @brief Computes residual of vector u with respect to the second order cone of
 * dimension n.
 *
 * @param u u = (u0, u1) is a vector in second-order cone of dimension n.
 * @param n Dimension of second order cone.
 * @return Residual: norm(u1) - u0. Negative if the vector is in the cone and
 * positive otherwise.
 */
QCOSFloat soc_residual(QCOSFloat* u, QCOSInt n);

/**
 * @brief Computes u0^2 - u1'*u1 of vector u with respect to the second order
 * cone of dimension n.
 *
 * @param u u = (u0, u1) is a vector in second order cone of dimension n.
 * @param n Dimension of second order cone.
 * @return Residual: u0^2 - u1'*u1.
 */
QCOSFloat soc_residual2(QCOSFloat* u, QCOSInt n);

/**
 * @brief Computes cone product u * v = p with respect to C.
 *
 * @param u Input vector.
 * @param v Input vector.
 * @param p Cone product of u and v.
 * @param l Dimension of LP cone.
 * @param ncones Number of second-order cones.
 * @param q Dimension of each second-order cone.
 */
void cone_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p, QCOSInt l,
                  QCOSInt ncones, QCOSInt* q);

/**
 * @brief Computed cone division lambda # v = d
 *
 * @param lambda Input vector.
 * @param v Input vector.
 * @param d Cone quotient of lambda and v.
 * @param l Dimension of LP cone.
 * @param ncones Number of second-order cones.
 * @param q Dimension of each second-order cone.
 */
void cone_division(QCOSFloat* lambda, QCOSFloat* v, QCOSFloat* d, QCOSInt l,
                   QCOSInt ncones, QCOSInt* q);

/**
 * @brief Computes residual of vector u with respect to cone C.
 *
 * @param u Vector to be tested.
 * @param l Dimension of LP cone.
 * @param ncones Number of second-order cones.
 * @param q Dimension of each second-order cone.
 * @return Residual: Negative if the vector is in the cone and positive
 * otherwise.
 */
QCOSFloat cone_residual(QCOSFloat* u, QCOSInt l, QCOSInt ncones, QCOSInt* q);

/**
 * @brief Performs u = u + (1 + a) * e where e is the cannonical vector for each
 * cone LP Cone: e = ones(n), second-order cone: e = (1,0,0,...) and a is the
 * minimum scalar value such that u + (1 + a) * e is in cone C.
 *
 * @param u Vector to bring to cone.
 * @param data Pointer to problem data.
 */
void bring2cone(QCOSFloat* u, QCOSProblemData* data);

/**
 * @brief Computes z = W * x where W is a full Nesterov-Todd scaling matrix. The
 * NT scaling array for the LP cones are stored first, then the NT scalings for
 * the second-order order cones are stores in column major order.
 *
 * @param W Nesterov Todd scaling matrix.
 * @param x Input vector.
 * @param z Output vector.
 * @param l Dimension of LP cone.
 * @param m Length of x.
 * @param ncones Number of second-order cones in C.
 * @param q Array of second-order cone dimensions.
 */
void nt_multiply(QCOSFloat* W, QCOSFloat* x, QCOSFloat* z, QCOSInt l, QCOSInt m,
                 QCOSInt ncones, QCOSInt* q);

/**
 * @brief Computes gap (z'*s / m) and stores in work->mu.
 *
 * @param work Pointer to workspace.
 */
void compute_mu(QCOSWorkspace* work);

/**
 * @brief Compute Nesterov-Todd scalings and scaled variables.
 *
 * @param work Pointer to workspace.
 */
void compute_nt_scaling(QCOSWorkspace* work);

/**
 * @brief Computes centering parameter.
 *
 * @param solver Pointer to solver.
 */
void compute_centering(QCOSSolver* solver);

/**
 * @brief Conducts linesearch by bisection to compute a \in (0, 1] such that
 * u + (a / f) * Du \in C
 * Warning: linesearch overwrites ubuff1. Do not pass in ubuff1 into u or Du.
 * Consider a dedicated buffer for linesearch.
 *
 * @param u Initial vector.
 * @param Du Search direction.
 * @param f Conservatism factor.
 * @param work Pointer to workspace.
 * @return Step-size.
 */
QCOSFloat linesearch(QCOSFloat* u, QCOSFloat* Du, QCOSFloat f,
                     QCOSSolver* solver);

#endif