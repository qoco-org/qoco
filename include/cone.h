/**
 * @file cone.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Includes various functions necessary for cone operations.
 */

#ifndef QOCO_CONE_H
#define QOCO_CONE_H

#ifdef __cplusplus
extern "C" {
#endif
#include "qoco_linalg.h"
#include "qoco_utils.h"
#include "structs.h"

/**
 * @brief Sets NT scaling data to identity.
 *
 * @param nt_scaling NT scaling data.
 * @param nt_scaling_nnz Number of elements in nt_scaling.
 * @param nt_scaling_soc_idx Vector pointing to the start of each SOC block.
 *                 Only used in the GPU (cone.cu) implementation.
 * @param data Pointer to problem data.
 */
void set_nt_scaling_identity(QOCOVectorf* nt_scaling, QOCOInt nt_scaling_nnz,
                             QOCOVectori* nt_scaling_soc_idx,
                             QOCOProblemData* data);

/**
 * @brief Computes cone product u * v = p with respect to C.
 *
 * @param u Input vector.
 * @param v Input vector.
 * @param p Cone product of u and v.
 * @param l Dimension of LP cone.
 * @param nsoc Number of second-order cones.
 * @param q Dimension of each second-order cone.
 * @param soc_idx Array pointing to the start of each SOC block.
 *                Only used in the GPU (cone.cu) implementation.
 */
void cone_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                  QOCOInt l, QOCOInt nsoc, const QOCOInt* q,
                  const QOCOInt* soc_idx);

/**
 * @brief Computed cone division lambda # v = d
 *
 * @param lambda Input vector.
 * @param v Input vector.
 * @param d Cone quotient of lambda and v.
 * @param l Dimension of LP cone.
 * @param nsoc Number of second-order cones.
 * @param q Dimension of each second-order cone.
 * @param soc_idx Array pointing to the start of each SOC block.
 *                Only used in the GPU (cone.cu) implementation.
 */
void cone_division(const QOCOFloat* lambda, const QOCOFloat* v, QOCOFloat* d,
                   QOCOInt l, QOCOInt nsoc, const QOCOInt* q,
                   const QOCOInt* soc_idx);

/**
 * @brief Performs u = u + (1 + a) * e where e is the cannonical vector for each
 * cone LP Cone: e = ones(n), second-order cone: e = (1,0,0,...) and a is the
 * minimum scalar value such that u + (1 + a) * e is in cone C.
 *
 * @param u Vector to bring to cone.
 * @param soc_idx Array pointing to the start of each SOC block.
 *                Only used in the GPU (cone.cu) implementation.
 * @param data Pointer to problem data.
 */
void bring2cone(QOCOFloat* u, QOCOInt* soc_idx, QOCOProblemData* data);

/**
 * @brief Computes z = W * x using Nesterov-Todd scaling data.
 * The LP cone scalings are stored first. Each SOC block stores the fast
 * scaling parameters [eta, w0, w1...] in the builtin backend.
 *
 * @param W Nesterov Todd scaling data.
 * @param nt_scaling_soc_idx Vector pointing to the start of each SOC block in
 * W. Only used in the GPU (cone.cu) implementation.
 * @param soc_idx Array pointing to the start of each SOC block in x and z.
 *                Only used in the GPU (cone.cu) implementation.
 * @param x Input vector.
 * @param z Output vector.
 * @param l Dimension of LP cone.
 * @param m Length of x.
 * @param nsoc Number of second-order cones in C.
 * @param q Array of second-order cone dimensions.
 */
void nt_multiply(QOCOFloat* W, QOCOInt* nt_scaling_soc_idx, QOCOInt* soc_idx,
                 QOCOFloat* x, QOCOFloat* z, QOCOInt l, QOCOInt m, QOCOInt nsoc,
                 QOCOInt* q);

/**
 * @brief Computes z = W^{-1} * x using the same NT scaling data as
 * nt_multiply().
 */
void nt_multiply_inv(QOCOFloat* W, QOCOInt* nt_scaling_soc_idx,
                     QOCOInt* soc_idx, QOCOFloat* x, QOCOFloat* z, QOCOInt l,
                     QOCOInt m, QOCOInt nsoc, QOCOInt* q);

/**
 * @brief Compute Nesterov-Todd scalings and scaled variables.
 *
 * @param work Pointer to workspace.
 */
void compute_nt_scaling(QOCOWorkspace* work);

/**
 * @brief Computes centering parameter.
 *
 * @param solver Pointer to solver.
 */
void compute_centering(QOCOSolver* solver);

/**
 * @brief Conducts linesearch to compute a \in (0, 1] such that
 * u + (a / f) * Du \in C. For QPs this calls exact_linesearch()
 * and for SOCPs this calls bisection_search()
 *
 * @param u Initial vector.
 * @param Du Search direction.
 * @param f Conservatism factor.
 * @param solver Pointer to solver.
 * @return Step-size.
 */
QOCOFloat linesearch(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                     QOCOSolver* solver);

/**
 * @brief Computes x = x - a*e where e is the cannonical vector for the cone.
 *
 * @param x Vector.
 * @param a Scaling factor.
 * @param l Dimension of LP cone.
 * @param nsoc Number of second-order cones.
 * @param q Dimensions of SOCs.
 */
void add_e(QOCOFloat* x, QOCOFloat a, QOCOInt l, QOCOInt nsoc, QOCOVectori* q);

#ifdef __cplusplus
}
#endif

#endif