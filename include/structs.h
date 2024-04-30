/**
 * @file structs.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines all structs used by QCOS.
 */

#ifndef STRUCTS_H
#define STRUCTS_H

#include "definitions.h"

/**
 * @brief Compressed sparse column format matrices.
 *
 */
typedef struct {
  /** Number of rows. */
  QCOSInt m;

  /** Number of columns. */
  QCOSInt n;

  /** Number of nonzero elements. */
  QCOSInt nnz;

  /** Row indices (length: nnz). */
  QCOSInt* i;

  /** Column pointers (length: n+1). */
  QCOSInt* p;

  /** Data (length: nnz). */
  QCOSFloat* x;

} QCOSCscMatrix;

/**
 * @brief SOCP problem data.
 *
 */
typedef struct {
  /** Quadratic cost term. */
  QCOSCscMatrix* P; //

  /** Linear cost term. */
  QCOSFloat* c;

  /** Affine equality constraint matrix. */
  QCOSCscMatrix* A;

  /** Affine equality constraint offset. */
  QCOSFloat* b;

  /** Conic constraint matrix. */
  QCOSCscMatrix* G;

  /** Conic constraint offset. */
  QCOSFloat* h;

  /** Dimension of non-negative orthant in cone C. */
  QCOSInt l;

  /** Number of second-order cones in C */
  QCOSInt ncones;

  /** Dimension of each second-order cone (length of ncones)*/
  QCOSInt* q;

  /** Number of primal variables. */
  QCOSInt n;

  /** Number of conic constraints. */
  QCOSInt m;

  /** Number of affine equality constraints. */
  QCOSInt p;

} QCOSProblemData;

/**
 * @brief QCOS solver settings
 *
 */
typedef struct {
  /** Maximum number of IPM iterations. */
  QCOSInt max_iters;

  /** Maximum number of bisection iterations for linesearch. */
  QCOSInt max_iter_bisection;

  /** Absolute tolerance. */
  QCOSFloat abstol;

  /** Relative tolerance. */
  QCOSFloat reltol;

  /** 0 for quiet anything else for verbose. */
  unsigned char verbose;
} QCOSSettings;

/**
 * @brief Contains all data needed for constructing and modifying KKT matrix and
 * performing predictor-corrector step.
 *
 */
typedef struct {
  /** KKT matrix in CSC form. */
  QCOSCscMatrix* K;

  /** Elimination tree for LDL factorization of K. */
  QCOSInt* etree;

  QCOSInt* Lnz;

  QCOSFloat* Lx;

  QCOSInt* Lp;

  QCOSInt* Li;

  QCOSFloat* D;

  QCOSFloat* Dinv;

  QCOSInt* iwork;

  unsigned char* bwork;

  QCOSFloat* fwork;

  /** RHS of KKT system. */
  QCOSFloat* rhs;

  /** Solution of KKT system. */
  QCOSFloat* xyz;

  /** Mapping from elements in the Nesterov-Todd scaling matrix to elements in
   * the KKT matrix. */
  QCOSInt* nt2kkt;

} QCOSKKT;

/**
 * @brief QCOS Workspace
 */
typedef struct {
  /** Contains SOCP problem data. */
  QCOSProblemData* data;

  /** Contains all data related to KKT system. */
  QCOSKKT* kkt;

  /** Iterate of primal variables. */
  QCOSFloat* x;

  /** Iterate of slack variables associated with conic constraint. */
  QCOSFloat* s;

  /** Iterate of dual variables associated with affine equality constraint. */
  QCOSFloat* y;

  /** Iterate of dual variables associated with conic constraint. */
  QCOSFloat* z;

  /** Gap (s'*z / m) */
  QCOSFloat mu;

  /** Number of nonzeros in Nesterov-Todd Scaling. */
  QCOSInt Wnnz;

  /** Upper triangular part of Nesterov-Todd Scaling */
  QCOSFloat* W;

  /** Full Nesterov-Todd Scaling */
  QCOSFloat* Wfull;

  /** Nesterov-Todd Scaling squared */
  QCOSFloat* WtW;

  /** Scaled variables. */
  QCOSFloat* lambda;

  /** Temporary array needed in Nesterov-Todd scaling calculations. Length of
   * max(q). */
  QCOSFloat* sbar;

  /** Temporary array needed in Nesterov-Todd scaling calculations. Length of
   * max(q). */
  QCOSFloat* zbar;

  /** Temporary variable of length m. */
  QCOSFloat* ubuff1;

  /** Temporary variable of length m. */
  QCOSFloat* ubuff2;

  /** Search direction for slack variables. Length of m. */
  QCOSFloat* Ds;

} QCOSWorkspace;

/**
 * @brief QCOS Solver struct. Contains all information about the state of the
 * solver.
 *
 */
typedef struct {
  /** Solver settings. */
  QCOSSettings* settings;

  /** Solver workspace. */
  QCOSWorkspace* work;

} QCOSSolver;

#endif /* #ifndef STRUCTS_H */