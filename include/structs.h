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
#include "timer.h"

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

  /** Transpose of A (used in Ruiz for fast row norm calculations of A). */
  QCOSCscMatrix* At;

  /** Affine equality constraint offset. */
  QCOSFloat* b;

  /** Conic constraint matrix. */
  QCOSCscMatrix* G;

  /** Transpose of G (used in Ruiz for fast row norm calculations of G). */
  QCOSCscMatrix* Gt;

  /** Conic constraint offset. */
  QCOSFloat* h;

  /** Dimension of non-negative orthant in cone C. */
  QCOSInt l;

  /** Number of second-order cones in C */
  QCOSInt nsoc;

  /** Dimension of each second-order cone (length of nsoc)*/
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

  /** Number of bisection iterations for linesearch. */
  QCOSInt bisection_iters;

  /** Number of Ruiz equilibration iterations. */
  QCOSInt ruiz_iters;

  /** Number of iterative refinement iterations performed. */
  QCOSInt iterative_refinement_iterations;

  /** Absolute tolerance. */
  QCOSFloat abstol;

  /** Relative tolerance. */
  QCOSFloat reltol;

  /** Low tolerance stopping criteria. */
  QCOSFloat abstol_inaccurate;

  /** Low tolerance stopping criteria. */
  QCOSFloat reltol_inaccurate;

  /** Static regularization parameter for KKT system. */
  QCOSFloat static_reg;

  /** Dynamic regularization parameter for KKT system. */
  QCOSFloat dyn_reg;

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

  /** Diagonal of scaling matrix. */
  QCOSFloat* delta;

  /** Diagonal of scaling matrix. */
  QCOSFloat* Druiz;

  /** Diagonal of scaling matrix. */
  QCOSFloat* Eruiz;

  /** Diagonal of scaling matrix. */
  QCOSFloat* Fruiz;

  /** Inverse of Druiz. */
  QCOSFloat* Dinvruiz;

  /** Inverse of Eruiz. */
  QCOSFloat* Einvruiz;

  /** Inverse of Fruiz. */
  QCOSFloat* Finvruiz;

  /** Cost scaling factor. */
  QCOSFloat k;

  /** Inverse of cost scaling factor. */
  QCOSFloat kinv;

  /** Permutation vector. */
  QCOSInt* p;

  /** Inverse of permutation vector. */
  QCOSInt* pinv;

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

  /** Buffer of size n + m + p. */
  QCOSFloat* xyzbuff;

  /** Residual of KKT condition. */
  QCOSFloat* kktres;

  /** Mapping from elements in the Nesterov-Todd scaling matrix to elements in
   * the KKT matrix. */
  QCOSInt* nt2kkt;

  /** Mapping from elements on the main diagonal of the Nesterov-Todd scaling
   * matrices to elements in the KKT matrix. Used for regularization.*/
  QCOSInt* ntdiag2kkt;

  /** Mapping from elements in regularized P to elements in the KKT matrix. */
  QCOSInt* PregtoKKT;

  /** Indices of P->x that were added due to regularization. */
  QCOSInt* Pnzadded_idx;

  /** Number of elements of P->x that were added due to regularization. */
  QCOSInt Pnum_nzadded;

  /** Mapping from elements in A to elements in the KKT matrix. */
  QCOSInt* AtoKKT;

  /** Mapping from elements in G to elements in the KKT matrix. */
  QCOSInt* GtoKKT;

} QCOSKKT;

/**
 * @brief QCOS Workspace
 */
typedef struct {
  /** Contains SOCP problem data. */
  QCOSProblemData* data;

  /** Solve timer. */
  QCOSTimer solve_timer;

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

  /** Newton Step-size */
  QCOSFloat a;

  /** Centering parameter */
  QCOSFloat sigma;

  /** Number of nonzeros in Nesterov-Todd Scaling. */
  QCOSInt Wnnz;

  /** Upper triangular part of Nesterov-Todd Scaling */
  QCOSFloat* W;

  /** Full Nesterov-Todd Scaling */
  QCOSFloat* Wfull;

  /** Upper triangular part of inverse of Nesterov-Todd Scaling */
  QCOSFloat* Winv;

  /** Full inverse of Nesterov-Todd Scaling */
  QCOSFloat* Winvfull;

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

  /** Temporary variable of length n. */
  QCOSFloat* xbuff;

  /** Temporary variable of length p. */
  QCOSFloat* ybuff;

  /** Temporary variable of length m. */
  QCOSFloat* ubuff1;

  /** Temporary variable of length m. */
  QCOSFloat* ubuff2;

  /** Temporary variable of length m. */
  QCOSFloat* ubuff3;

  /** Search direction for slack variables. Length of m. */
  QCOSFloat* Ds;

} QCOSWorkspace;

typedef struct {
  /* Primal solution. */
  QCOSFloat* x;

  /* Slack variable for conic constraints. */
  QCOSFloat* s;

  /* Dual variables for affine equality constraints. */
  QCOSFloat* y;

  /* Dual variables for conic constraints. */
  QCOSFloat* z;

  /* Number of iterations. */
  QCOSInt iters;

  /* Setup time. */
  QCOSFloat setup_time_sec;

  /* Solve time. */
  QCOSFloat solve_time_sec;

  /* Optimal objective value. */
  QCOSFloat obj;

  /** Primal residual. */
  QCOSFloat pres;

  /** Dual residual. */
  QCOSFloat dres;

  /** Duality gap. */
  QCOSFloat gap;

  /* Solve status. */
  QCOSInt status;

} QCOSSolution;

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

  /* Solution. */
  QCOSSolution* sol;

} QCOSSolver;

#endif /* #ifndef STRUCTS_H */