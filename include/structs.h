/**
 * @file structs.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Defines all structs used by QOCO.
 */

#ifndef QOCO_STRUCTS_H
#define QOCO_STRUCTS_H

#include "definitions.h"
#include "timer.h"

/**
 * @brief Compressed sparse column format matrices.
 *
 */
typedef struct {
  /** Number of rows. */
  QOCOInt m;

  /** Number of columns. */
  QOCOInt n;

  /** Number of nonzero elements. */
  QOCOInt nnz;

  /** Row indices (length: nnz). */
  QOCOInt* i;

  /** Column pointers (length: n+1). */
  QOCOInt* p;

  /** Data (length: nnz). */
  QOCOFloat* x;

} QOCOCscMatrix;

/**
 * @brief SOCP problem data.
 *
 */
typedef struct {
  /** Quadratic cost term. */
  QOCOCscMatrix* P; //

  /** Linear cost term. */
  QOCOFloat* c;

  /** Affine equality constraint matrix. */
  QOCOCscMatrix* A;

  /** Transpose of A (used in Ruiz for fast row norm calculations of A). */
  QOCOCscMatrix* At;

  /** Affine equality constraint offset. */
  QOCOFloat* b;

  /** Conic constraint matrix. */
  QOCOCscMatrix* G;

  /** Transpose of G (used in Ruiz for fast row norm calculations of G). */
  QOCOCscMatrix* Gt;

  /** Mapping from A to At. */
  QOCOInt* AtoAt;

  /** Mapping from G to Gt. */
  QOCOInt* GtoGt;

  /** Conic constraint offset. */
  QOCOFloat* h;

  /** Dimension of non-negative orthant in cone C. */
  QOCOInt l;

  /** Number of second-order cones in C */
  QOCOInt nsoc;

  /** Dimension of each second-order cone (length of nsoc)*/
  QOCOInt* q;

  /** Number of primal variables. */
  QOCOInt n;

  /** Number of conic constraints. */
  QOCOInt m;

  /** Number of affine equality constraints. */
  QOCOInt p;

} QOCOProblemData;

/**
 * @brief QOCO solver settings
 *
 */
typedef struct {
  /** Maximum number of IPM iterations. */
  QOCOInt max_iters;

  /** Number of bisection iterations for linesearch. */
  QOCOInt bisect_iters;

  /** Number of Ruiz equilibration iterations. */
  QOCOInt ruiz_iters;

  /** Number of iterative refinement iterations performed. */
  QOCOInt iter_ref_iters;

  /** Static regularization parameter for KKT system. */
  QOCOFloat kkt_static_reg;

  /** Dynamic regularization parameter for KKT system. */
  QOCOFloat kkt_dynamic_reg;

  /** Absolute tolerance. */
  QOCOFloat abstol;

  /** Relative tolerance. */
  QOCOFloat reltol;

  /** Low tolerance stopping criteria. */
  QOCOFloat abstol_inacc;

  /** Low tolerance stopping criteria. */
  QOCOFloat reltol_inacc;

  /** 0 for quiet anything else for verbose. */
  unsigned char verbose;
} QOCOSettings;

/**
 * @brief Contains all data needed for constructing and modifying KKT matrix and
 * performing predictor-corrector step.
 *
 */
typedef struct {
  /** KKT matrix in CSC form. */
  QOCOCscMatrix* K;

  /** Diagonal of scaling matrix. */
  QOCOFloat* delta;

  /** Diagonal of scaling matrix. */
  QOCOFloat* Druiz;

  /** Diagonal of scaling matrix. */
  QOCOFloat* Eruiz;

  /** Diagonal of scaling matrix. */
  QOCOFloat* Fruiz;

  /** Inverse of Druiz. */
  QOCOFloat* Dinvruiz;

  /** Inverse of Eruiz. */
  QOCOFloat* Einvruiz;

  /** Inverse of Fruiz. */
  QOCOFloat* Finvruiz;

  /** Cost scaling factor. */
  QOCOFloat k;

  /** Inverse of cost scaling factor. */
  QOCOFloat kinv;

  /** Permutation vector. */
  QOCOInt* p;

  /** Inverse of permutation vector. */
  QOCOInt* pinv;

  /** Elimination tree for LDL factorization of K. */
  QOCOInt* etree;

  QOCOInt* Lnz;

  QOCOFloat* Lx;

  QOCOInt* Lp;

  QOCOInt* Li;

  QOCOFloat* D;

  QOCOFloat* Dinv;

  QOCOInt* iwork;

  unsigned char* bwork;

  QOCOFloat* fwork;

  /** RHS of KKT system. */
  QOCOFloat* rhs;

  /** Solution of KKT system. */
  QOCOFloat* xyz;

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff1;

  /** Buffer of size n + m + p. */
  QOCOFloat* xyzbuff2;

  /** Residual of KKT condition. */
  QOCOFloat* kktres;

  /** Mapping from elements in the Nesterov-Todd scaling matrix to elements in
   * the KKT matrix. */
  QOCOInt* nt2kkt;

  /** Mapping from elements on the main diagonal of the Nesterov-Todd scaling
   * matrices to elements in the KKT matrix. Used for regularization.*/
  QOCOInt* ntdiag2kkt;

  /** Mapping from elements in regularized P to elements in the KKT matrix. */
  QOCOInt* PregtoKKT;

  /** Indices of P->x that were added due to regularization. */
  QOCOInt* Pnzadded_idx;

  /** Number of elements of P->x that were added due to regularization. */
  QOCOInt Pnum_nzadded;

  /** Mapping from elements in At to elements in the KKT matrix. */
  QOCOInt* AttoKKT;

  /** Mapping from elements in Gt to elements in the KKT matrix. */
  QOCOInt* GttoKKT;

} QOCOKKT;

/**
 * @brief QOCO Workspace
 */
typedef struct {
  /** Contains SOCP problem data. */
  QOCOProblemData* data;

  /** Solve timer. */
  QOCOTimer solve_timer;

  /** Contains all data related to KKT system. */
  QOCOKKT* kkt;

  /** Iterate of primal variables. */
  QOCOFloat* x;

  /** Iterate of slack variables associated with conic constraint. */
  QOCOFloat* s;

  /** Iterate of dual variables associated with affine equality constraint. */
  QOCOFloat* y;

  /** Iterate of dual variables associated with conic constraint. */
  QOCOFloat* z;

  /** Gap (s'*z / m) */
  QOCOFloat mu;

  /** Newton Step-size */
  QOCOFloat a;

  /** Centering parameter */
  QOCOFloat sigma;

  /** Number of nonzeros in upper triangular part of Nesterov-Todd Scaling. */
  QOCOInt Wnnz;

  /** Number of nonzeros in full Nesterov-Todd Scaling. */
  QOCOInt Wnnzfull;

  /** Upper triangular part of Nesterov-Todd Scaling */
  QOCOFloat* W;

  /** Full Nesterov-Todd Scaling */
  QOCOFloat* Wfull;

  /** Upper triangular part of inverse of Nesterov-Todd Scaling */
  QOCOFloat* Winv;

  /** Full inverse of Nesterov-Todd Scaling */
  QOCOFloat* Winvfull;

  /** Nesterov-Todd Scaling squared */
  QOCOFloat* WtW;

  /** Scaled variables. */
  QOCOFloat* lambda;

  /** Temporary array needed in Nesterov-Todd scaling calculations. Length of
   * max(q). */
  QOCOFloat* sbar;

  /** Temporary array needed in Nesterov-Todd scaling calculations. Length of
   * max(q). */
  QOCOFloat* zbar;

  /** Temporary variable of length n. */
  QOCOFloat* xbuff;

  /** Temporary variable of length p. */
  QOCOFloat* ybuff;

  /** Temporary variable of length m. */
  QOCOFloat* ubuff1;

  /** Temporary variable of length m. */
  QOCOFloat* ubuff2;

  /** Temporary variable of length m. */
  QOCOFloat* ubuff3;

  /** Search direction for slack variables. Length of m. */
  QOCOFloat* Ds;

} QOCOWorkspace;

typedef struct {
  /**Primal solution. */
  QOCOFloat* x;

  /**Slack variable for conic constraints. */
  QOCOFloat* s;

  /**Dual variables for affine equality constraints. */
  QOCOFloat* y;

  /**Dual variables for conic constraints. */
  QOCOFloat* z;

  /**Number of iterations. */
  QOCOInt iters;

  /**Setup time. */
  QOCOFloat setup_time_sec;

  /**Solve time. */
  QOCOFloat solve_time_sec;

  /**Optimal objective value. */
  QOCOFloat obj;

  /** Primal residual. */
  QOCOFloat pres;

  /** Dual residual. */
  QOCOFloat dres;

  /** Duality gap. */
  QOCOFloat gap;

  /**Solve status. */
  QOCOInt status;

} QOCOSolution;

/**
 * @brief QOCO Solver struct. Contains all information about the state of the
 * solver.
 *
 */
typedef struct {
  /** Solver settings. */
  QOCOSettings* settings;

  /** Solver workspace. */
  QOCOWorkspace* work;

  /** Solution struct. */
  QOCOSolution* sol;

} QOCOSolver;

#endif /* #ifndef QOCO_STRUCTS_H */