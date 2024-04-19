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
 * @brief Internal QCOS vector.
 *
 */
typedef struct {
  /** Data. */
  QCOSFloat* x;

  /** Length of vector. */
  QCOSInt n;

} QCOSVector;

/**
 * @brief SOCP problem data.
 *
 */
typedef struct {
  /** Quadratic cost term. */
  QCOSCscMatrix* P; //

  /** Linear cost term. */
  QCOSVector* c;

  /** Affine equality constraint matrix. */
  QCOSCscMatrix* A;

  /** Affine equality constraint offset. */
  QCOSVector* b;

  /** Conic constraint matrix. */
  QCOSCscMatrix* G;

  /** Conic constraint offset. */
  QCOSVector* h;

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
  QCOSFloat tol;
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

  QCOSInt* etree;

  QCOSInt* Lnz;

  QCOSInt* Lp;

  QCOSInt* Li;

  QCOSFloat* Lx;

  QCOSFloat* D;

  QCOSFloat* Dinv;

  QCOSInt* iwork;

  unsigned char* bwork;

  QCOSFloat* fwork;

  /** Temporary variable for rhs of KKT system. */
  QCOSVector* rhs;

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
  QCOSVector* x;

  /** Iterate of slack variables associated with conic constraint. */
  QCOSVector* s;

  /** Iterate of dual variables associated with affine equality constraint. */
  QCOSVector* y;

  /** Iterate of dual variables associated with conic constraint. */
  QCOSVector* z;

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