#ifndef STRUCTS_H
#define STRUCTS_H

#include "definitions.h"

/**
 * @brief Compressed sparse column format matrices.
 *
 */
typedef struct {
  QCOSInt m;    // number of rows
  QCOSInt n;    // number of columns
  QCOSInt nnz;  // number of nonzero elements
  QCOSInt* p;   // column pointers (length n+1)
  QCOSInt* i;   // row indices (length nnz)
  QCOSFloat* x; // data (length nnz)
} QCOSCscMatrix;

/**
 * @brief Internal QCOS vector.
 *
 */
typedef struct {
  QCOSFloat* x; // Data
  QCOSInt n;    // Length of vector
} QCOSVector;

/**
 * @brief SOCP problem data.
 *
 */
typedef struct {
  QCOSCscMatrix* P; // quadratic cost term
  QCOSVector* c;    // linear cost term
  QCOSCscMatrix* A; // affine equality constraint matrix
  QCOSVector* b;    // affine equality constraint offset
  QCOSCscMatrix* G; // conic constraint matrix
  QCOSVector* h;    // conic constraint offset
  QCOSInt l;        // dimension of non-negative orthant of C
  QCOSInt ncones;   // number of second-order cones in C
  QCOSInt* q;       // dimension of each second-order cone (length ncones)
  QCOSInt n;        // Number of decision variables
  QCOSInt m;        // Number of conic constraints
  QCOSInt p;        // Number of affine equality constraints
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
 * @brief QCOS Workspace
 *
 */
typedef struct {
  QCOSProblemData* data;
  QCOSInt* nt2kkt;
  QCOSVector* xyz;
  QCOSVector* x;
  QCOSVector* s;
  QCOSVector* y;
  QCOSVector* z;
  QCOSCscMatrix* kkt;
} QCOSWorkspace;

/**
 * @brief QCOS Solver struct. Contains all information about the state of the
 * solver.
 *
 */
typedef struct {
  QCOSSettings* settings;
  QCOSWorkspace* work;
} QCOSSolver;

#endif /* #ifndef STRUCTS_H */