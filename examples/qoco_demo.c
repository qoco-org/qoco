#include "qoco.h"

int main()
{
  QOCOInt p = 2;     // Number of affine equality constraints (rows of A).
  QOCOInt m = 6;     // Number of conic constraints (rows of G).
  QOCOInt n = 6;     // Number of optimization variables.
  QOCOInt l = 3;     // Dimension of non-negative orthant.
  QOCOInt nsoc = 1;  // Number of second-order cones.
  QOCOInt q[] = {3}; // Dimension of second-order cones.

  QOCOFloat Px[] = {1, 2, 3, 4, 5, 6};  // Data for upper triangular part of P.
  QOCOInt Pnnz = 6;                     // Number of nonzero elements.
  QOCOInt Pp[] = {0, 1, 2, 3, 4, 5, 6}; // Column pointers.
  QOCOInt Pi[] = {0, 1, 2, 3, 4, 5};    // Row indices.

  QOCOFloat Ax[] = {1, 1, 1, 2};
  QOCOInt Annz = 4;
  QOCOInt Ap[] = {0, 1, 3, 4, 4, 4, 4};
  QOCOInt Ai[] = {0, 0, 1, 1};

  QOCOFloat Gx[] = {-1, -1, -1, -1, -1, -1};
  QOCOInt Gnnz = 6;
  QOCOInt Gp[] = {0, 1, 2, 3, 4, 5, 6};
  QOCOInt Gi[] = {0, 1, 2, 3, 4, 5};

  QOCOFloat c[] = {1, 2, 3, 4, 5, 6};
  QOCOFloat b[] = {1, 2};
  QOCOFloat h[] = {0, 0, 0, 0, 0, 0};

  // Allocate storage for data matrices.
  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));

  // Set data matrices.
  qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));

  // Set default settings.
  set_default_settings(settings);
  settings->verbose = 1;

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

  // Set up solver.
  QOCOInt exit =
      qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);

  // Solve problem.
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  // Free allocated memory.
  qoco_cleanup(solver);
  free(P);
  free(A);
  free(G);
}