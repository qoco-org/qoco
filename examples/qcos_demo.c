#include "qcos.h"

int main()
{
  QCOSInt p = 2;     // Number of affine equality constraints (rows of A).
  QCOSInt m = 6;     // Number of conic constraints (rows of G).
  QCOSInt n = 6;     // Number of optimization variables.
  QCOSInt l = 3;     // Dimension of non-negative orthant.
  QCOSInt nsoc = 1;  // Number of second-order cones.
  QCOSInt q[] = {3}; // Dimension of second-order cones.

  QCOSFloat Px[] = {1, 2, 3, 4, 5, 6};  // Data for upper triangular part of P.
  QCOSInt Pnnz = 6;                     // Number of nonzero elements.
  QCOSInt Pp[] = {0, 1, 2, 3, 4, 5, 6}; // Column pointers.
  QCOSInt Pi[] = {0, 1, 2, 3, 4, 5};    // Row indices.

  QCOSFloat Ax[] = {1, 1, 1, 2};
  QCOSInt Annz = 4;
  QCOSInt Ap[] = {0, 1, 3, 4, 4, 4, 4};
  QCOSInt Ai[] = {0, 0, 1, 1};

  QCOSFloat Gx[] = {-1, -1, -1, -1, -1, -1};
  QCOSInt Gnnz = 6;
  QCOSInt Gp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Gi[] = {0, 1, 2, 3, 4, 5};

  QCOSFloat c[] = {1, 2, 3, 4, 5, 6};
  QCOSFloat b[] = {1, 2};
  QCOSFloat h[] = {0, 0, 0, 0, 0, 0};

  // Allocate storage for data matrices.
  QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

  // Set data matrices.
  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));

  // Set default settings.
  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

  // Set up solver.
  QCOSInt exit =
      qcos_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);

  // Solve problem.
  if (exit == QCOS_NO_ERROR) {
    exit = qcos_solve(solver);
  }

  // Free allocated memory.
  qcos_cleanup(solver);
  free(P);
  free(A);
  free(G);
}