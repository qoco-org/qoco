#include "qcos.h"
#include <stdio.h>

int main()
{

  QCOSInt p = 2;
  QCOSInt m = 6;
  QCOSInt n = 6;
  QCOSInt l = 3;
  QCOSInt ncones = 1;

  QCOSFloat Px[] = {1, 2, 3, 4, 5, 6};
  QCOSInt Pnnz = 6;
  QCOSInt Pp[] = {0, 1, 2, 3, 4, 5, 6};
  QCOSInt Pi[] = {0, 1, 2, 3, 4, 5};

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
  QCOSInt q[] = {3};

  QCOSCscMatrix* P = malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* A = malloc(sizeof(QCOSCscMatrix));
  QCOSCscMatrix* G = malloc(sizeof(QCOSCscMatrix));

  qcos_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  qcos_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  qcos_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);

  QCOSSettings* settings = malloc(sizeof(QCOSSettings));

  set_default_settings(settings);
  settings->verbose = 1;

  QCOSSolver* solver = qcos_setup(P, c, A, b, G, h, l, ncones, q, settings);

  QCOSInt exit = qcos_solve(solver);

  // print_arrayf(solver->work->lambda, m);

  // print_arrayf(solver->work->kkt->xyz, m + n + p);

  // QCOSFloat a[] = {2, 3, 4};
  // QCOSFloat z[] = {6, 7, 8};
  // QCOSFloat ans[] = {0, 0, 0};
  // soc_division(a, z, ans, 3);
  // print_arrayf(ans, 3);

  // printf("x: ");
  // print_arrayf(solver->work->x, n);
  // printf("s: ");
  // print_arrayf(solver->work->s, m);
  // printf("y: ");
  // print_arrayf(solver->work->y, p);
  // printf("z: ");
  // print_arrayf(solver->work->z, m);

  // // print_qcos_csc_matrix(solver->work->kkt->K);

  // compute_kkt_residual(solver->work);

  // print_arrayf(solver->work->kkt->rhs, n + m + p);

  // print_qcos_csc_matrix(solver->work->kkt->K);

  if (solver && !exit) {
    qcos_cleanup(solver);
    printf("Success");
  }
  else {
    printf("Error");
  }

  free(P);
  free(A);
  free(G);

  return 0;
}