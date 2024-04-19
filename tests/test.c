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

  QCOSSolver* solver = qcos_setup(P, c, A, b, G, h, l, ncones, q, settings);

  print_qcos_csc_matrix(solver->work->kkt->K);

  // for (int i = 0; i < m; ++i) {
  //   printf("%d, ", solver->work->kkt->nt2kkt[i]);
  // }
  // QCOSFloat u[] = {0, 0, 0, 4, 5, 6};
  // bring2cone(u, solver->work->data);
  // QCOSFloat res = cone_residual(u, solver->work->data);
  // printf("Residual: %f\n", res);
  // QCOSFloat v[] = {1, 2, 3, 4, 5, 6};
  // QCOSFloat prod[] = {0, 0, 0, 0, 0, 0};
  // cone_product(u, v, prod, solver->work->data);
  // for (int i = 0; i < 6; ++i) {
  //   printf("%f, ", u[i]);
  // }

  if (solver) {
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