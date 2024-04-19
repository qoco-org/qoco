#include "utils.h"

void print_qcos_csc_matrix(QCOSCscMatrix* M)
{
  printf("\nPrinting CSC Matrix:\n");
  printf("m: %d\n", M->m);
  printf("n: %d\n", M->n);
  printf("nnz: %d\n", M->nnz);
  printf("Data: {");
  for (QCOSInt i = 0; i < M->nnz; ++i) {
    printf("%.3f", M->x[i]);
    if (i != M->nnz - 1) {
      printf(",");
    }
  }
  printf("}\n");

  printf("Row Indices: {");
  for (QCOSInt i = 0; i < M->nnz; ++i) {
    printf("%d", M->i[i]);
    if (i != M->nnz - 1) {
      printf(",");
    }
  }
  printf("}\n");

  printf("Column Pointers: {");
  for (QCOSInt i = 0; i < M->n + 1; ++i) {
    printf("%d", M->p[i]);
    if (i != M->n) {
      printf(",");
    }
  }
  printf("}\n");
}

void print_qcos_vector(QCOSVector* V)
{
  printf("\nPrinting Vector:\n");
  printf("n: %d\n", V->n);
  printf("Data: {");
  for (QCOSInt i = 0; i < V->n; ++i) {
    printf("%.3f", V->x[i]);
    if (i != V->n - 1) {
      printf(",");
    }
  }
  printf("}\n");
}

void print_header()
{
  // clang-format off
    printf("\n");
    printf("+--------------------------------------------------------------------------------------+\n");
    printf("|            ..:: QCOS - Quadratic Objective Conic Optimization Solver ::..            |\n");
    printf("|                              (c) Govind M. Chari, 2024                               |\n");
    printf("|                   University of Washington Autonomous Controls Lab                   |\n");
    printf("+-----------+----------------+-------------+-----------------+------------+------------+\n");
    printf("| Iteration |   Primal cost  |     Gap     | Primal residual |     Mu     |    Step    |\n");
    printf("+-----------+----------------+-------------+-----------------+------------+------------+\n");
  // clang-format on
}