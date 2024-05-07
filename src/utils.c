/**
 * @file utils.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 */

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

void print_arrayf(QCOSFloat* x, QCOSInt n)
{
  printf("{");
  for (QCOSInt i = 0; i < n; ++i) {
    printf("%f", x[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("}\n");
}

void print_arrayi(QCOSInt* x, QCOSInt n)
{
  printf("{");
  for (QCOSInt i = 0; i < n; ++i) {
    printf("%d", x[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("}\n");
}

void print_header()
{
  // clang-format off
    printf("\n");
    printf("+----------------------------------------------------------------------+\n");
    printf("|    ..:: QCOS - Quadratic Objective Conic Optimization Solver ::..    |\n");
    printf("|                      (c) Govind M. Chari, 2024                       |\n");
    printf("|           University of Washington Autonomous Controls Lab           |\n");
    printf("+--------+-----------+------------+------------+-----------+-----------+\n");
    printf("|  Iter  |   Pcost   |    Pres    |    Dres    |     Mu    |    Step   |\n");
    printf("+--------+-----------+------------+------------+-----------+-----------+\n");
  // clang-format on
}

unsigned char check_stopping(QCOSSolver* solver)
{
  if (solver->work->mu < solver->settings->abstol) {
    return 1;
  }
  return 0;
}

void copy_solution(QCOSSolver* solver)
{
  QCOSWorkspace* work = solver->work;
  QCOSProblemData* data = solver->work->data;

  // Copy optimization variables
  copy_arrayf(solver->work->x, solver->sol->x, solver->work->data->n);
  copy_arrayf(solver->work->s, solver->sol->s, solver->work->data->m);
  copy_arrayf(solver->work->y, solver->sol->y, solver->work->data->p);
  copy_arrayf(solver->work->z, solver->sol->z, solver->work->data->m);

  // Compute objective.
  QCOSFloat obj = dot(work->x, data->c, data->n);
  USpMv(data->P, work->x, work->xbuff);
  obj += 0.5 * (dot(work->xbuff, work->x, data->n));
  solver->sol->obj = obj;
}

void log_iter(QCOSSolver* solver)
{
  QCOSWorkspace* work = solver->work;
  QCOSProblemData* data = solver->work->data;

  // Compute objective.
  QCOSFloat obj = dot(work->x, data->c, data->n);
  USpMv(data->P, work->x, work->xbuff);
  obj += 0.5 * (dot(work->xbuff, work->x, data->n));

  QCOSFloat dres = norm_inf(work->kkt->kktres, data->n);
  QCOSFloat pres = norm_inf(&work->kkt->kktres[data->n], data->n);

  // clang-format off
  printf("|  %2d    | %+.2e | %+.3e | %+.3e | %+.2e |   %.3f   |\n",
         solver->sol->iters, obj, pres, dres, work->mu, work->a);
  printf("+--------+-----------+------------+------------+-----------+-----------+\n");
  // clang-format on
}

void print_footer(QCOSSolution* solution)
{
  // clang-format off
  printf("|                              Converged.                              |\n");
  printf("+----------------------------------------------------------------------+\n");
  printf("\n");
  // clang-format on

  printf("\nstatus:                solved\n");
  printf("number of iterations:  %d\n", solution->iters);
  printf("optimal objective:     %+.3f\n", solution->obj);
  printf("run time:              %.2es\n", solution->solve_time_ms);
}