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
  printf("+-----------------------------------------------------------------------------------+\n");
  printf("|          ..:: QCOS - Quadratic Objective Conic Optimization Solver ::..           |\n");
  printf("|                            (c) Govind M. Chari, 2024                              |\n");
  printf("|                 University of Washington Autonomous Controls Lab                  |\n");
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  printf("|  Iter  |   Pcost   |    Pres    |    Dres    |     Gap    |     Mu    |    Step   |\n");
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  // clang-format on
}

void log_iter(QCOSSolver* solver)
{
  // clang-format off
  printf("|  %2d    | %+.2e | %+.3e | %+.3e | %+.3e | %+.2e |   %.3f   |\n",
         solver->sol->iters, solver->sol->obj, solver->sol->pres, solver->sol->dres, solver->sol->gap, solver->work->mu, solver->work->a);
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  // clang-format on
}

void print_footer(QCOSSolution* solution, enum qcos_solve_status status)
{
  printf("\n");
  printf("status:                %s\n", QCOS_SOLVE_STATUS_MESSAGE[status]);
  printf("number of iterations:  %d\n", solution->iters);
  printf("optimal objective:     %+.3f\n", solution->obj);
  printf("setup time:            %.2e ms\n", solution->setup_time_ms);
  printf("solve time:            %.2e ms\n", solution->solve_time_ms);
  printf("\n");
}

unsigned char check_stopping(QCOSSolver* solver)
{
  QCOSWorkspace* work = solver->work;
  QCOSProblemData* data = solver->work->data;
  QCOSFloat eabs = solver->settings->abstol;
  QCOSFloat erel = solver->settings->reltol;

  QCOSFloat ctx = dot(data->c, work->x, data->n);
  QCOSFloat bty = dot(data->b, work->y, data->p);
  QCOSFloat htz = dot(data->h, work->z, data->m);
  QCOSFloat ctxabs = qcos_abs(ctx);
  QCOSFloat btyabs = qcos_abs(bty);
  QCOSFloat htzabs = qcos_abs(htz);
  QCOSFloat binf = norm_inf(data->b, data->p);
  QCOSFloat sinf = norm_inf(work->s, data->m);
  QCOSFloat cinf = norm_inf(data->c, data->n);
  QCOSFloat hinf = norm_inf(data->h, data->m);

  // Compute objective.
  QCOSFloat obj = dot(work->x, data->c, data->n);
  USpMv(data->P, work->x, work->xbuff);
  obj += 0.5 * (dot(work->xbuff, work->x, data->n));
  solver->sol->obj = obj;

  // Compute ||A^T * y||_\infty
  SpMtv(data->A, work->y, work->xbuff);
  QCOSFloat Atyinf = norm_inf(work->xbuff, data->n);

  // Compute ||G^T * z||_\infty
  SpMtv(data->G, work->z, work->xbuff);
  QCOSFloat Gtzinf = norm_inf(work->xbuff, data->n);

  // Compute ||P * x||_\infty
  SpMv(data->P, work->x, work->xbuff);
  QCOSFloat xtPx = dot(work->xbuff, work->x, data->n);
  QCOSFloat xtPxabs = qcos_abs(xtPx);
  QCOSFloat Pxinf = norm_inf(work->xbuff, data->n);

  // Compute ||A * x||_\infty
  SpMv(data->A, work->x, work->ybuff);
  QCOSFloat Axinf = norm_inf(work->ybuff, data->p);

  // Compute ||G * x||_\infty
  SpMv(data->G, work->x, work->ubuff1);
  QCOSFloat Gxinf = norm_inf(work->ubuff1, data->m);

  // Compute primal residual.
  QCOSFloat pres = norm_inf(&work->kkt->kktres[data->n], data->n);
  solver->sol->pres = pres;

  // COmpute dual residual.
  QCOSFloat dres = norm_inf(work->kkt->kktres, data->n);
  solver->sol->dres = dres;

  // Compute duality gap.
  QCOSFloat gap = xtPx + ctx + bty + htz;
  gap = qcos_abs(gap);
  solver->sol->gap = gap;

  // Compute max{Axinf, binf, Gxinf, hinf, sinf}.
  QCOSFloat pres_rel = qcos_max(Axinf, binf);
  pres_rel = qcos_max(pres_rel, Gxinf);
  pres_rel = qcos_max(pres_rel, hinf);
  pres_rel = qcos_max(pres_rel, sinf);

  // Compute max{Pxinf, Atyinf, Gtzinf, cinf}.
  QCOSFloat dres_rel = qcos_max(Pxinf, Atyinf);
  dres_rel = qcos_max(pres_rel, Gtzinf);
  dres_rel = qcos_max(pres_rel, cinf);

  // Compute max{xtPxabs, ctxabs, btyabs, htzabs}.
  QCOSFloat gap_rel = qcos_max(xtPxabs, ctxabs);
  gap_rel = qcos_max(gap_rel, btyabs);
  gap_rel = qcos_max(gap_rel, htzabs);

  if (solver->work->mu < eabs && pres < eabs + erel * pres_rel &&
      dres < eabs + erel * dres_rel && gap < eabs + erel * gap_rel) {
    return 1;
  }
  return 0;
}

void copy_solution(QCOSSolver* solver)
{
  // Copy optimization variables.
  copy_arrayf(solver->work->x, solver->sol->x, solver->work->data->n);
  copy_arrayf(solver->work->s, solver->sol->s, solver->work->data->m);
  copy_arrayf(solver->work->y, solver->sol->y, solver->work->data->p);
  copy_arrayf(solver->work->z, solver->sol->z, solver->work->data->m);
}