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
    printf("%.17g", M->x[i]);
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
    printf("%.17g", x[i]);
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

void print_header(QCOSSolver* solver)
{
  QCOSProblemData* data = solver->work->data;
  QCOSSettings* settings = solver->settings;

  // clang-format off
  printf("\n");
  printf("+-------------------------------------------------------+\n");
  printf("|  QCOS - Quadratic Objective Conic Optimization Solver |\n");
  printf("|             (c) Govind M. Chari, 2024                 |\n");
  printf("|    University of Washington Autonomous Controls Lab   |\n");
  printf("+-------------------------------------------------------+\n");
  printf("| Problem Data:                                         |\n");
  printf("|     variables:        %-9d                       |\n", data->n);
  printf("|     constraints:      %-9d                       |\n", data->l + data->p + data->nsoc);
  printf("|     eq constraints:   %-9d                       |\n", data->p);
  printf("|     ineq constraints: %-9d                       |\n", data->l);
  printf("|     soc constraints:  %-9d                       |\n", data->l);
  printf("|     nnz(P):           %-9d                       |\n", data->P->nnz);
  printf("|     nnz(A):           %-9d                       |\n", data->A->nnz);
  printf("|     nnz(G):           %-9d                       |\n", data->G->nnz);
  printf("| Solver Settings:                                      |\n");
  printf("|     max_iter: %-3d abstol: %3.2e reltol: %3.2e   |\n", settings->max_iters, settings->abstol, settings->reltol);
  printf("|     abstol_inacc: %3.2e reltol_inacc: %3.2e     |\n", settings->abstol_inaccurate, settings->reltol_inaccurate);
  printf("|     bisection_iters: %-2d iterative_refine_iters: %-2d    |\n", settings->bisection_iters, settings->iterative_refinement_iterations);
  printf("|     ruiz_iters: %-2d static_regularization: %3.2e    |\n", settings->ruiz_iters, settings->reg);
  printf("+-------------------------------------------------------+\n");
  printf("\n");
  // printf("+-----------------------------------------------------------------------------------+\n");
  // printf("|          ..:: QCOS - Quadratic Objective Conic Optimization Solver ::..           |\n");
  // printf("|                            (c) Govind M. Chari, 2024                              |\n");
  // printf("|                 University of Washington Autonomous Controls Lab                  |\n");
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  printf("|  Iter  |   Pcost   |    Pres    |    Dres    |     Gap    |     Mu    |    Step   |\n");
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  // clang-format on
}

void log_iter(QCOSSolver* solver)
{
  // clang-format off
  printf("|   %2d   | %+.2e | %+.3e | %+.3e | %+.3e | %+.2e |   %.3f   |\n",
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
  printf("solve time:            %.2e sec\n", solution->solve_time_sec);
  printf("\n");
}

unsigned char check_stopping(QCOSSolver* solver)
{
  QCOSWorkspace* work = solver->work;
  QCOSProblemData* data = solver->work->data;
  QCOSFloat eabs = solver->settings->abstol;
  QCOSFloat erel = solver->settings->reltol;
  QCOSFloat eabsinacc = solver->settings->abstol_inaccurate;
  QCOSFloat erelinacc = solver->settings->reltol_inaccurate;

  ew_product(work->kkt->Einvruiz, data->b, work->ybuff, data->p);
  QCOSFloat binf = data->p > 0 ? inf_norm(work->ybuff, data->p) : 0;

  ew_product(work->kkt->Fruiz, work->s, work->ubuff1, data->m);
  QCOSFloat sinf = data->m > 0 ? inf_norm(work->ubuff1, data->m) : 0;

  ew_product(work->kkt->Fruiz, work->z, work->ubuff2, data->m);
  QCOSFloat zinf = data->m > 0 ? inf_norm(work->ubuff2, data->m) : 0;

  ew_product(work->kkt->Dinvruiz, work->x, work->xbuff, data->n);
  QCOSFloat cinf = inf_norm(work->xbuff, data->n);

  ew_product(work->kkt->Finvruiz, data->h, work->ubuff3, data->m);
  QCOSFloat hinf = data->m > 0 ? inf_norm(work->ubuff3, data->m) : 0;

  // Compute ||A^T * y||_\infty. If equality constraints aren't present, A->m =
  // A->n = 0 and SpMtv is a nullop.
  SpMtv(data->A, work->y, work->xbuff);
  ew_product(work->xbuff, work->kkt->Dinvruiz, work->xbuff, data->n);
  QCOSFloat Atyinf = data->p ? inf_norm(work->xbuff, data->n) : 0;

  // Compute ||G^T * z||_\infty. If inequality constraints aren't present, G->m
  // = G->n = 0 and SpMtv is a nullop.
  SpMtv(data->G, work->z, work->xbuff);
  ew_product(work->xbuff, work->kkt->Dinvruiz, work->xbuff, data->n);
  QCOSFloat Gtzinf = data->m > 0 ? inf_norm(work->xbuff, data->n) : 0;

  // Compute ||P * x||_\infty
  SpMv(data->P, work->x, work->xbuff);
  for (QCOSInt i = 0; i < data->n; ++i) {
    work->xbuff[i] -= solver->settings->reg * work->x[i];
  }
  ew_product(work->xbuff, work->kkt->Dinvruiz, work->xbuff, data->n);
  QCOSFloat Pxinf = inf_norm(work->xbuff, data->n);

  // Compute ||A * x||_\infty
  SpMv(data->A, work->x, work->ybuff);
  ew_product(work->ybuff, work->kkt->Einvruiz, work->ybuff, data->p);
  QCOSFloat Axinf = data->p ? inf_norm(work->ybuff, data->p) : 0;

  // Compute ||G * x||_\infty
  SpMv(data->G, work->x, work->ubuff1);
  ew_product(work->ubuff1, work->kkt->Finvruiz, work->ubuff1, data->m);
  QCOSFloat Gxinf = data->m ? inf_norm(work->ubuff1, data->m) : 0;

  // Compute primal residual.
  ew_product(&work->kkt->kktres[data->n], work->kkt->Einvruiz, work->ybuff,
             data->p);
  QCOSFloat eq_res = inf_norm(work->ybuff, data->p);

  ew_product(&work->kkt->kktres[data->n + data->p], work->kkt->Finvruiz,
             work->ubuff1, data->m);
  QCOSFloat conic_res = inf_norm(work->ubuff1, data->m);

  QCOSFloat pres = qcos_max(eq_res, conic_res);
  solver->sol->pres = pres;

  // Compute dual residual.
  ew_product(work->kkt->kktres, work->kkt->Dinvruiz, work->xbuff, data->n);
  scale_arrayf(work->xbuff, work->xbuff, work->kkt->kinv, data->n);
  QCOSFloat dres = inf_norm(work->xbuff, data->n);
  solver->sol->dres = dres;

  // Compute duality gap.
  ew_product(work->s, work->kkt->Fruiz, work->ubuff1, data->m);
  ew_product(work->z, work->kkt->Fruiz, work->ubuff2, data->m);
  QCOSFloat gap = dot(work->ubuff1, work->ubuff2, data->m);
  gap *= work->kkt->kinv;
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
  dres_rel *= work->kkt->kinv;

  // Compute max{sinf, zinf}.
  QCOSFloat gap_rel = qcos_max(sinf, zinf);

  // If the solver stalled (stepsize = 0) check if low tolerance stopping
  // criteria is met.
  if ((solver->work->a < 1e-8) &&
      (solver->work->mu < eabsinacc &&
       pres < eabsinacc + erelinacc * pres_rel &&
       dres < eabsinacc + erelinacc * dres_rel &&
       solver->sol->gap < eabsinacc + erelinacc * gap_rel)) {
    solver->sol->status = QCOS_SOLVED_INACCURATE;
    return 1;
  }

  if (solver->work->mu < eabs && pres < eabs + erel * pres_rel &&
      dres < eabs + erel * dres_rel &&
      solver->sol->gap < eabs + erel * gap_rel) {
    solver->sol->status = QCOS_SOLVED;
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

  solver->sol->solve_time_sec =
      get_elapsed_time_sec(&(solver->work->solve_timer));
}