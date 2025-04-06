/**
 * @file utils.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "qoco_utils.h"

void print_qoco_csc_matrix(QOCOCscMatrix* M)
{
  printf("\nPrinting CSC Matrix:\n");
  printf("m: %d\n", M->m);
  printf("n: %d\n", M->n);
  printf("nnz: %d\n", M->nnz);
  printf("Data: {");
  for (QOCOInt i = 0; i < M->nnz; ++i) {
    printf("%.17g", M->x[i]);
    if (i != M->nnz - 1) {
      printf(",");
    }
  }
  printf("}\n");

  printf("Row Indices: {");
  for (QOCOInt i = 0; i < M->nnz; ++i) {
    printf("%d", M->i[i]);
    if (i != M->nnz - 1) {
      printf(",");
    }
  }
  printf("}\n");

  printf("Column Pointers: {");
  for (QOCOInt i = 0; i < M->n + 1; ++i) {
    printf("%d", M->p[i]);
    if (i != M->n) {
      printf(",");
    }
  }
  printf("}\n");
}

void print_arrayf(QOCOFloat* x, QOCOInt n)
{
  printf("{");
  for (QOCOInt i = 0; i < n; ++i) {
    printf("%.17g", x[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("}\n");
}

void print_arrayi(QOCOInt* x, QOCOInt n)
{
  printf("{");
  for (QOCOInt i = 0; i < n; ++i) {
    printf("%d", x[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("}\n");
}

void print_header(QOCOSolver* solver)
{
  QOCOProblemData* data = solver->work->data;
  QOCOSettings* settings = solver->settings;

  // clang-format off
  printf("\n");
  printf("+-------------------------------------------------------+\n");
  printf("|     QOCO - Quadratic Objective Conic Optimizer        |\n");
  printf("|                        v%i.%i.%i                         |\n", QOCO_VERSION_MAJOR, QOCO_VERSION_MINOR, QOCO_VERSION_PATCH);
  printf("|             (c) Govind M. Chari, 2025                 |\n");
  printf("|    University of Washington Autonomous Controls Lab   |\n");
  printf("+-------------------------------------------------------+\n");
  printf("| Problem Data:                                         |\n");
  printf("|     variables:        %-9d                       |\n", data->n);
  printf("|     constraints:      %-9d                       |\n", data->l + data->p + data->nsoc);
  printf("|     eq constraints:   %-9d                       |\n", data->p);
  printf("|     ineq constraints: %-9d                       |\n", data->l);
  printf("|     soc constraints:  %-9d                       |\n", data->nsoc);
  printf("|     nnz(P):           %-9d                       |\n", data->P->nnz - solver->work->kkt->Pnum_nzadded);
  printf("|     nnz(A):           %-9d                       |\n", data->A->nnz);
  printf("|     nnz(G):           %-9d                       |\n", data->G->nnz);
  printf("| Solver Settings:                                      |\n");
  printf("|     max_iters: %-3d abstol: %3.2e reltol: %3.2e  |\n", settings->max_iters, settings->abstol, settings->reltol);
  printf("|     abstol_inacc: %3.2e reltol_inacc: %3.2e     |\n", settings->abstol_inacc, settings->reltol_inacc);
  printf("|     bisect_iters: %-2d iter_ref_iters: %-2d               |\n", settings->bisect_iters, settings->iter_ref_iters);
  printf("|     ruiz_iters: %-2d kkt_static_reg: %3.2e           |\n", settings->ruiz_iters, settings->kkt_static_reg);
  printf("|     kkt_dynamic_reg: %3.2e                         |\n", settings->kkt_dynamic_reg);
  printf("+-------------------------------------------------------+\n");
  printf("\n");
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  printf("|  Iter  |   Pcost   |    Pres    |    Dres    |     Gap    |     Mu    |    Step   |\n");
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  // clang-format on
}

void log_iter(QOCOSolver* solver)
{
  // clang-format off
  printf("|   %2d   | %+.2e | %+.3e | %+.3e | %+.3e | %+.2e |   %.3f   |\n",
         solver->sol->iters, solver->sol->obj, solver->sol->pres, solver->sol->dres, solver->sol->gap, solver->work->mu, solver->work->a);
  printf("+--------+-----------+------------+------------+------------+-----------+-----------+\n");
  // clang-format on
}

void print_footer(QOCOSolution* solution, enum qoco_solve_status status)
{
  printf("\n");
  printf("status:                %s\n", QOCO_SOLVE_STATUS_MESSAGE[status]);
  printf("number of iterations:  %d\n", solution->iters);
  printf("objective:             %+.3f\n", solution->obj);
  printf("setup time:            %.2e sec\n", solution->setup_time_sec);
  printf("solve time:            %.2e sec\n", solution->solve_time_sec);
  printf("\n");
}

unsigned char check_stopping(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOProblemData* data = solver->work->data;
  QOCOFloat eabs = solver->settings->abstol;
  QOCOFloat erel = solver->settings->reltol;
  QOCOFloat eabsinacc = solver->settings->abstol_inacc;
  QOCOFloat erelinacc = solver->settings->reltol_inacc;

  ew_product(work->kkt->Einvruiz, data->b, work->ybuff, data->p);
  QOCOFloat binf = data->p > 0 ? inf_norm(work->ybuff, data->p) : 0;

  ew_product(work->kkt->Fruiz, work->s, work->ubuff1, data->m);
  QOCOFloat sinf = data->m > 0 ? inf_norm(work->ubuff1, data->m) : 0;

  ew_product(work->kkt->Dinvruiz, work->x, work->xbuff, data->n);
  QOCOFloat cinf = inf_norm(work->xbuff, data->n);

  ew_product(work->kkt->Finvruiz, data->h, work->ubuff3, data->m);
  QOCOFloat hinf = data->m > 0 ? inf_norm(work->ubuff3, data->m) : 0;

  // Compute ||A^T * y||_\infty. If equality constraints aren't present, A->m =
  // A->n = 0 and SpMtv is a nullop.
  SpMtv(data->A, work->y, work->xbuff);
  ew_product(work->xbuff, work->kkt->Dinvruiz, work->xbuff, data->n);
  QOCOFloat Atyinf = data->p ? inf_norm(work->xbuff, data->n) : 0;

  // Compute ||G^T * z||_\infty. If inequality constraints aren't present, G->m
  // = G->n = 0 and SpMtv is a nullop.
  SpMtv(data->G, work->z, work->xbuff);
  ew_product(work->xbuff, work->kkt->Dinvruiz, work->xbuff, data->n);
  QOCOFloat Gtzinf = data->m > 0 ? inf_norm(work->xbuff, data->n) : 0;

  // Compute ||P * x||_\infty
  USpMv(data->P, work->x, work->xbuff);
  for (QOCOInt i = 0; i < data->n; ++i) {
    work->xbuff[i] -= solver->settings->kkt_static_reg * work->x[i];
  }
  ew_product(work->xbuff, work->kkt->Dinvruiz, work->xbuff, data->n);
  QOCOFloat Pxinf = inf_norm(work->xbuff, data->n);
  QOCOFloat xPx = qoco_dot(work->x, work->xbuff, work->data->n);

  // Compute ||A * x||_\infty
  SpMv(data->A, work->x, work->ybuff);
  ew_product(work->ybuff, work->kkt->Einvruiz, work->ybuff, data->p);
  QOCOFloat Axinf = data->p ? inf_norm(work->ybuff, data->p) : 0;

  // Compute ||G * x||_\infty
  SpMv(data->G, work->x, work->ubuff1);
  ew_product(work->ubuff1, work->kkt->Finvruiz, work->ubuff1, data->m);
  QOCOFloat Gxinf = data->m ? inf_norm(work->ubuff1, data->m) : 0;

  // Compute primal residual.
  ew_product(&work->kkt->kktres[data->n], work->kkt->Einvruiz, work->ybuff,
             data->p);
  QOCOFloat eq_res = inf_norm(work->ybuff, data->p);

  ew_product(&work->kkt->kktres[data->n + data->p], work->kkt->Finvruiz,
             work->ubuff1, data->m);
  QOCOFloat conic_res = inf_norm(work->ubuff1, data->m);

  QOCOFloat pres = qoco_max(eq_res, conic_res);
  solver->sol->pres = pres;

  // Compute dual residual.
  ew_product(work->kkt->kktres, work->kkt->Dinvruiz, work->xbuff, data->n);
  scale_arrayf(work->xbuff, work->xbuff, work->kkt->kinv, data->n);
  QOCOFloat dres = inf_norm(work->xbuff, data->n);
  solver->sol->dres = dres;

  // Compute complementary slackness residual.
  ew_product(work->s, work->kkt->Fruiz, work->ubuff1, data->m);
  ew_product(work->z, work->kkt->Fruiz, work->ubuff2, data->m);
  QOCOFloat gap = qoco_dot(work->ubuff1, work->ubuff2, data->m);
  gap *= work->kkt->kinv;
  solver->sol->gap = gap;

  // Compute max{Axinf, binf, Gxinf, hinf, sinf}.
  QOCOFloat pres_rel = qoco_max(Axinf, binf);
  pres_rel = qoco_max(pres_rel, Gxinf);
  pres_rel = qoco_max(pres_rel, hinf);
  pres_rel = qoco_max(pres_rel, sinf);

  // Compute max{Pxinf, Atyinf, Gtzinf, cinf}.
  QOCOFloat dres_rel = qoco_max(Pxinf, Atyinf);
  dres_rel = qoco_max(dres_rel, Gtzinf);
  dres_rel = qoco_max(dres_rel, cinf);
  dres_rel *= work->kkt->kinv;

  // Compute max{1, abs(pobj), abs(dobj)}.
  QOCOFloat ctx = qoco_dot(work->data->c, work->x, work->data->n);
  QOCOFloat bty = qoco_dot(work->data->b, work->y, work->data->p);
  QOCOFloat htz = qoco_dot(work->data->h, work->z, work->data->m);
  QOCOFloat pobj = 0.5 * xPx + ctx;
  QOCOFloat dobj = -0.5 * xPx - bty - htz;

  pobj = qoco_abs(pobj);
  dobj = qoco_abs(dobj);

  QOCOFloat gap_rel = qoco_max(1, pobj);
  gap_rel = qoco_max(gap_rel, dobj);

  // If the solver stalled (stepsize = 0) check if low tolerance stopping
  // criteria is met.
  if (solver->work->a < 1e-8) {
    if (pres < eabsinacc + erelinacc * pres_rel &&
        dres < eabsinacc + erelinacc * dres_rel &&
        solver->sol->gap < eabsinacc + erelinacc * gap_rel) {
      solver->sol->status = QOCO_SOLVED_INACCURATE;
      return 1;
    }
    else {
      solver->sol->status = QOCO_NUMERICAL_ERROR;
      return 1;
    }
  }

  if (pres < eabs + erel * pres_rel && dres < eabs + erel * dres_rel &&
      solver->sol->gap < eabs + erel * gap_rel) {
    solver->sol->status = QOCO_SOLVED;
    return 1;
  }
  return 0;
}

void copy_solution(QOCOSolver* solver)
{
  // Copy optimization variables.
  copy_arrayf(solver->work->x, solver->sol->x, solver->work->data->n);
  copy_arrayf(solver->work->s, solver->sol->s, solver->work->data->m);
  copy_arrayf(solver->work->y, solver->sol->y, solver->work->data->p);
  copy_arrayf(solver->work->z, solver->sol->z, solver->work->data->m);

  solver->sol->solve_time_sec =
      get_elapsed_time_sec(&(solver->work->solve_timer));
}

QOCOSettings* copy_settings(QOCOSettings* settings)
{
  QOCOSettings* new_settings = malloc(sizeof(QOCOSettings));
  new_settings->abstol = settings->abstol;
  new_settings->abstol_inacc = settings->abstol_inacc;
  new_settings->bisect_iters = settings->bisect_iters;
  new_settings->iter_ref_iters = settings->iter_ref_iters;
  new_settings->max_iters = settings->max_iters;
  new_settings->kkt_static_reg = settings->kkt_static_reg;
  new_settings->kkt_dynamic_reg = settings->kkt_dynamic_reg;
  new_settings->reltol = settings->reltol;
  new_settings->reltol_inacc = settings->reltol_inacc;
  new_settings->ruiz_iters = settings->ruiz_iters;
  new_settings->verbose = settings->verbose;

  return new_settings;
}