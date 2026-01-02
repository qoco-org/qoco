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

void compute_scaling_statistics(QOCOProblemData* data)
{
  // This function runs on the CPU.
  set_cpu_mode(1);

  // Initialize min/max values
  data->obj_range_min = QOCOFloat_MAX;
  data->obj_range_max = 0.0;
  data->constraint_range_min = QOCOFloat_MAX;
  data->constraint_range_max = 0.0;
  data->rhs_range_min = QOCOFloat_MAX;
  data->rhs_range_max = 0.0;

  // Compute Objective range: P (prior to regularization) and c
  // Note: This is called before regularization, so P is in its original form
  if (data->P) {
    QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
    data->obj_range_min = min_abs_val(Pcsc->x, Pcsc->nnz);
    data->obj_range_max = inf_norm(Pcsc->x, Pcsc->nnz);
  }

  // Add c to objective range
  QOCOFloat* cdata = get_data_vectorf(data->c);
  data->obj_range_min = min_abs_val(cdata, data->n);
  data->obj_range_max = inf_norm(cdata, data->n);

  // Compute Constraint range: A and G
  if (data->A && get_nnz(data->A) > 0) {
    QOCOCscMatrix* Acsc = get_csc_matrix(data->A);
    data->constraint_range_min = min_abs_val(Acsc->x, Acsc->nnz);
    data->constraint_range_max = inf_norm(Acsc->x, Acsc->nnz);
  }

  if (data->G && get_nnz(data->G) > 0) {
    QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
    data->constraint_range_min =
        qoco_min(data->constraint_range_min, min_abs_val(Gcsc->x, Gcsc->nnz));
    data->constraint_range_max =
        qoco_max(data->constraint_range_max, inf_norm(Gcsc->x, Gcsc->nnz));
  }

  // Compute RHS range: b and h
  if (data->p > 0) {
    QOCOFloat* bdata = get_data_vectorf(data->b);
    data->rhs_range_min = min_abs_val(bdata, data->p);
    data->rhs_range_max = inf_norm(bdata, data->p);
  }

  if (data->m > 0) {
    QOCOFloat* hdata = get_data_vectorf(data->h);
    data->rhs_range_min = min_abs_val(hdata, data->m);
    data->rhs_range_max = inf_norm(hdata, data->m);
  }
  // Handle case where all values are zero or value is -0.0 (set to 0.0)
  if (data->obj_range_min == QOCOFloat_MAX || data->obj_range_min == 0.0) {
    data->obj_range_min = 0.0;
  }
  if (data->obj_range_max == QOCOFloat_MAX || data->obj_range_max == 0.0) {
    data->obj_range_max = 0.0;
  }
  if (data->constraint_range_min == QOCOFloat_MAX ||
      data->constraint_range_min == 0.0) {
    data->constraint_range_min = 0.0;
  }
  if (data->constraint_range_max == QOCOFloat_MAX ||
      data->constraint_range_max == 0.0) {
    data->constraint_range_max = 0.0;
  }
  if (data->rhs_range_min == QOCOFloat_MAX || data->rhs_range_min == 0.0) {
    data->rhs_range_min = 0.0;
  }
  if (data->rhs_range_max == QOCOFloat_MAX || data->rhs_range_max == 0.0) {
    data->rhs_range_max = 0.0;
  }
  set_cpu_mode(0);
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
  printf("|     nnz(P):           %-9d                       |\n", (data->P ? get_nnz(data->P) : 0) - solver->work->data->Pnum_nzadded);
  printf("|     nnz(A):           %-9d                       |\n", get_nnz(data->A));
  printf("|     nnz(G):           %-9d                       |\n", get_nnz(data->G));
  printf("| Scaling Statistics:                                   |\n");
  printf("|     Objective range      [%.0e, %.0e]               |\n", data->obj_range_min, data->obj_range_max);
  printf("|     Constraint range     [%.0e, %.0e]               |\n", data->constraint_range_min, data->constraint_range_max);
  printf("|     RHS range            [%.0e, %.0e]               |\n", data->rhs_range_min, data->rhs_range_max);
  printf("| Solver Settings:                                      |\n");
  printf("|     algebra: %-27s              |\n", solver->linsys->linsys_name());
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
  QOCOFloat* xbuff = get_data_vectorf(work->xbuff);
  QOCOFloat* ybuff = get_data_vectorf(work->ybuff);
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
  QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
  QOCOFloat* ubuff3 = get_data_vectorf(work->ubuff3);
  QOCOFloat* kktres = get_data_vectorf(work->kktres);

  QOCOFloat* Einvruiz_data = get_data_vectorf(work->scaling->Einvruiz);
  QOCOFloat* bdata = get_data_vectorf(data->b);
  ew_product(Einvruiz_data, bdata, ybuff, data->p);
  QOCOFloat binf = data->p > 0 ? inf_norm(ybuff, data->p) : 0;

  QOCOFloat* Fruiz_data = get_data_vectorf(work->scaling->Fruiz);
  QOCOFloat* sdata = get_data_vectorf(work->s);
  ew_product(Fruiz_data, sdata, ubuff1, data->m);
  QOCOFloat sinf = data->m > 0 ? inf_norm(ubuff1, data->m) : 0;

  QOCOFloat* Dinvruiz_data = get_data_vectorf(work->scaling->Dinvruiz);
  QOCOFloat* xdata = get_data_vectorf(work->x);
  ew_product(Dinvruiz_data, xdata, xbuff, data->n);
  QOCOFloat cinf = inf_norm(xbuff, data->n);

  QOCOFloat* Finvruiz_data = get_data_vectorf(work->scaling->Finvruiz);
  QOCOFloat* hdata = get_data_vectorf(data->h);
  ew_product(Finvruiz_data, hdata, ubuff3, data->m);
  QOCOFloat hinf = data->m > 0 ? inf_norm(ubuff3, data->m) : 0;

  // Compute ||A^T * y||_\infty. If equality constraints aren't present, A->m =
  // A->n = 0 and SpMtv is a nullop.
  QOCOFloat* ydata = get_data_vectorf(work->y);
  SpMtv(data->A, ydata, xbuff);
  ew_product(xbuff, Dinvruiz_data, xbuff, data->n);
  QOCOFloat Atyinf = data->p ? inf_norm(xbuff, data->n) : 0;

  // Compute ||G^T * z||_\infty. If inequality constraints aren't present, G->m
  // = G->n = 0 and SpMtv is a nullop.
  QOCOFloat* zdata = get_data_vectorf(work->z);
  SpMtv(data->G, zdata, xbuff);
  ew_product(xbuff, Dinvruiz_data, xbuff, data->n);
  QOCOFloat Gtzinf = data->m > 0 ? inf_norm(xbuff, data->n) : 0;

  // Compute ||P * x||_\infty
  USpMv(data->P, xdata, xbuff);
  qoco_axpy(xdata, xbuff, xbuff, -solver->settings->kkt_static_reg, data->n);
  ew_product(xbuff, Dinvruiz_data, xbuff, data->n);
  QOCOFloat Pxinf = inf_norm(xbuff, data->n);
  QOCOFloat xPx = qoco_dot(xdata, xbuff, work->data->n);

  // Compute ||A * x||_\infty
  SpMv(data->A, xdata, ybuff);
  ew_product(ybuff, Einvruiz_data, ybuff, data->p);
  QOCOFloat Axinf = data->p ? inf_norm(ybuff, data->p) : 0;

  // Compute ||G * x||_\infty
  SpMv(data->G, xdata, ubuff1);
  ew_product(ubuff1, Finvruiz_data, ubuff1, data->m);
  QOCOFloat Gxinf = data->m ? inf_norm(ubuff1, data->m) : 0;

  // Compute primal residual.
  ew_product(&kktres[data->n], Einvruiz_data, ybuff, data->p);
  QOCOFloat eq_res = inf_norm(ybuff, data->p);

  ew_product(&kktres[data->n + data->p], Finvruiz_data, ubuff1, data->m);
  QOCOFloat conic_res = inf_norm(ubuff1, data->m);

  QOCOFloat pres = qoco_max(eq_res, conic_res);
  solver->sol->pres = pres;

  // Compute dual residual.
  ew_product(kktres, Dinvruiz_data, xbuff, data->n);
  scale_arrayf(xbuff, xbuff, work->scaling->kinv, data->n);
  QOCOFloat dres = inf_norm(xbuff, data->n);
  solver->sol->dres = dres;

  // Compute complementary slackness residual.
  ew_product(sdata, Fruiz_data, ubuff1, data->m);
  ew_product(zdata, Fruiz_data, ubuff2, data->m);
  QOCOFloat gap = qoco_dot(ubuff1, ubuff2, data->m);
  gap *= work->scaling->kinv;
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
  dres_rel *= work->scaling->kinv;

  // Compute max{1, abs(pobj), abs(dobj)}.
  QOCOFloat* cdata = get_data_vectorf(work->data->c);
  QOCOFloat ctx = qoco_dot(cdata, xdata, work->data->n);
  QOCOFloat bty = qoco_dot(bdata, ydata, work->data->p);
  QOCOFloat htz = qoco_dot(hdata, zdata, work->data->m);
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
  // Copy optimization variables from device to host (CUDA backend).
  // No-op for builtin backend.
  sync_vector_to_host(solver->work->x);
  sync_vector_to_host(solver->work->s);
  sync_vector_to_host(solver->work->y);
  sync_vector_to_host(solver->work->z);

  // Set cpu mode to 1 so get_data_vectorf returns host pointer.
  set_cpu_mode(1);
  copy_arrayf(get_data_vectorf(solver->work->x), solver->sol->x,
              solver->work->data->n);
  copy_arrayf(get_data_vectorf(solver->work->s), solver->sol->s,
              solver->work->data->m);
  copy_arrayf(get_data_vectorf(solver->work->y), solver->sol->y,
              solver->work->data->p);
  copy_arrayf(get_data_vectorf(solver->work->z), solver->sol->z,
              solver->work->data->m);
  set_cpu_mode(0);

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