/**
 * @file qoco_api.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "qoco_api.h"
#include "amd.h"
#include "qdldl_backend.h" // TODO: make this modular so we can use any backend.

QOCOInt qoco_setup(QOCOSolver* solver, QOCOInt n, QOCOInt m, QOCOInt p,
                   QOCOCscMatrix* P, QOCOFloat* c, QOCOCscMatrix* A,
                   QOCOFloat* b, QOCOCscMatrix* G, QOCOFloat* h, QOCOInt l,
                   QOCOInt nsoc, QOCOInt* q, QOCOSettings* settings)
{
  // Start setup timer.
  QOCOTimer setup_timer;
  start_timer(&setup_timer);

  // Validate problem data.
  if (qoco_validate_data(P, c, A, b, G, h, l, nsoc, q)) {
    return qoco_error(QOCO_DATA_VALIDATION_ERROR);
  }

  // Validate settings.
  if (qoco_validate_settings(settings)) {
    return qoco_error(QOCO_SETTINGS_VALIDATION_ERROR);
  }

  solver->settings = copy_settings(settings);

  // Allocate workspace.
  solver->work = qoco_malloc(sizeof(QOCOWorkspace));

  // Allocate problem data.
  solver->work->data = qoco_malloc(sizeof(QOCOProblemData));
  QOCOProblemData* data = solver->work->data;

  // Copy problem data.
  data->m = m;
  data->n = n;
  data->p = p;
  data->A = new_qoco_csc_matrix(A);
  data->G = new_qoco_csc_matrix(G);
  data->c = qoco_malloc(n * sizeof(QOCOFloat));
  data->b = qoco_malloc(p * sizeof(QOCOFloat));
  data->h = qoco_malloc(m * sizeof(QOCOFloat));
  data->q = qoco_malloc(nsoc * sizeof(QOCOInt));

  copy_arrayf(c, data->c, n);
  copy_arrayf(b, data->b, p);
  copy_arrayf(h, data->h, m);
  copy_arrayi(q, data->q, nsoc);

  data->l = l;
  data->nsoc = nsoc;

  // Copy P.
  if (P) {
    data->P = new_qoco_csc_matrix(P);
  }
  else {
    data->P = NULL;
  }

  // Equilibrate data.
  QOCOInt Annz = A ? A->nnz : 0;
  QOCOInt Gnnz = G ? G->nnz : 0;

  solver->work->scaling = qoco_malloc(sizeof(QOCOScaling));
  solver->work->scaling->delta = qoco_malloc((n + p + m) * sizeof(QOCOFloat));
  solver->work->scaling->Druiz = qoco_malloc(n * sizeof(QOCOFloat));
  solver->work->scaling->Eruiz = qoco_malloc(p * sizeof(QOCOFloat));
  solver->work->scaling->Fruiz = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->scaling->Dinvruiz = qoco_malloc(n * sizeof(QOCOFloat));
  solver->work->scaling->Einvruiz = qoco_malloc(p * sizeof(QOCOFloat));
  solver->work->scaling->Finvruiz = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->data->AtoAt = qoco_malloc(Annz * sizeof(QOCOInt));
  solver->work->data->GtoGt = qoco_malloc(Gnnz * sizeof(QOCOInt));

  solver->work->data->At = create_transposed_matrix(data->A, data->AtoAt);
  solver->work->data->Gt = create_transposed_matrix(data->G, data->GtoGt);
  ruiz_equilibration(data, solver->work->scaling, solver->settings->ruiz_iters);

  // Regularize P.
  data->Pnzadded_idx = qoco_calloc(n, sizeof(QOCOInt));
  if (P) {
    QOCOInt num_diagP = count_diag(P);
    data->Pnum_nzadded = n - num_diagP;
    QOCOCscMatrix* Preg =
        regularize_P(num_diagP, data->P, solver->settings->kkt_static_reg,
                     data->Pnzadded_idx);
    data->P = Preg;
  }
  else {
    data->P = construct_identity(n, solver->settings->kkt_static_reg);
    data->Pnum_nzadded = n;
  }

  // Compute number of nonzeros in upper triangular NT scaling matrix.
  QOCOInt Wsoc_nnz = 0;
  for (QOCOInt i = 0; i < nsoc; ++i) {
    Wsoc_nnz += q[i] * q[i] - q[i];
  }
  Wsoc_nnz /= 2;
  QOCOInt Wnnz = m + Wsoc_nnz;
  solver->work->Wnnz = Wnnz;

  solver->linsys = &backend;

  // Set up linear system data.
  solver->linsys_data =
      solver->linsys->linsys_setup(data, solver->settings, Wnnz);
  if (!solver->linsys_data) {
    return QOCO_SETUP_ERROR;
  }

  // Allocate primal and dual variables.
  solver->work->x = qoco_malloc(n * sizeof(QOCOFloat));
  solver->work->s = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->y = qoco_malloc(p * sizeof(QOCOFloat));
  solver->work->z = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->mu = 0.0;

  // Allocate Nesterov-Todd scalings and scaled variables.
  QOCOInt Wnnzfull = data->l;
  for (QOCOInt i = 0; i < data->nsoc; ++i) {
    Wnnzfull += data->q[i] * data->q[i];
  }

  solver->work->W = qoco_malloc(solver->work->Wnnz * sizeof(QOCOFloat));
  solver->work->Wfull = qoco_malloc(Wnnzfull * sizeof(QOCOFloat));
  for (QOCOInt i = 0; i < Wnnzfull; ++i) {
    solver->work->Wfull[i] = 0.0;
  }
  solver->work->Wnnzfull = Wnnzfull;
  solver->work->Winv = qoco_malloc(solver->work->Wnnz * sizeof(QOCOFloat));
  solver->work->Winvfull = qoco_malloc(Wnnzfull * sizeof(QOCOFloat));
  solver->work->WtW = qoco_malloc(solver->work->Wnnz * sizeof(QOCOFloat));
  solver->work->lambda = qoco_malloc(m * sizeof(QOCOFloat));
  QOCOInt qmax = 0;
  if (solver->work->data->nsoc) {
    qmax = max_arrayi(solver->work->data->q, solver->work->data->nsoc);
  }
  solver->work->sbar = qoco_malloc(qmax * sizeof(QOCOFloat));
  solver->work->zbar = qoco_malloc(qmax * sizeof(QOCOFloat));
  solver->work->xbuff = qoco_malloc(n * sizeof(QOCOFloat));
  solver->work->ybuff = qoco_malloc(p * sizeof(QOCOFloat));
  solver->work->ubuff1 = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->ubuff2 = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->ubuff3 = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->Ds = qoco_malloc(m * sizeof(QOCOFloat));
  solver->work->rhs = qoco_malloc((n + m + p) * sizeof(QOCOFloat));
  solver->work->kktres = qoco_malloc((n + m + p) * sizeof(QOCOFloat));
  solver->work->xyz = qoco_malloc((n + m + p) * sizeof(QOCOFloat));
  solver->work->xyzbuff1 = qoco_malloc((n + m + p) * sizeof(QOCOFloat));
  solver->work->xyzbuff2 = qoco_malloc((n + m + p) * sizeof(QOCOFloat));

  // Allocate solution struct.
  solver->sol = qoco_malloc(sizeof(QOCOSolution));
  solver->sol->x = qoco_malloc(n * sizeof(QOCOFloat));
  solver->sol->s = qoco_malloc(m * sizeof(QOCOFloat));
  solver->sol->y = qoco_malloc(p * sizeof(QOCOFloat));
  solver->sol->z = qoco_malloc(m * sizeof(QOCOFloat));
  solver->sol->iters = 0;
  solver->sol->status = QOCO_UNSOLVED;

  stop_timer(&setup_timer);
  solver->sol->setup_time_sec = get_elapsed_time_sec(&setup_timer);

  return QOCO_NO_ERROR;
}

void qoco_set_csc(QOCOCscMatrix* A, QOCOInt m, QOCOInt n, QOCOInt Annz,
                  QOCOFloat* Ax, QOCOInt* Ap, QOCOInt* Ai)
{
  A->m = m;
  A->n = n;
  A->nnz = Annz;
  A->x = Ax;
  A->p = Ap;
  A->i = Ai;
}

void set_default_settings(QOCOSettings* settings)
{
  settings->max_iters = 200;
  settings->bisect_iters = 5;
  settings->ruiz_iters = 0;
  settings->iter_ref_iters = 1;
  settings->kkt_static_reg = 1e-8;
  settings->kkt_dynamic_reg = 1e-8;
  settings->abstol = 1e-7;
  settings->reltol = 1e-7;
  settings->abstol_inacc = 1e-5;
  settings->reltol_inacc = 1e-5;
  settings->verbose = 0;
}

QOCOInt qoco_update_settings(QOCOSolver* solver,
                             const QOCOSettings* new_settings)
{
  if (qoco_validate_settings(new_settings)) {
    return qoco_error(QOCO_SETTINGS_VALIDATION_ERROR);
  }

  solver->settings->max_iters = new_settings->max_iters;
  solver->settings->bisect_iters = new_settings->bisect_iters;
  solver->settings->ruiz_iters = new_settings->ruiz_iters;
  solver->settings->iter_ref_iters = new_settings->iter_ref_iters;
  solver->settings->kkt_static_reg = new_settings->kkt_static_reg;
  solver->settings->kkt_dynamic_reg = new_settings->kkt_dynamic_reg;
  solver->settings->abstol = new_settings->abstol;
  solver->settings->reltol = new_settings->reltol;
  solver->settings->abstol_inacc = new_settings->abstol_inacc;
  solver->settings->abstol_inacc = new_settings->abstol_inacc;
  solver->settings->verbose = new_settings->verbose;

  return 0;
}

void update_vector_data(QOCOSolver* solver, QOCOFloat* cnew, QOCOFloat* bnew,
                        QOCOFloat* hnew)
{
  solver->sol->status = QOCO_UNSOLVED;
  QOCOProblemData* data = solver->work->data;

  // Update cost vector.
  if (cnew) {
    for (QOCOInt i = 0; i < data->n; ++i) {
      data->c[i] =
          solver->work->scaling->k * solver->work->scaling->Druiz[i] * cnew[i];
    }
  }

  // Update equality constraint vector.
  if (bnew) {
    for (QOCOInt i = 0; i < data->p; ++i) {
      data->b[i] = solver->work->scaling->Eruiz[i] * bnew[i];
    }
  }

  // Update conic constraint vector.
  if (hnew) {
    for (QOCOInt i = 0; i < data->m; ++i) {
      data->h[i] = solver->work->scaling->Fruiz[i] * hnew[i];
    }
  }
}

void update_matrix_data(QOCOSolver* solver, QOCOFloat* Pxnew, QOCOFloat* Axnew,
                        QOCOFloat* Gxnew)
{
  solver->sol->status = QOCO_UNSOLVED;
  QOCOProblemData* data = solver->work->data;
  QOCOScaling* scaling = solver->work->scaling;

  // Undo regularization.
  unregularize(data->P, solver->settings->kkt_static_reg);

  // Unequilibrate P.
  scale_arrayf(data->P->x, data->P->x, scaling->kinv, data->P->nnz);
  row_col_scale(data->P, scaling->Dinvruiz, scaling->Dinvruiz);

  // Unequilibrate c.
  scale_arrayf(data->c, data->c, scaling->kinv, data->n);
  ew_product(data->c, scaling->Dinvruiz, data->c, data->n);

  // Unequilibrate A.
  row_col_scale(data->A, scaling->Einvruiz, scaling->Dinvruiz);

  // Unequilibrate G.
  row_col_scale(data->G, scaling->Finvruiz, scaling->Dinvruiz);

  // Unequilibrate b.
  ew_product(data->b, scaling->Einvruiz, data->b, data->p);

  // Unequilibrate h.
  ew_product(data->h, scaling->Finvruiz, data->h, data->m);

  // Update P and avoid nonzeros that were added for regularization.
  if (Pxnew) {
    QOCOInt avoid =
        data->Pnum_nzadded > 0 ? data->Pnzadded_idx[0] : data->P->nnz + 1;
    QOCOInt offset = 0;
    for (QOCOInt i = 0; i < data->P->nnz - data->Pnum_nzadded; ++i) {
      if (i == avoid) {
        offset++;
        avoid = data->Pnzadded_idx[offset];
      }
      else {
        data->P->x[i + offset] = Pxnew[i];
      }
    }
  }

  // Update A.
  if (Axnew) {
    for (QOCOInt i = 0; i < data->A->nnz; ++i) {
      data->A->x[i] = Axnew[i];
      data->At->x[i] = Axnew[data->AtoAt[i]];
    }
  }

  // Update G.
  if (Gxnew) {
    for (QOCOInt i = 0; i < data->G->nnz; ++i) {
      data->G->x[i] = Gxnew[i];
      data->Gt->x[i] = Gxnew[data->GtoGt[i]];
    }
  }

  // Equilibrate new matrix data.
  ruiz_equilibration(solver->work->data, solver->work->scaling,
                     solver->settings->ruiz_iters);

  // Regularize P.
  unregularize(data->P, -solver->settings->kkt_static_reg);

  solver->linsys->linsys_update_data(solver->linsys_data, solver->work->data);
}

QOCOInt qoco_solve(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOProblemData* data = solver->work->data;

  start_timer(&(work->solve_timer));

  // Validate settings.
  if (qoco_validate_settings(solver->settings)) {
    return qoco_error(QOCO_SETTINGS_VALIDATION_ERROR);
  }

  if (solver->settings->verbose) {
    print_header(solver);
  }

  // Get initializations for primal and dual variables.
  initialize_ipm(solver);
  for (QOCOInt i = 1; i <= solver->settings->max_iters; ++i) {

    // Compute kkt residual.
    compute_kkt_residual(data, work->x, work->y, work->s, work->z, work->kktres,
                         solver->settings->kkt_static_reg, work->xyzbuff1,
                         work->xbuff, work->ubuff1, work->ubuff2);

    // Compute objective function.
    solver->sol->obj =
        compute_objective(data, work->x, work->xbuff,
                          solver->settings->kkt_static_reg, work->scaling->k);

    // Compute mu = s'*z / m.
    work->mu = (data->m > 0)
                   ? safe_div(qoco_dot(work->s, work->z, data->m), data->m)
                   : 0;

    // Check stopping criteria.
    if (check_stopping(solver)) {
      stop_timer(&(work->solve_timer));
      unscale_variables(work);
      copy_solution(solver);
      if (solver->settings->verbose) {
        print_footer(solver->sol, solver->sol->status);
      }
      return solver->sol->status;
    }

    // Compute Nesterov-Todd scalings.
    compute_nt_scaling(work);

    // Update Nestrov-Todd block of KKT matrix.
    solver->linsys->linsys_update_nt(solver->linsys_data, work->WtW,
                                     solver->settings->kkt_static_reg, data->m);

    // Perform predictor-corrector.
    predictor_corrector(solver);

    // Update iteration count.
    solver->sol->iters = i;

    // Log solver progress to console if we are solving in verbose mode.
    if (solver->settings->verbose) {
      log_iter(solver);
    }
  }

  stop_timer(&(work->solve_timer));
  unscale_variables(work);
  copy_solution(solver);
  solver->sol->status = QOCO_MAX_ITER;
  if (solver->settings->verbose) {
    print_footer(solver->sol, solver->sol->status);
  }
  return QOCO_MAX_ITER;
}

QOCOInt qoco_cleanup(QOCOSolver* solver)
{

  // Free problem data.
  free_qoco_csc_matrix(solver->work->data->P);
  free_qoco_csc_matrix(solver->work->data->A);
  free_qoco_csc_matrix(solver->work->data->G);
  free_qoco_csc_matrix(solver->work->data->At);
  free_qoco_csc_matrix(solver->work->data->Gt);

  qoco_free(solver->work->data->AtoAt);
  qoco_free(solver->work->data->GtoGt);
  qoco_free(solver->work->data->b);
  qoco_free(solver->work->data->c);
  qoco_free(solver->work->data->h);
  qoco_free(solver->work->data->q);
  qoco_free(solver->work->data->Pnzadded_idx);
  qoco_free(solver->work->data);

  // Cleanup linsys.
  solver->linsys->linsys_cleanup(solver->linsys_data);

  // Free primal and dual variables.
  qoco_free(solver->work->rhs);
  qoco_free(solver->work->kktres);
  qoco_free(solver->work->xyz);
  qoco_free(solver->work->xyzbuff1);
  qoco_free(solver->work->xyzbuff2);
  qoco_free(solver->work->x);
  qoco_free(solver->work->s);
  qoco_free(solver->work->y);
  qoco_free(solver->work->z);

  // Free Nesterov-Todd scalings and scaled variables.
  qoco_free(solver->work->W);
  qoco_free(solver->work->Wfull);
  qoco_free(solver->work->Winv);
  qoco_free(solver->work->Winvfull);
  qoco_free(solver->work->WtW);
  qoco_free(solver->work->lambda);
  qoco_free(solver->work->sbar);
  qoco_free(solver->work->zbar);
  qoco_free(solver->work->xbuff);
  qoco_free(solver->work->ybuff);
  qoco_free(solver->work->ubuff1);
  qoco_free(solver->work->ubuff2);
  qoco_free(solver->work->ubuff3);
  qoco_free(solver->work->Ds);

  // Free scaling struct.
  qoco_free(solver->work->scaling->delta);
  qoco_free(solver->work->scaling->Druiz);
  qoco_free(solver->work->scaling->Eruiz);
  qoco_free(solver->work->scaling->Fruiz);
  qoco_free(solver->work->scaling->Dinvruiz);
  qoco_free(solver->work->scaling->Einvruiz);
  qoco_free(solver->work->scaling->Finvruiz);
  qoco_free(solver->work->scaling);

  // Free solution struct.
  qoco_free(solver->sol->x);
  qoco_free(solver->sol->s);
  qoco_free(solver->sol->y);
  qoco_free(solver->sol->z);
  qoco_free(solver->sol);

  qoco_free(solver->work);
  qoco_free(solver->settings);
  qoco_free(solver);

  return 1;
}