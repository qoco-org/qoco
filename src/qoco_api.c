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
#include "backend.h"

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
  data->A = new_qoco_matrix(A);
  data->G = new_qoco_matrix(G);
  data->q = new_qoco_vectori(q, n);
  data->c = new_qoco_vectorf(c, n);
  data->b = new_qoco_vectorf(b, p);
  data->h = new_qoco_vectorf(h, m);
  data->l = l;
  data->nsoc = nsoc;

  // Copy P.
  if (P) {
    data->P = new_qoco_matrix(P);
  }
  else {
    data->P = NULL;
  }

  // Equilibrate data.
  QOCOInt Annz = A ? get_nnz(data->A) : 0;
  QOCOInt Gnnz = G ? get_nnz(data->G) : 0;

  solver->work->scaling = qoco_malloc(sizeof(QOCOScaling));
  solver->work->scaling->delta = new_qoco_vectorf(NULL, n + p + m);
  solver->work->scaling->Druiz = new_qoco_vectorf(NULL, n);
  solver->work->scaling->Eruiz = new_qoco_vectorf(NULL, p);
  solver->work->scaling->Fruiz = new_qoco_vectorf(NULL, m);
  solver->work->scaling->Dinvruiz = new_qoco_vectorf(NULL, n);
  solver->work->scaling->Einvruiz = new_qoco_vectorf(NULL, p);
  solver->work->scaling->Finvruiz = new_qoco_vectorf(NULL, m);
  solver->work->data->AtoAt = qoco_malloc(Annz * sizeof(QOCOInt));
  solver->work->data->GtoGt = qoco_malloc(Gnnz * sizeof(QOCOInt));

  // When creating transposed matrices, get_csc_matrix should return host
  // pointers, since create_transposed_matrix is a host function.
  set_cpu_mode(1);
  QOCOCscMatrix* Atcsc =
      create_transposed_matrix(get_csc_matrix(data->A), data->AtoAt);
  solver->work->data->At = new_qoco_matrix(Atcsc);
  free_qoco_csc_matrix(Atcsc);
  QOCOCscMatrix* Gtcsc =
      create_transposed_matrix(get_csc_matrix(data->G), data->GtoGt);
  solver->work->data->Gt = new_qoco_matrix(Gtcsc);
  free_qoco_csc_matrix(Gtcsc);
  set_cpu_mode(0);

  // Compute scaling statistics before equilibration and regularization.
  compute_scaling_statistics(data);

  ruiz_equilibration(data, solver->work->scaling, solver->settings->ruiz_iters);

  // Regularize P.
  set_cpu_mode(1);
  data->Pnzadded_idx = qoco_calloc(n, sizeof(QOCOInt));
  if (P) {
    QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
    QOCOInt num_diagP = count_diag(Pcsc);
    data->Pnum_nzadded = n - num_diagP;
    // Create a copy of the CSC matrix since regularize_P will free it
    QOCOCscMatrix* Pcsc_copy = new_qoco_csc_matrix(Pcsc);
    QOCOCscMatrix* Preg =
        regularize_P(num_diagP, Pcsc_copy, solver->settings->kkt_static_reg,
                     data->Pnzadded_idx);
    free_qoco_matrix(data->P);
    data->P = new_qoco_matrix(Preg);
    free_qoco_csc_matrix(Preg);
  }
  else {
    QOCOCscMatrix* Pid =
        construct_identity(n, solver->settings->kkt_static_reg);
    data->P = new_qoco_matrix(Pid);
    free_qoco_csc_matrix(Pid);
    data->Pnum_nzadded = n;
  }
  set_cpu_mode(0);

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
  solver->work->x = new_qoco_vectorf(NULL, n);
  solver->work->s = new_qoco_vectorf(NULL, m);
  solver->work->y = new_qoco_vectorf(NULL, p);
  solver->work->z = new_qoco_vectorf(NULL, m);
  solver->work->mu = 0.0;

  // Allocate Nesterov-Todd scalings and scaled variables.
  QOCOInt Wnnzfull = data->l;
  set_cpu_mode(1);
  for (QOCOInt i = 0; i < data->nsoc; ++i) {
    Wnnzfull +=
        get_element_vectori(data->q, i) * get_element_vectori(data->q, i);
  }
  QOCOInt qmax = 0;
  if (solver->work->data->nsoc) {
    qmax = max_arrayi(get_data_vectori(data->q), solver->work->data->nsoc);
  }
  set_cpu_mode(0);

  solver->work->W = new_qoco_vectorf(NULL, solver->work->Wnnz);
  solver->work->Wfull = new_qoco_vectorf(NULL, Wnnzfull);
  solver->work->Wnnzfull = Wnnzfull;
  solver->work->Winv = new_qoco_vectorf(NULL, solver->work->Wnnz);
  solver->work->Winvfull = new_qoco_vectorf(NULL, Wnnzfull);
  solver->work->WtW = new_qoco_vectorf(NULL, solver->work->Wnnz);
  solver->work->lambda = new_qoco_vectorf(NULL, m);

  solver->work->sbar = new_qoco_vectorf(NULL, qmax);
  solver->work->zbar = new_qoco_vectorf(NULL, qmax);
  solver->work->xbuff = new_qoco_vectorf(NULL, n);
  solver->work->ybuff = new_qoco_vectorf(NULL, p);
  solver->work->ubuff1 = new_qoco_vectorf(NULL, m);
  solver->work->ubuff2 = new_qoco_vectorf(NULL, m);
  solver->work->ubuff3 = new_qoco_vectorf(NULL, m);
  solver->work->Ds = new_qoco_vectorf(NULL, m);
  solver->work->rhs = new_qoco_vectorf(NULL, n + m + p);
  solver->work->kktres = new_qoco_vectorf(NULL, n + m + p);
  solver->work->xyz = new_qoco_vectorf(NULL, n + m + p);
  solver->work->xyzbuff1 = new_qoco_vectorf(NULL, n + m + p);
  solver->work->xyzbuff2 = new_qoco_vectorf(NULL, n + m + p);

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

  // Copy new data into CPU buffer.
  set_cpu_mode(1);
  if (cnew) {
    copy_arrayf(cnew, get_data_vectorf(data->c), data->n);
  }
  if (bnew) {
    copy_arrayf(bnew, get_data_vectorf(data->b), data->p);
  }
  if (hnew) {
    copy_arrayf(hnew, get_data_vectorf(data->h), data->m);
  }
  QOCOFloat* Druiz_data = get_data_vectorf(solver->work->scaling->Druiz);
  QOCOFloat* Eruiz_data = get_data_vectorf(solver->work->scaling->Eruiz);
  QOCOFloat* Fruiz_data = get_data_vectorf(solver->work->scaling->Fruiz);
  QOCOFloat* cdata = get_data_vectorf(data->c);
  QOCOFloat* bdata = get_data_vectorf(data->b);
  QOCOFloat* hdata = get_data_vectorf(data->h);
  set_cpu_mode(0);

  compute_scaling_statistics(data);

  // Update cost vector on CPU.
  if (cnew) {
    for (QOCOInt i = 0; i < data->n; ++i) {
      cdata[i] = solver->work->scaling->k * Druiz_data[i] * cdata[i];
    }
  }

  // Update equality constraint vector on CPU.
  if (bnew) {
    for (QOCOInt i = 0; i < data->p; ++i) {
      bdata[i] = Eruiz_data[i] * bdata[i];
    }
  }

  // Update conic constraint vector on CPU.
  if (hnew) {
    for (QOCOInt i = 0; i < data->m; ++i) {
      hdata[i] = Fruiz_data[i] * hdata[i];
    }
  }

  // Sync the new data to the GPU.
  sync_vector_to_device(data->c);
  sync_vector_to_device(data->b);
  sync_vector_to_device(data->h);
}

void update_matrix_data(QOCOSolver* solver, QOCOFloat* Pxnew, QOCOFloat* Axnew,
                        QOCOFloat* Gxnew)
{
  solver->sol->status = QOCO_UNSOLVED;
  QOCOProblemData* data = solver->work->data;
  QOCOScaling* scaling = solver->work->scaling;

  set_cpu_mode(1);
  // Undo regularization.
  QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
  unregularize(Pcsc, solver->settings->kkt_static_reg);

  // Unequilibrate P.
  QOCOFloat* Px = Pcsc->x;
  QOCOInt Pnnz = get_nnz(data->P);
  QOCOFloat* Dinvruiz_data = get_data_vectorf(scaling->Dinvruiz);
  scale_arrayf(Px, Px, scaling->kinv, Pnnz);
  row_col_scale(Pcsc, Dinvruiz_data, Dinvruiz_data);

  // Unequilibrate c.
  QOCOFloat* cdata = get_data_vectorf(data->c);
  scale_arrayf(cdata, cdata, scaling->kinv, data->n);
  for (QOCOInt i = 0; i < data->n; ++i) {
    cdata[i] = Dinvruiz_data[i] * cdata[i];
  }

  // Unequilibrate A.
  QOCOCscMatrix* Acsc = get_csc_matrix(data->A);
  QOCOFloat* Einvruiz_data = get_data_vectorf(scaling->Einvruiz);
  row_col_scale(Acsc, Einvruiz_data, Dinvruiz_data);

  // Unequilibrate G.
  QOCOCscMatrix* Gcsc = get_csc_matrix(data->G);
  QOCOFloat* Finvruiz_data = get_data_vectorf(scaling->Finvruiz);
  row_col_scale(Gcsc, Finvruiz_data, Dinvruiz_data);

  // Unequilibrate b.
  QOCOFloat* bdata = get_data_vectorf(data->b);
  for (QOCOInt i = 0; i < data->p; ++i) {
    bdata[i] = Einvruiz_data[i] * bdata[i];
  }

  // Unequilibrate h.
  QOCOFloat* hdata = get_data_vectorf(data->h);
  for (QOCOInt i = 0; i < data->m; ++i) {
    hdata[i] = Finvruiz_data[i] * hdata[i];
  }

  // Update P and avoid nonzeros that were added for regularization.
  if (Pxnew) {
    QOCOInt avoid = data->Pnum_nzadded > 0 ? data->Pnzadded_idx[0] : Pnnz + 1;
    QOCOInt offset = 0;
    for (QOCOInt i = 0; i < Pnnz - data->Pnum_nzadded; ++i) {
      if (i == avoid) {
        offset++;
        avoid = data->Pnzadded_idx[offset];
      }
      else {
        Px[i + offset] = Pxnew[i];
      }
    }
  }

  // Update A.
  if (Axnew) {
    QOCOCscMatrix* Atcsc = get_csc_matrix(data->At);
    QOCOFloat* Ax = Acsc->x;
    QOCOFloat* Atx = Atcsc->x;
    QOCOInt Annz = get_nnz(data->A);
    for (QOCOInt i = 0; i < Annz; ++i) {
      Ax[i] = Axnew[i];
      Atx[i] = Axnew[data->AtoAt[i]];
    }
  }

  // Update G.
  if (Gxnew) {
    QOCOCscMatrix* Gtcsc = get_csc_matrix(data->Gt);
    QOCOFloat* Gx = Gcsc->x;
    QOCOFloat* Gtx = Gtcsc->x;
    QOCOInt Gnnz = get_nnz(data->G);
    for (QOCOInt i = 0; i < Gnnz; ++i) {
      Gx[i] = Gxnew[i];
      Gtx[i] = Gxnew[data->GtoGt[i]];
    }
  }

  compute_scaling_statistics(data);

  // Equilibrate new matrix data.
  ruiz_equilibration(solver->work->data, solver->work->scaling,
                     solver->settings->ruiz_iters);

  // Regularize P.
  unregularize(Pcsc, -solver->settings->kkt_static_reg);
  set_cpu_mode(0);

  // Sync the new data to the GPU.
  sync_matrix_to_device(data->P);
  sync_matrix_to_device(data->A);
  sync_matrix_to_device(data->G);

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
    compute_kkt_residual(
        data, get_data_vectorf(work->x), get_data_vectorf(work->y),
        get_data_vectorf(work->s), get_data_vectorf(work->z),
        get_data_vectorf(work->kktres), solver->settings->kkt_static_reg,
        get_data_vectorf(work->xyzbuff1), get_data_vectorf(work->xbuff),
        get_data_vectorf(work->ubuff1), get_data_vectorf(work->ubuff2));

    // Compute objective function.
    solver->sol->obj = compute_objective(
        data, get_data_vectorf(work->x), get_data_vectorf(work->xbuff),
        solver->settings->kkt_static_reg, work->scaling->k);

    // Compute mu = s'*z / m.
    work->mu = (data->m > 0)
                   ? safe_div(qoco_dot(get_data_vectorf(work->s),
                                       get_data_vectorf(work->z), data->m),
                              data->m)
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
    solver->linsys->linsys_update_nt(solver->linsys_data,
                                     get_data_vectorf(work->WtW),
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
  free_qoco_matrix(solver->work->data->P);
  free_qoco_matrix(solver->work->data->A);
  free_qoco_matrix(solver->work->data->G);
  free_qoco_matrix(solver->work->data->At);
  free_qoco_matrix(solver->work->data->Gt);

  qoco_free(solver->work->data->AtoAt);
  qoco_free(solver->work->data->GtoGt);
  free_qoco_vectorf(solver->work->data->b);
  free_qoco_vectorf(solver->work->data->c);
  free_qoco_vectorf(solver->work->data->h);
  free_qoco_vectori(solver->work->data->q);
  qoco_free(solver->work->data->Pnzadded_idx);
  qoco_free(solver->work->data);

  // Cleanup linsys.
  solver->linsys->linsys_cleanup(solver->linsys_data);

  // Free primal and dual variables.
  free_qoco_vectorf(solver->work->rhs);
  free_qoco_vectorf(solver->work->kktres);
  free_qoco_vectorf(solver->work->xyz);
  free_qoco_vectorf(solver->work->xyzbuff1);
  free_qoco_vectorf(solver->work->xyzbuff2);
  free_qoco_vectorf(solver->work->x);
  free_qoco_vectorf(solver->work->s);
  free_qoco_vectorf(solver->work->y);
  free_qoco_vectorf(solver->work->z);

  // Free Nesterov-Todd scalings and scaled variables.
  free_qoco_vectorf(solver->work->W);
  free_qoco_vectorf(solver->work->Wfull);
  free_qoco_vectorf(solver->work->Winv);
  free_qoco_vectorf(solver->work->Winvfull);
  free_qoco_vectorf(solver->work->WtW);
  free_qoco_vectorf(solver->work->lambda);
  free_qoco_vectorf(solver->work->sbar);
  free_qoco_vectorf(solver->work->zbar);
  free_qoco_vectorf(solver->work->xbuff);
  free_qoco_vectorf(solver->work->ybuff);
  free_qoco_vectorf(solver->work->ubuff1);
  free_qoco_vectorf(solver->work->ubuff2);
  free_qoco_vectorf(solver->work->ubuff3);
  free_qoco_vectorf(solver->work->Ds);

  // Free scaling struct.
  free_qoco_vectorf(solver->work->scaling->delta);
  free_qoco_vectorf(solver->work->scaling->Druiz);
  free_qoco_vectorf(solver->work->scaling->Eruiz);
  free_qoco_vectorf(solver->work->scaling->Fruiz);
  free_qoco_vectorf(solver->work->scaling->Dinvruiz);
  free_qoco_vectorf(solver->work->scaling->Einvruiz);
  free_qoco_vectorf(solver->work->scaling->Finvruiz);
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