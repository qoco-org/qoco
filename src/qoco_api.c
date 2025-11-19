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
  solver->work->data->A = new_qoco_matrix(A);
  solver->work->data->G = new_qoco_matrix(G);
  data->q = qoco_malloc(nsoc * sizeof(QOCOInt));

  copy_arrayi(q, data->q, nsoc);

  solver->work->data->c = new_qoco_vectorf(c, n);
  solver->work->data->b = new_qoco_vectorf(b, p);
  solver->work->data->h = new_qoco_vectorf(h, m);

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

  QOCOCscMatrix* Atcsc =
      create_transposed_matrix(get_csc_matrix(data->A), data->AtoAt);
  solver->work->data->At = new_qoco_matrix(Atcsc);
  free_qoco_csc_matrix(Atcsc);
  QOCOCscMatrix* Gtcsc =
      create_transposed_matrix(get_csc_matrix(data->G), data->GtoGt);
  solver->work->data->Gt = new_qoco_matrix(Gtcsc);
  free_qoco_csc_matrix(Gtcsc);
  ruiz_equilibration(data, solver->work->scaling, solver->settings->ruiz_iters);

  // Regularize P.
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
  solver->work->rhs = new_qoco_vectorf(NULL, n + m + p);
  solver->work->kktres = qoco_malloc((n + m + p) * sizeof(QOCOFloat));
  solver->work->xyz = new_qoco_vectorf(NULL, n + m + p);
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
    QOCOFloat* cdata = get_data_vectorf(data->c);
    QOCOFloat* Druiz_data = get_data_vectorf(solver->work->scaling->Druiz);
    for (QOCOInt i = 0; i < data->n; ++i) {
      cdata[i] = solver->work->scaling->k * Druiz_data[i] * cnew[i];
    }
  }

  // Update equality constraint vector.
  if (bnew) {
    QOCOFloat* bdata = get_data_vectorf(data->b);
    QOCOFloat* Eruiz_data = get_data_vectorf(solver->work->scaling->Eruiz);
    for (QOCOInt i = 0; i < data->p; ++i) {
      bdata[i] = Eruiz_data[i] * bnew[i];
    }
  }

  // Update conic constraint vector.
  if (hnew) {
    QOCOFloat* hdata = get_data_vectorf(data->h);
    QOCOFloat* Fruiz_data = get_data_vectorf(solver->work->scaling->Fruiz);
    for (QOCOInt i = 0; i < data->m; ++i) {
      hdata[i] = Fruiz_data[i] * hnew[i];
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
  ew_product(cdata, Dinvruiz_data, cdata, data->n);

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
  ew_product(bdata, Einvruiz_data, bdata, data->p);

  // Unequilibrate h.
  QOCOFloat* hdata = get_data_vectorf(data->h);
  ew_product(hdata, Finvruiz_data, hdata, data->m);

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

  // Equilibrate new matrix data.
  ruiz_equilibration(solver->work->data, solver->work->scaling,
                     solver->settings->ruiz_iters);

  // Regularize P.
  unregularize(get_csc_matrix(data->P), -solver->settings->kkt_static_reg);

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

// Set solve phase flag for CUDA backend (prevents CPU-GPU copies during solve)
// During solve phase, get_data_vectorf returns device pointers automatically
#ifdef QOCO_ALGEBRA_BACKEND_CUDA
  extern void set_solve_phase(int active);
  set_solve_phase(1);
#endif

  // Get initializations for primal and dual variables.
  initialize_ipm(solver);
  for (QOCOInt i = 1; i <= solver->settings->max_iters; ++i) {

    // Compute kkt residual.
    compute_kkt_residual(data, get_data_vectorf(work->x),
                         get_data_vectorf(work->y), get_data_vectorf(work->s),
                         get_data_vectorf(work->z), work->kktres,
                         solver->settings->kkt_static_reg, work->xyzbuff1,
                         work->xbuff, work->ubuff1, work->ubuff2);

    // Compute objective function.
    solver->sol->obj =
        compute_objective(data, get_data_vectorf(work->x), work->xbuff,
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
// Clear solve phase flag before copying solution (allows copy from device to
// host)
#ifdef QOCO_ALGEBRA_BACKEND_CUDA
      extern void set_solve_phase(int active);
      set_solve_phase(0);
#endif
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
// Clear solve phase flag before copying solution (allows copy from device to
// host)
#ifdef QOCO_ALGEBRA_BACKEND_CUDA
  extern void set_solve_phase(int active);
  set_solve_phase(0);
#endif
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
  if (solver->work->data->P) {
    free_qoco_matrix(solver->work->data->P);
  }
  free_qoco_matrix(solver->work->data->A);
  free_qoco_matrix(solver->work->data->G);
  free_qoco_matrix(solver->work->data->At);
  free_qoco_matrix(solver->work->data->Gt);

  qoco_free(solver->work->data->AtoAt);
  qoco_free(solver->work->data->GtoGt);
  free_qoco_vectorf(solver->work->data->b);
  free_qoco_vectorf(solver->work->data->c);
  free_qoco_vectorf(solver->work->data->h);
  qoco_free(solver->work->data->q);
  qoco_free(solver->work->data->Pnzadded_idx);
  qoco_free(solver->work->data);

  // Cleanup linsys.
  solver->linsys->linsys_cleanup(solver->linsys_data);

  // Free primal and dual variables.
  free_qoco_vectorf(solver->work->rhs);
  qoco_free(solver->work->kktres);
  free_qoco_vectorf(solver->work->xyz);
  qoco_free(solver->work->xyzbuff1);
  qoco_free(solver->work->xyzbuff2);
  free_qoco_vectorf(solver->work->x);
  free_qoco_vectorf(solver->work->s);
  free_qoco_vectorf(solver->work->y);
  free_qoco_vectorf(solver->work->z);

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