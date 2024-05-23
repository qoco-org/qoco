/**
 * @file qcos_api.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 */

#include "qcos_api.h"

QCOSInt qcos_setup(QCOSSolver* solver, QCOSInt n, QCOSInt m, QCOSInt p,
                   QCOSCscMatrix* P, QCOSFloat* c, QCOSCscMatrix* A,
                   QCOSFloat* b, QCOSCscMatrix* G, QCOSFloat* h, QCOSInt l,
                   QCOSInt nsoc, QCOSInt* q, QCOSSettings* settings)
{
  // Validate problem data.
  if (qcos_validate_data(P, c, A, b, G, h, l, nsoc, q)) {
    return qcos_error(QCOS_DATA_VALIDATION_ERROR);
  }

  // Validate settings.
  if (qcos_validate_settings(settings)) {
    return qcos_error(QCOS_SETTINGS_VALIDATION_ERROR);
  }

  solver->settings = settings;

  // Allocate workspace.
  solver->work = qcos_malloc(sizeof(QCOSWorkspace));

  // Malloc error.
  if (!(solver->work)) {
    return QCOS_MALLOC_ERROR;
  }

  solver->work->data = qcos_malloc(sizeof(QCOSProblemData));
  // Malloc error
  if (!(solver->work->data)) {
    return QCOS_MALLOC_ERROR;
  }

  // Copy problem data.
  solver->work->data->m = m;
  solver->work->data->n = n;
  solver->work->data->p = p;
  solver->work->data->A = new_qcos_csc_matrix(A);
  solver->work->data->G = new_qcos_csc_matrix(G);
  solver->work->data->c = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->data->b = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->data->h = qcos_malloc(m * sizeof(QCOSFloat));
  copy_arrayf(c, solver->work->data->c, n);
  copy_arrayf(b, solver->work->data->b, p);
  copy_arrayf(h, solver->work->data->h, m);
  solver->work->data->q = q;
  solver->work->data->l = l;
  solver->work->data->nsoc = nsoc;

  // Copy and regularize P.
  if (P) {
    solver->work->data->P = new_qcos_csc_matrix(P);
    regularize(solver->work->data->P, solver->settings->reg);
  }
  else {
    solver->work->data->P = construct_identity(n, solver->settings->reg);
  }

  // Allocate KKT struct.
  solver->work->kkt = qcos_malloc(sizeof(QCOSKKT));
  allocate_kkt(solver->work);
  solver->work->kkt->nt2kkt = qcos_calloc(solver->work->Wnnz, sizeof(QCOSInt));
  solver->work->kkt->rhs = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  solver->work->kkt->kktres = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  solver->work->kkt->xyz = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  construct_kkt(solver->work);

  // Allocate primal and dual variables.
  solver->work->x = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->s = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->y = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->z = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->mu = 0.0;

  // Allocate Nesterov-Todd scalings and scaled variables.
  QCOSInt Wnnzfull = solver->work->data->l;
  for (QCOSInt i = 0; i < solver->work->data->nsoc; ++i) {
    Wnnzfull += solver->work->data->q[i] * solver->work->data->q[i];
  }

  solver->work->W = qcos_malloc(solver->work->Wnnz * sizeof(QCOSFloat));
  solver->work->Wfull = qcos_malloc(Wnnzfull * sizeof(QCOSFloat));
  solver->work->Winv = qcos_malloc(solver->work->Wnnz * sizeof(QCOSFloat));
  solver->work->Winvfull = qcos_malloc(Wnnzfull * sizeof(QCOSFloat));
  solver->work->WtW = qcos_malloc(solver->work->Wnnz * sizeof(QCOSFloat));
  solver->work->lambda = qcos_malloc(m * sizeof(QCOSFloat));
  QCOSInt qmax = 0;
  if (solver->work->data->q) {
    qmax = max_arrayi(solver->work->data->q, solver->work->data->nsoc);
  }
  solver->work->sbar = qcos_malloc(qmax * sizeof(QCOSFloat));
  solver->work->zbar = qcos_malloc(qmax * sizeof(QCOSFloat));
  solver->work->xbuff = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->ybuff = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->ubuff1 = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->ubuff2 = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->ubuff3 = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->Ds = qcos_malloc(m * sizeof(QCOSFloat));

  // Number of columns of KKT matrix.
  QCOSInt Kn = solver->work->kkt->K->n;

  // Allocate memory for QDLDL.
  solver->work->kkt->etree = qcos_malloc(sizeof(QCOSInt) * Kn);
  solver->work->kkt->Lnz = qcos_malloc(sizeof(QCOSInt) * Kn);
  solver->work->kkt->Lp = qcos_malloc(sizeof(QCOSInt) * (Kn + 1));
  solver->work->kkt->D = qcos_malloc(sizeof(QCOSFloat) * Kn);
  solver->work->kkt->Dinv = qcos_malloc(sizeof(QCOSFloat) * Kn);
  solver->work->kkt->iwork = qcos_malloc(sizeof(QCOSInt) * 3 * Kn);
  solver->work->kkt->bwork = qcos_malloc(sizeof(unsigned char) * Kn);
  solver->work->kkt->fwork = qcos_malloc(sizeof(QCOSFloat) * Kn);

  // Compute elimination tree.
  QCOSInt sumLnz =
      QDLDL_etree(Kn, solver->work->kkt->K->p, solver->work->kkt->K->i,
                  solver->work->kkt->iwork, solver->work->kkt->Lnz,
                  solver->work->kkt->etree);
  solver->work->kkt->Li = qcos_malloc(sizeof(QCOSInt) * sumLnz);
  solver->work->kkt->Lx = qcos_malloc(sizeof(QCOSFloat) * sumLnz);

  // Allocate solution struct.
  solver->sol = qcos_malloc(sizeof(QCOSSolution));
  solver->sol->x = qcos_malloc(n * sizeof(QCOSFloat));
  solver->sol->s = qcos_malloc(m * sizeof(QCOSFloat));
  solver->sol->y = qcos_malloc(p * sizeof(QCOSFloat));
  solver->sol->z = qcos_malloc(m * sizeof(QCOSFloat));
  solver->sol->iters = 0;

  return QCOS_NO_ERROR;
}

void qcos_set_csc(QCOSCscMatrix* A, QCOSInt m, QCOSInt n, QCOSInt Annz,
                  QCOSFloat* Ax, QCOSInt* Ap, QCOSInt* Ai)
{
  A->m = m;
  A->n = n;
  A->nnz = Annz;
  A->x = Ax;
  A->p = Ap;
  A->i = Ai;
}

void set_default_settings(QCOSSettings* settings)
{
  settings->max_iters = 50;
  settings->max_iter_bisection = 5;
  settings->verbose = 0;
  settings->abstol = 1e-7;
  settings->reltol = 1e-7;
  settings->reg = 1e-8;
}

QCOSInt qcos_solve(QCOSSolver* solver)
{
  start_timer(&(solver->work->solve_timer));

  if (solver->settings->verbose) {
    print_header();
  }

  // Get initializations for primal and dual variables.
  initialize_ipm(solver);

  for (QCOSInt i = 1; i <= solver->settings->max_iters; ++i) {

    // Compute kkt residual.
    compute_kkt_residual(solver->work);

    // Compute mu.
    compute_mu(solver->work);

    // Check stopping criteria.
    if (check_stopping(solver)) {
      stop_timer(&(solver->work->solve_timer));
      copy_solution(solver);
      solver->sol->status = QCOS_SOLVED;
      if (solver->settings->verbose) {
        print_footer(solver->sol, solver->sol->status);
      }
      return QCOS_SOLVED;
    }

    // Compute Nesterov-Todd scalings.
    compute_nt_scaling(solver->work);

    // Update Nestrov-Todd block of KKT matrix.
    update_nt_block(solver->work);

    // Perform predictor-corrector.
    predictor_corrector(solver);

    // Update iteration count.
    solver->sol->iters = i;

    // Log solver progress to console if we are solving in verbose mode.
    if (solver->settings->verbose) {
      log_iter(solver);
    }
  }

  stop_timer(&(solver->work->solve_timer));
  copy_solution(solver);
  return QCOS_MAX_ITER;
}

QCOSInt qcos_cleanup(QCOSSolver* solver)
{

  // Free problem data.
  free_qcos_csc_matrix(solver->work->data->P);
  free_qcos_csc_matrix(solver->work->data->A);
  free_qcos_csc_matrix(solver->work->data->G);
  qcos_free(solver->work->data->b);
  qcos_free(solver->work->data->c);
  qcos_free(solver->work->data->h);
  qcos_free(solver->work->data);

  // Free primal and dual variables.
  qcos_free(solver->work->kkt->rhs);
  qcos_free(solver->work->kkt->kktres);
  qcos_free(solver->work->kkt->xyz);
  qcos_free(solver->work->x);
  qcos_free(solver->work->s);
  qcos_free(solver->work->y);
  qcos_free(solver->work->z);

  // Free Nesterov-Todd scalings and scaled variables.
  qcos_free(solver->work->W);
  qcos_free(solver->work->Wfull);
  qcos_free(solver->work->Winv);
  qcos_free(solver->work->Winvfull);
  qcos_free(solver->work->WtW);
  qcos_free(solver->work->lambda);
  qcos_free(solver->work->sbar);
  qcos_free(solver->work->zbar);
  qcos_free(solver->work->xbuff);
  qcos_free(solver->work->ybuff);
  qcos_free(solver->work->ubuff1);
  qcos_free(solver->work->ubuff2);
  qcos_free(solver->work->ubuff3);
  qcos_free(solver->work->Ds);

  // Free KKT struct.
  free_qcos_csc_matrix(solver->work->kkt->K);
  qcos_free(solver->work->kkt->nt2kkt);
  qcos_free(solver->work->kkt->etree);
  qcos_free(solver->work->kkt->Lnz);
  qcos_free(solver->work->kkt->Lp);
  qcos_free(solver->work->kkt->D);
  qcos_free(solver->work->kkt->Dinv);
  qcos_free(solver->work->kkt->iwork);
  qcos_free(solver->work->kkt->bwork);
  qcos_free(solver->work->kkt->fwork);
  qcos_free(solver->work->kkt->Li);
  qcos_free(solver->work->kkt->Lx);
  qcos_free(solver->work->kkt);

  // Free solution struct.
  qcos_free(solver->sol->x);
  qcos_free(solver->sol->s);
  qcos_free(solver->sol->y);
  qcos_free(solver->sol->z);
  qcos_free(solver->sol);

  qcos_free(solver->work);
  qcos_free(solver->settings);
  qcos_free(solver);

  return 1;
}