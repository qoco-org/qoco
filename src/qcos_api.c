#include "qcos_api.h"

QCOSSolver* qcos_setup(QCOSCscMatrix* P, QCOSFloat* c, QCOSCscMatrix* A,
                       QCOSFloat* b, QCOSCscMatrix* G, QCOSFloat* h, QCOSInt l,
                       QCOSInt ncones, QCOSInt* q, QCOSSettings* settings)
{
  QCOSSolver* solver = qcos_malloc(sizeof(QCOSSolver));

  // Malloc error
  if (!(solver)) {
    return NULL;
  }

  // Validate problem data.
  if (qcos_validate_data(P, c, A, b, G, h, l, ncones, q)) {
    return NULL;
  }

  // Validate settings.
  if (qcos_validate_settings(settings)) {
    return NULL;
  }

  solver->settings = settings;

  QCOSInt n = P->n;
  QCOSInt m = G->m;
  QCOSInt p = A->m;

  // Allocate workspace.
  solver->work = qcos_malloc(sizeof(QCOSWorkspace));

  // Malloc error.
  if (!(solver->work)) {
    return NULL;
  }

  // Copy problem data.
  solver->work->data = qcos_malloc(sizeof(QCOSProblemData));
  // Malloc error
  if (!(solver->work->data)) {
    return NULL;
  }

  // Copy problem data.
  solver->work->data->m = m;
  solver->work->data->n = n;
  solver->work->data->p = p;
  solver->work->data->P = new_qcos_csc_matrix(P);
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
  solver->work->data->ncones = ncones;

  // Allocate KKT struct.
  solver->work->kkt = qcos_malloc(sizeof(QCOSKKT));
  solver->work->kkt->nt2kkt = qcos_calloc(m, sizeof(QCOSInt));
  solver->work->kkt->K = allocate_kkt(solver->work->data);
  solver->work->kkt->rhs = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  solver->work->kkt->xyz = qcos_malloc((n + m + p) * sizeof(QCOSFloat));
  construct_kkt(solver->work);

  // Allocate primal and dual variables.
  solver->work->x = qcos_malloc(n * sizeof(QCOSFloat));
  solver->work->s = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->y = qcos_malloc(p * sizeof(QCOSFloat));
  solver->work->z = qcos_malloc(m * sizeof(QCOSFloat));
  solver->work->mu = 0.0;

  QCOSInt Kn = solver->work->kkt->K->n;
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

  return solver;
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
  settings->verbose = 0;
}

QCOSInt qcos_solve(QCOSSolver* solver)
{
  if (solver->settings->verbose) {
    print_header();
  }

  // Get initializations for primal and dual variables.
  initialize_ipm(solver);

  for (QCOSInt i = 0; i < 1; ++i) {

    // Compute kkt residual.
    compute_kkt_residual(solver->work);

    // Compute mu.
    compute_mu(solver->work);

    // // Check stopping criteria.
    // if (check_stopping(solver)) {
    //   break;
    // }

    // // Compute Nesterov-Todd scalings.
    // compute_nt_scaling(solver->work);

    // // Perform predictor-corrector
    // predictor_corrector(solver->work);

    // if (solver->settings->verbose) {
    //   log_iter(solver->work);
    // }
  }

  return 0;
}

QCOSInt qcos_cleanup(QCOSSolver* solver)
{

  // Free problem data.
  qcos_free(solver->work->data->A->i);
  qcos_free(solver->work->data->A->p);
  qcos_free(solver->work->data->A->x);
  qcos_free(solver->work->data->A);
  qcos_free(solver->work->data->G->i);
  qcos_free(solver->work->data->G->p);
  qcos_free(solver->work->data->G->x);
  qcos_free(solver->work->data->G);
  qcos_free(solver->work->data->P->i);
  qcos_free(solver->work->data->P->p);
  qcos_free(solver->work->data->P->x);
  qcos_free(solver->work->data->P);
  qcos_free(solver->work->data->b);
  qcos_free(solver->work->data->c);
  qcos_free(solver->work->data->h);
  qcos_free(solver->work->data);

  // Free primal and dual variables.
  qcos_free(solver->work->kkt->rhs);
  qcos_free(solver->work->kkt->xyz);
  qcos_free(solver->work->x);
  qcos_free(solver->work->s);
  qcos_free(solver->work->y);
  qcos_free(solver->work->z);

  // Free KKT struct.
  qcos_free(solver->work->kkt->K->i);
  qcos_free(solver->work->kkt->K->p);
  qcos_free(solver->work->kkt->K->x);
  qcos_free(solver->work->kkt->K);
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

  qcos_free(solver->work);
  qcos_free(solver->settings);
  qcos_free(solver);

  return 1;
}