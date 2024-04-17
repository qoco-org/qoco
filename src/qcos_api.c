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
  solver->work->data->m = m;
  solver->work->data->n = n;
  solver->work->data->p = p;
  solver->work->data->P = new_qcos_csc_matrix(P);
  solver->work->data->A = new_qcos_csc_matrix(A);
  solver->work->data->G = new_qcos_csc_matrix(G);
  solver->work->data->c = new_qcos_vector_from_array(c, n);
  solver->work->data->b = new_qcos_vector_from_array(b, p);
  solver->work->data->h = new_qcos_vector_from_array(h, m);
  solver->work->data->q = q;
  solver->work->data->l = l;
  solver->work->data->ncones = ncones;

  solver->work->kkt = initialize_kkt(solver->work->data);
  solver->work->x = qcos_vector_calloc(n);
  solver->work->s = qcos_vector_calloc(m);
  solver->work->y = qcos_vector_calloc(p);
  solver->work->z = qcos_vector_calloc(m);

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
  settings->tol = 1e-6;
  settings->verbose = 0;
}

QCOSInt qcos_solve() { return 1; }

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
  qcos_free(solver->work->data->b->x);
  qcos_free(solver->work->data->b);
  qcos_free(solver->work->data->c->x);
  qcos_free(solver->work->data->c);
  qcos_free(solver->work->data->h->x);
  qcos_free(solver->work->data->h);
  qcos_free(solver->work->data);

  // Free primal and dual variables.
  qcos_free(solver->work->x->x);
  qcos_free(solver->work->x);
  qcos_free(solver->work->s->x);
  qcos_free(solver->work->s);
  qcos_free(solver->work->y->x);
  qcos_free(solver->work->y);
  qcos_free(solver->work->z->x);
  qcos_free(solver->work->z);

  // Free KKT matrix.
  qcos_free(solver->work->kkt->i);
  qcos_free(solver->work->kkt->p);
  qcos_free(solver->work->kkt->x);
  qcos_free(solver->work->kkt);

  qcos_free(solver->work);
  qcos_free(solver->settings);
  qcos_free(solver);

  return 1;
}