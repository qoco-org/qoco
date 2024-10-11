#include "equilibration.h"

void ruiz_equilibration(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOProblemData* data = solver->work->data;

  // Initialize ruiz data.
  for (QOCOInt i = 0; i < data->n; ++i) {
    work->kkt->Druiz[i] = 1.0;
    work->kkt->Dinvruiz[i] = 1.0;
  }
  for (QOCOInt i = 0; i < data->p; ++i) {
    work->kkt->Eruiz[i] = 1.0;
    work->kkt->Einvruiz[i] = 1.0;
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    work->kkt->Fruiz[i] = 1.0;
    work->kkt->Finvruiz[i] = 1.0;
  }
  QOCOFloat g = 1.0;
  work->kkt->k = 1.0;
  work->kkt->kinv = 1.0;

  for (QOCOInt i = 0; i < solver->settings->ruiz_iters; ++i) {

    // Compute infinity norm of rows of [P A' G']
    for (QOCOInt j = 0; j < data->n; ++j) {
      work->kkt->delta[j] = 0.0;
    }
    g = inf_norm(data->c, data->n);
    QOCOFloat Pinf_mean = 0.0;
    if (data->P) {
      col_inf_norm_USymm(data->P, work->kkt->delta);
      for (QOCOInt j = 0; j < data->P->n; ++j) {
        Pinf_mean += work->kkt->delta[j];
      }
      Pinf_mean /= data->n;
    }

    // g = 1 / max(mean(Pinf), norm(c, "inf"));
    g = qoco_max(Pinf_mean, g);
    g = safe_div(1.0, g);
    work->kkt->k *= g;

    if (data->A->nnz > 0) {
      for (QOCOInt j = 0; j < data->A->n; ++j) {
        QOCOFloat nrm = inf_norm(&data->A->x[data->A->p[j]],
                                 data->A->p[j + 1] - data->A->p[j]);
        work->kkt->delta[j] = qoco_max(work->kkt->delta[j], nrm);
      }
    }
    if (data->G->nnz > 0) {
      for (QOCOInt j = 0; j < data->G->n; ++j) {
        QOCOFloat nrm = inf_norm(&data->G->x[data->G->p[j]],
                                 data->G->p[j + 1] - data->G->p[j]);
        work->kkt->delta[j] = qoco_max(work->kkt->delta[j], nrm);
      }
    }

    // d(i) = 1 / sqrt(max([Pinf(i), Atinf(i), Gtinf(i)]));
    for (QOCOInt j = 0; j < data->n; ++j) {
      QOCOFloat temp = qoco_sqrt(work->kkt->delta[j]);
      temp = safe_div(1.0, temp);
      work->kkt->delta[j] = temp;
    }

    // Compute infinity norm of rows of [A 0 0].
    if (data->A->nnz > 0) {
      for (QOCOInt j = 0; j < solver->work->data->At->n; ++j) {
        QOCOFloat nrm = inf_norm(
            &solver->work->data->At->x[solver->work->data->At->p[j]],
            solver->work->data->At->p[j + 1] - solver->work->data->At->p[j]);
        work->kkt->delta[data->n + j] = nrm;
      }
      // d(i) = 1 / sqrt(Ainf(i));
      for (QOCOInt k = 0; k < data->p; ++k) {
        QOCOFloat temp = qoco_sqrt(work->kkt->delta[data->n + k]);
        temp = safe_div(1.0, temp);
        work->kkt->delta[data->n + k] = temp;
      }
    }

    // Compute infinity norm of rows of [G 0 0].
    if (data->G->nnz > 0) {
      for (QOCOInt j = 0; j < solver->work->data->Gt->n; ++j) {
        QOCOFloat nrm = inf_norm(
            &solver->work->data->Gt->x[solver->work->data->Gt->p[j]],
            solver->work->data->Gt->p[j + 1] - solver->work->data->Gt->p[j]);
        work->kkt->delta[data->n + data->p + j] = nrm;
      }
      // d(i) = 1 / sqrt(Ginf(i));
      for (QOCOInt k = 0; k < data->m; ++k) {
        QOCOFloat temp = qoco_sqrt(work->kkt->delta[data->n + data->p + k]);
        temp = safe_div(1.0, temp);
        work->kkt->delta[data->n + data->p + k] = temp;
      }
    }

    QOCOFloat* D = work->kkt->delta;
    QOCOFloat* E = &work->kkt->delta[data->n];
    QOCOFloat* F = &work->kkt->delta[data->n + data->p];

    // Make scalings for all variables in a second-order cone equal.
    QOCOInt idx = data->l;
    for (QOCOInt j = 0; j < data->nsoc; ++j) {
      for (QOCOInt k = idx + 1; k < idx + data->q[j]; ++k) {
        F[k] = F[idx];
      }
      idx += data->q[j];
    }

    // Scale P.
    if (data->P) {
      scale_arrayf(data->P->x, data->P->x, g, data->P->nnz);
      row_col_scale(data->P, D, D);
    }

    // Scale c.
    scale_arrayf(data->c, data->c, g, data->n);
    ew_product(data->c, D, data->c, data->n);

    // Scale A and G.
    row_col_scale(data->A, E, D);
    row_col_scale(data->G, F, D);
    row_col_scale(solver->work->data->At, D, E);
    row_col_scale(solver->work->data->Gt, D, F);

    // Update scaling matrices with delta.
    ew_product(work->kkt->Druiz, D, work->kkt->Druiz, data->n);
    ew_product(work->kkt->Eruiz, E, work->kkt->Eruiz, data->p);
    ew_product(work->kkt->Fruiz, F, work->kkt->Fruiz, data->m);
  }

  // Scale b.
  ew_product(data->b, work->kkt->Eruiz, data->b, data->p);

  // Scale h.
  ew_product(data->h, work->kkt->Fruiz, data->h, data->m);

  // Compute Dinv, Einv, Finv.
  for (QOCOInt i = 0; i < data->n; ++i) {
    work->kkt->Dinvruiz[i] = safe_div(1.0, work->kkt->Druiz[i]);
  }
  for (QOCOInt i = 0; i < data->p; ++i) {
    work->kkt->Einvruiz[i] = safe_div(1.0, work->kkt->Eruiz[i]);
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    work->kkt->Finvruiz[i] = safe_div(1.0, work->kkt->Fruiz[i]);
  }
  work->kkt->kinv = safe_div(1.0, work->kkt->k);
}

void unscale_variables(QOCOWorkspace* work)
{
  ew_product(work->x, work->kkt->Druiz, work->x, work->data->n);
  ew_product(work->s, work->kkt->Finvruiz, work->s, work->data->m);

  ew_product(work->y, work->kkt->Eruiz, work->y, work->data->p);
  scale_arrayf(work->y, work->y, work->kkt->kinv, work->data->p);

  ew_product(work->z, work->kkt->Fruiz, work->z, work->data->m);
  scale_arrayf(work->z, work->z, work->kkt->kinv, work->data->m);
}