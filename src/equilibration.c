#include "equilibration.h"

void ruiz_equilibration(QCOSSolver* solver)
{
  QCOSWorkspace* work = solver->work;
  QCOSProblemData* data = solver->work->data;

  // Initialize ruiz data.
  for (QCOSInt i = 0; i < data->n; ++i) {
    work->kkt->Druiz[i] = 1.0;
    work->kkt->Dinvruiz[i] = 1.0;
  }
  for (QCOSInt i = 0; i < data->p; ++i) {
    work->kkt->Eruiz[i] = 1.0;
    work->kkt->Einvruiz[i] = 1.0;
  }
  for (QCOSInt i = 0; i < data->m; ++i) {
    work->kkt->Fruiz[i] = 1.0;
    work->kkt->Finvruiz[i] = 1.0;
  }
  QCOSFloat g = 1.0;
  work->kkt->k = 1.0;
  work->kkt->kinv = 1.0;

  for (QCOSInt i = 0; i < solver->settings->ruiz_iters; ++i) {

    // Compute infinity norm of rows of [P A' G']
    for (QCOSInt j = 0; j < data->n; ++j) {
      work->kkt->delta[j] = 1.0;
    }
    g = inf_norm(data->c, data->n);
    QCOSFloat Pinf_mean = 0.0;
    if (data->P) {
      col_inf_norm_USymm(data->P, work->kkt->delta);
      for (QCOSInt j = 0; j < data->P->n; ++j) {
        Pinf_mean += work->kkt->delta[j];
      }
      Pinf_mean /= data->n;
    }

    // g = 1 / max(mean(Pinf), norm(c, "inf"));
    g = qcos_max(Pinf_mean, g);
    g = safe_div(1.0, g);
    work->kkt->k *= g;

    if (data->A->nnz > 0) {
      for (QCOSInt j = 0; j < data->A->n; ++j) {
        QCOSFloat nrm = inf_norm(&data->A->x[data->A->p[j]],
                                 data->A->p[j + 1] - data->A->p[j]);
        work->kkt->delta[j] = qcos_max(work->kkt->delta[j], nrm);
      }
    }
    if (data->G->nnz > 0) {
      for (QCOSInt j = 0; j < data->G->n; ++j) {
        QCOSFloat nrm = inf_norm(&data->G->x[data->G->p[j]],
                                 data->G->p[j + 1] - data->G->p[j]);
        work->kkt->delta[j] = qcos_max(work->kkt->delta[j], nrm);
      }
    }

    // d(i) = 1 / sqrt(max([Pinf(i), Atinf(i), Gtinf(i)]));
    for (QCOSInt j = 0; j < data->n; ++j) {
      QCOSFloat temp = qcos_sqrt(work->kkt->delta[j]);
      temp = safe_div(1.0, temp);
      work->kkt->delta[j] = temp;
    }

    // Compute infinity norm of rows of [A 0 0].
    if (data->A->nnz > 0) {
      row_inf_norm(data->A, &work->kkt->delta[data->n]);

      // d(i) = 1 / sqrt(Ainf(i));
      for (QCOSInt k = 0; k < data->p; ++k) {
        QCOSFloat temp = qcos_sqrt(work->kkt->delta[data->n + k]);
        temp = safe_div(1.0, temp);
        work->kkt->delta[data->n + k] = temp;
      }
    }

    // Compute infinity norm of rows of [G 0 0].
    if (data->G->nnz > 0) {
      row_inf_norm(data->G, &work->kkt->delta[data->n + data->p]);

      // d(i) = 1 / sqrt(Ginf(i));
      for (QCOSInt k = 0; k < data->m; ++k) {
        QCOSFloat temp = qcos_sqrt(work->kkt->delta[data->n + data->p + k]);
        temp = safe_div(1.0, temp);
        work->kkt->delta[data->n + data->p + k] = temp;
      }
    }

    QCOSFloat* D = work->kkt->delta;
    QCOSFloat* E = &work->kkt->delta[data->n];
    QCOSFloat* F = &work->kkt->delta[data->n + data->p];

    // Make scalings for all variables in a second-order cone equal.
    QCOSInt idx = data->l;
    for (QCOSInt j = 0; j < data->nsoc; ++j) {
      for (QCOSInt k = idx + 1; k < idx + data->q[j]; ++k) {
        F[k] = F[idx];
      }
      idx += data->q[j];
    }

    // Scale P.
    if (data->P) {
      scale_arrayf(data->P->x, data->P->x, g, data->P->nnz);
      row_scale(data->P, D);
      col_scale(data->P, D);
    }

    // Scale c.
    scale_arrayf(data->c, data->c, g, data->n);
    ew_product(data->c, D, data->c, data->n);

    // Scale A.
    row_scale(data->A, E);
    col_scale(data->A, D);

    // Scale G.
    row_scale(data->G, F);
    col_scale(data->G, D);

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
  for (QCOSInt i = 0; i < data->n; ++i) {
    work->kkt->Dinvruiz[i] = safe_div(1.0, work->kkt->Druiz[i]);
  }
  for (QCOSInt i = 0; i < data->p; ++i) {
    work->kkt->Einvruiz[i] = safe_div(1.0, work->kkt->Eruiz[i]);
  }
  for (QCOSInt i = 0; i < data->m; ++i) {
    work->kkt->Finvruiz[i] = safe_div(1.0, work->kkt->Fruiz[i]);
  }
  work->kkt->kinv = safe_div(1.0, work->kkt->k);
}

void unscale_variables(QCOSWorkspace* work)
{
  ew_product(work->x, work->kkt->Druiz, work->x, work->data->n);
  ew_product(work->s, work->kkt->Finvruiz, work->s, work->data->m);

  ew_product(work->y, work->kkt->Eruiz, work->y, work->data->p);
  scale_arrayf(work->y, work->y, work->kkt->kinv, work->data->p);

  ew_product(work->z, work->kkt->Fruiz, work->z, work->data->m);
  scale_arrayf(work->z, work->z, work->kkt->kinv, work->data->m);
}