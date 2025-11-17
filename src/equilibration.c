#include "equilibration.h"

void ruiz_equilibration(QOCOProblemData* data, QOCOKKT* kkt, QOCOInt ruiz_iters)
{
  // Initialize ruiz data.
  for (QOCOInt i = 0; i < data->n; ++i) {
    kkt->Druiz[i] = 1.0;
    kkt->Dinvruiz[i] = 1.0;
  }
  for (QOCOInt i = 0; i < data->p; ++i) {
    kkt->Eruiz[i] = 1.0;
    kkt->Einvruiz[i] = 1.0;
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    kkt->Fruiz[i] = 1.0;
    kkt->Finvruiz[i] = 1.0;
  }
  QOCOFloat g = 1.0;
  kkt->k = 1.0;
  kkt->kinv = 1.0;

  for (QOCOInt i = 0; i < ruiz_iters; ++i) {

    // Compute infinity norm of rows of [P A' G']
    for (QOCOInt j = 0; j < data->n; ++j) {
      kkt->delta[j] = 0.0;
    }
    g = inf_norm(data->c, data->n);
    QOCOFloat Pinf_mean = 0.0;
    if (data->P) {
      col_inf_norm_USymm(data->P, kkt->delta);
      for (QOCOInt j = 0; j < data->P->n; ++j) {
        Pinf_mean += kkt->delta[j];
      }
      Pinf_mean /= data->n;
    }

    // g = 1 / max(mean(Pinf), norm(c, "inf"));
    g = qoco_max(Pinf_mean, g);
    g = safe_div(1.0, g);
    kkt->k *= g;

    if (data->A->nnz > 0) {
      for (QOCOInt j = 0; j < data->A->n; ++j) {
        QOCOFloat nrm = inf_norm(&data->A->x[data->A->p[j]],
                                 data->A->p[j + 1] - data->A->p[j]);
        kkt->delta[j] = qoco_max(kkt->delta[j], nrm);
      }
    }
    if (data->G->nnz > 0) {
      for (QOCOInt j = 0; j < data->G->n; ++j) {
        QOCOFloat nrm = inf_norm(&data->G->x[data->G->p[j]],
                                 data->G->p[j + 1] - data->G->p[j]);
        kkt->delta[j] = qoco_max(kkt->delta[j], nrm);
      }
    }

    // d(i) = 1 / sqrt(max([Pinf(i), Atinf(i), Gtinf(i)]));
    for (QOCOInt j = 0; j < data->n; ++j) {
      QOCOFloat temp = qoco_sqrt(kkt->delta[j]);
      temp = safe_div(1.0, temp);
      kkt->delta[j] = temp;
    }

    // Compute infinity norm of rows of [A 0 0].
    if (data->A->nnz > 0) {
      for (QOCOInt j = 0; j < data->At->n; ++j) {
        QOCOFloat nrm = inf_norm(&data->At->x[data->At->p[j]],
                                 data->At->p[j + 1] - data->At->p[j]);
        kkt->delta[data->n + j] = nrm;
      }
      // d(i) = 1 / sqrt(Ainf(i));
      for (QOCOInt k = 0; k < data->p; ++k) {
        QOCOFloat temp = qoco_sqrt(kkt->delta[data->n + k]);
        temp = safe_div(1.0, temp);
        kkt->delta[data->n + k] = temp;
      }
    }

    // Compute infinity norm of rows of [G 0 0].
    if (data->G->nnz > 0) {
      for (QOCOInt j = 0; j < data->Gt->n; ++j) {
        QOCOFloat nrm = inf_norm(&data->Gt->x[data->Gt->p[j]],
                                 data->Gt->p[j + 1] - data->Gt->p[j]);
        kkt->delta[data->n + data->p + j] = nrm;
      }
      // d(i) = 1 / sqrt(Ginf(i));
      for (QOCOInt k = 0; k < data->m; ++k) {
        QOCOFloat temp = qoco_sqrt(kkt->delta[data->n + data->p + k]);
        temp = safe_div(1.0, temp);
        kkt->delta[data->n + data->p + k] = temp;
      }
    }

    QOCOFloat* D = kkt->delta;
    QOCOFloat* E = &kkt->delta[data->n];
    QOCOFloat* F = &kkt->delta[data->n + data->p];

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
    row_col_scale(data->At, D, E);
    row_col_scale(data->Gt, D, F);

    // Update scaling matrices with delta.
    ew_product(kkt->Druiz, D, kkt->Druiz, data->n);
    ew_product(kkt->Eruiz, E, kkt->Eruiz, data->p);
    ew_product(kkt->Fruiz, F, kkt->Fruiz, data->m);
  }

  // Scale b.
  ew_product(data->b, kkt->Eruiz, data->b, data->p);

  // Scale h.
  ew_product(data->h, kkt->Fruiz, data->h, data->m);

  // Compute Dinv, Einv, Finv.
  for (QOCOInt i = 0; i < data->n; ++i) {
    kkt->Dinvruiz[i] = safe_div(1.0, kkt->Druiz[i]);
  }
  for (QOCOInt i = 0; i < data->p; ++i) {
    kkt->Einvruiz[i] = safe_div(1.0, kkt->Eruiz[i]);
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    kkt->Finvruiz[i] = safe_div(1.0, kkt->Fruiz[i]);
  }
  kkt->kinv = safe_div(1.0, kkt->k);
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