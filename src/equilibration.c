#include "equilibration.h"

void ruiz_equilibration(QOCOProblemData* data, QOCOScaling* scaling,
                        QOCOInt ruiz_iters)
{
  // Initialize ruiz data.
  for (QOCOInt i = 0; i < data->n; ++i) {
    set_element_vectorf(scaling->Druiz, i, 1.0);
    set_element_vectorf(scaling->Dinvruiz, i, 1.0);
  }
  for (QOCOInt i = 0; i < data->p; ++i) {
    set_element_vectorf(scaling->Eruiz, i, 1.0);
    set_element_vectorf(scaling->Einvruiz, i, 1.0);
  }
  for (QOCOInt i = 0; i < data->m; ++i) {
    set_element_vectorf(scaling->Fruiz, i, 1.0);
    set_element_vectorf(scaling->Finvruiz, i, 1.0);
  }
  QOCOFloat g = 1.0;
  scaling->k = 1.0;
  scaling->kinv = 1.0;

  // for (QOCOInt i = 0; i < ruiz_iters; ++i) {

  //   // Compute infinity norm of rows of [P A' G']
  //   for (QOCOInt j = 0; j < data->n; ++j) {
  //     set_element_vectorf(scaling->delta, j, 0.0);
  //   }
  //   g = inf_norm(data->c, data->n);
  //   QOCOFloat Pinf_mean = 0.0;
  //   if (data->P) {
  //     col_inf_norm_USymm(data->P, scaling->delta);
  //     for (QOCOInt j = 0; j < data->n; ++j) {
  //       Pinf_mean += get_element_vectorf(scaling->delta, j);
  //     }
  //     Pinf_mean /= data->n;
  //   }

  //   // g = 1 / max(mean(Pinf), norm(c, "inf"));
  //   g = qoco_max(Pinf_mean, g);
  //   g = safe_div(1.0, g);
  //   scaling->k *= g;

  //   // TODO: Replace all this with row_inf norm and col inf norm functions
  //   that
  //   // operate on QOCOMatrix.
  //   if (get_nnz(data->A) > 0) {
  //     for (QOCOInt j = 0; j < data->n; ++j) {
  //       QOCOFloat nrm = inf_norm(&data->A->x[data->A->p[j]],
  //                                data->A->p[j + 1] - data->A->p[j]);
  //       nrm = qoco_max(get_element_vectorf(scaling->delta, j), nrm);
  //       set_element_vectorf(scaling->delta, j, nrm);
  //     }
  //   }
  //   if (get_nnz(data->G) > 0) {
  //     for (QOCOInt j = 0; j < data->n; ++j) {
  //       QOCOFloat nrm = inf_norm(&data->G->x[data->G->p[j]],
  //                                data->G->p[j + 1] - data->G->p[j]);
  //       nrm = qoco_max(get_element_vectorf(scaling->delta, j), nrm);
  //       set_element_vectorf(scaling->delta, j, nrm);
  //     }
  //   }

  //   // d(i) = 1 / sqrt(max([Pinf(i), Atinf(i), Gtinf(i)]));
  //   for (QOCOInt j = 0; j < data->n; ++j) {
  //     QOCOFloat temp = qoco_sqrt(get_element_vectorf(scaling->delta, j));
  //     temp = safe_div(1.0, temp);
  //     set_element_vectorf(scaling->delta, j, temp);
  //   }

  //   // Compute infinity norm of rows of [A 0 0].
  //   if (get_nnz(data->A) > 0) {
  //     for (QOCOInt j = 0; j < data->p; ++j) {
  //       QOCOFloat nrm = inf_norm(&data->At->x[data->At->p[j]],
  //                                data->At->p[j + 1] - data->At->p[j]);
  //       set_element_vectorf(scaling->delta, data->n + j, nrm);
  //     }
  //     // d(i) = 1 / sqrt(Ainf(i));
  //     for (QOCOInt k = 0; k < data->p; ++k) {
  //       QOCOFloat temp =
  //           qoco_sqrt(get_element_vectorf(scaling->delta, data->n + k));
  //       temp = safe_div(1.0, temp);
  //       set_element_vectorf(scaling->delta, data->n + k, temp);
  //     }
  //   }

  //   // Compute infinity norm of rows of [G 0 0].
  //   if (get_nnz(data->G) > 0) {
  //     for (QOCOInt j = 0; j < data->m; ++j) {
  //       QOCOFloat nrm = inf_norm(&data->Gt->x[data->Gt->p[j]],
  //                                data->Gt->p[j + 1] - data->Gt->p[j]);
  //       set_element_vectorf(scaling->delta, data->n + data->p + j, nrm);
  //     }
  //     // d(i) = 1 / sqrt(Ginf(i));
  //     for (QOCOInt k = 0; k < data->m; ++k) {
  //       QOCOFloat temp = qoco_sqrt(
  //           get_element_vectorf(scaling->delta, data->n + data->p + k));
  //       temp = safe_div(1.0, temp);
  //       set_element_vectorf(scaling->delta, data->n + data->p + k, temp);
  //     }
  //   }

  //   QOCOFloat* D = scaling->delta;
  //   QOCOFloat* E = get_pointer_vectorf(scaling->delta, data->n);
  //   QOCOFloat* F = get_pointer_vectorf(scaling->delta, data->n + data->p);

  //   // Make scalings for all variables in a second-order cone equal.
  //   QOCOInt idx = data->l;
  //   for (QOCOInt j = 0; j < data->nsoc; ++j) {
  //     for (QOCOInt k = idx + 1; k < idx + data->q[j]; ++k) {
  //       F[k] = F[idx];
  //     }
  //     idx += data->q[j];
  //   }

  //   // Scale P.
  //   if (data->P) {
  //     scale_arrayf(data->P->x, data->P->x, g, get_nnz(data->P));
  //     row_col_scale(data->P, D, D);
  //   }

  //   // Scale c.
  //   scale_arrayf(data->c, data->c, g, data->n);
  //   ew_product(data->c, D, data->c, data->n);

  //   // Scale A and G.
  //   row_col_scale(data->A, E, D);
  //   row_col_scale(data->G, F, D);
  //   row_col_scale(data->At, D, E);
  //   row_col_scale(data->Gt, D, F);

  //   // Update scaling matrices with delta.
  //   ew_product(scaling->Druiz, D, scaling->Druiz, data->n);
  //   ew_product(scaling->Eruiz, E, scaling->Eruiz, data->p);
  //   ew_product(scaling->Fruiz, F, scaling->Fruiz, data->m);
  // }

  // // Scale b.
  // ew_product(data->b, scaling->Eruiz, data->b, data->p);

  // // Scale h.
  // ew_product(data->h, scaling->Fruiz, data->h, data->m);

  // // Compute Dinv, Einv, Finv.
  // reciprocal_vectorf(scaling->Druiz, scaling->Dinvruiz);
  // reciprocal_vectorf(scaling->Eruiz, scaling->Einvruiz);
  // reciprocal_vectorf(scaling->Fruiz, scaling->Finvruiz);
  // scaling->kinv = safe_div(1.0, scaling->k);
}

void unscale_variables(QOCOWorkspace* work)
{
  ew_product(get_pointer_vectorf(work->x, 0),
             get_pointer_vectorf(work->scaling->Druiz, 0),
             get_pointer_vectorf(work->x, 0), work->data->n);
  ew_product(get_pointer_vectorf(work->x, 0),
             get_pointer_vectorf(work->scaling->Finvruiz, 0),
             get_pointer_vectorf(work->s, 0), work->data->m);

  ew_product(get_pointer_vectorf(work->y, 0),
             get_pointer_vectorf(work->scaling->Eruiz, 0),
             get_pointer_vectorf(work->y, 0), work->data->p);
  scale_arrayf(get_pointer_vectorf(work->y, 0), get_pointer_vectorf(work->y, 0),
               work->scaling->kinv, work->data->p);

  ew_product(get_pointer_vectorf(work->z, 0),
             get_pointer_vectorf(work->scaling->Fruiz, 0),
             get_pointer_vectorf(work->z, 0), work->data->m);
  scale_arrayf(get_pointer_vectorf(work->z, 0), get_pointer_vectorf(work->z, 0),
               work->scaling->kinv, work->data->m);
}