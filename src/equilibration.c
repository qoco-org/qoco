#include "equilibration.h"

void ruiz_equilibration(QOCOProblemData* data, QOCOScaling* scaling,
                        QOCOInt ruiz_iters)
{
  // Enable host data mode for CUDA backend (no-op for builtin backend)
  set_scaling_statistics_mode(1);

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

  QOCOFloat* delta_data = get_data_vectorf(scaling->delta);
  QOCOFloat* cdata = get_data_vectorf(data->c);
  QOCOFloat* Druiz_data = get_data_vectorf(scaling->Druiz);
  QOCOFloat* Eruiz_data = get_data_vectorf(scaling->Eruiz);
  QOCOFloat* Fruiz_data = get_data_vectorf(scaling->Fruiz);
  QOCOFloat* bdata = get_data_vectorf(data->b);
  QOCOFloat* hdata = get_data_vectorf(data->h);

  for (QOCOInt i = 0; i < ruiz_iters; ++i) {

    // Compute infinity norm of rows of [P A' G']
    for (QOCOInt j = 0; j < data->n; ++j) {
      set_element_vectorf(scaling->delta, j, 0.0);
    }
    g = inf_norm(cdata, data->n);
    QOCOFloat Pinf_mean = 0.0;
    if (data->P) {
      col_inf_norm_USymm_matrix(data->P, delta_data);
      for (QOCOInt j = 0; j < data->n; ++j) {
        Pinf_mean += get_element_vectorf(scaling->delta, j);
      }
      Pinf_mean /= data->n;
    }

    // g = 1 / max(mean(Pinf), norm(c, "inf"));
    g = qoco_max(Pinf_mean, g);
    g = safe_div(1.0, g);
    scaling->k *= g;

    // Compute column infinity norms of A and G
    // For CSC format, column norms are computed efficiently
    QOCOFloat* Anorm = (QOCOFloat*)qoco_malloc(sizeof(QOCOFloat) * data->n);
    QOCOFloat* Gnorm = (QOCOFloat*)qoco_malloc(sizeof(QOCOFloat) * data->n);
    if (get_nnz(data->A) > 0) {
      col_inf_norm_matrix(data->A, Anorm);
      for (QOCOInt j = 0; j < data->n; ++j) {
        QOCOFloat nrm =
            qoco_max(get_element_vectorf(scaling->delta, j), Anorm[j]);
        set_element_vectorf(scaling->delta, j, nrm);
      }
    }
    if (get_nnz(data->G) > 0) {
      col_inf_norm_matrix(data->G, Gnorm);
      for (QOCOInt j = 0; j < data->n; ++j) {
        QOCOFloat nrm =
            qoco_max(get_element_vectorf(scaling->delta, j), Gnorm[j]);
        set_element_vectorf(scaling->delta, j, nrm);
      }
    }
    qoco_free(Anorm);
    qoco_free(Gnorm);

    // d(i) = 1 / sqrt(max([Pinf(i), Atinf(i), Gtinf(i)]));
    for (QOCOInt j = 0; j < data->n; ++j) {
      QOCOFloat temp = qoco_sqrt(get_element_vectorf(scaling->delta, j));
      temp = safe_div(1.0, temp);
      set_element_vectorf(scaling->delta, j, temp);
    }

    // Compute infinity norm of rows of [A 0 0].
    // For row norms, compute column norms of the transpose (At is stored in CSC
    // format)
    if (get_nnz(data->A) > 0) {
      col_inf_norm_matrix(data->At, &delta_data[data->n]);
      // d(i) = 1 / sqrt(Ainf(i));
      for (QOCOInt k = 0; k < data->p; ++k) {
        QOCOFloat temp =
            qoco_sqrt(get_element_vectorf(scaling->delta, data->n + k));
        temp = safe_div(1.0, temp);
        set_element_vectorf(scaling->delta, data->n + k, temp);
      }
    }

    // Compute infinity norm of rows of [G 0 0].
    // For row norms, compute column norms of the transpose (Gt is stored in CSC
    // format)
    if (get_nnz(data->G) > 0) {
      col_inf_norm_matrix(data->Gt, &delta_data[data->n + data->p]);
      // d(i) = 1 / sqrt(Ginf(i));
      for (QOCOInt k = 0; k < data->m; ++k) {
        QOCOFloat temp = qoco_sqrt(
            get_element_vectorf(scaling->delta, data->n + data->p + k));
        temp = safe_div(1.0, temp);
        set_element_vectorf(scaling->delta, data->n + data->p + k, temp);
      }
    }

    QOCOFloat* D = delta_data;
    QOCOFloat* E = &delta_data[data->n];
    QOCOFloat* F = &delta_data[data->n + data->p];

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
      QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
      scale_arrayf(Pcsc->x, Pcsc->x, g, get_nnz(data->P));
      row_col_scale_matrix(data->P, D, D);
    }

    // Scale c.
    scale_arrayf(cdata, cdata, g, data->n);
    ew_product(cdata, D, cdata, data->n);

    // Scale A and G.
    row_col_scale_matrix(data->A, E, D);
    row_col_scale_matrix(data->G, F, D);
    row_col_scale_matrix(data->At, D, E);
    row_col_scale_matrix(data->Gt, D, F);

    // Update scaling matrices with delta.
    ew_product(Druiz_data, D, Druiz_data, data->n);
    ew_product(Eruiz_data, E, Eruiz_data, data->p);
    ew_product(Fruiz_data, F, Fruiz_data, data->m);
  }

  // Scale b.
  ew_product(bdata, Eruiz_data, bdata, data->p);

  // Scale h.
  ew_product(hdata, Fruiz_data, hdata, data->m);

  // Compute Dinv, Einv, Finv.
  reciprocal_vectorf(scaling->Druiz, scaling->Dinvruiz);
  reciprocal_vectorf(scaling->Eruiz, scaling->Einvruiz);
  reciprocal_vectorf(scaling->Fruiz, scaling->Finvruiz);
  scaling->kinv = safe_div(1.0, scaling->k);

  // Disable host data mode for CUDA backend (no-op for builtin backend)
  set_scaling_statistics_mode(0);

  // Sync updated scaling vectors and scaled data to device (CUDA backend).
  sync_vector_to_device(scaling->Druiz);
  sync_vector_to_device(scaling->Eruiz);
  sync_vector_to_device(scaling->Fruiz);
  sync_vector_to_device(scaling->Dinvruiz);
  sync_vector_to_device(scaling->Einvruiz);
  sync_vector_to_device(scaling->Finvruiz);
  sync_vector_to_device(data->c);
  sync_vector_to_device(data->b);
  sync_vector_to_device(data->h);
}

void unscale_variables(QOCOWorkspace* work)
{
  ew_product(get_pointer_vectorf(work->x, 0),
             get_pointer_vectorf(work->scaling->Druiz, 0),
             get_pointer_vectorf(work->x, 0), work->data->n);

  ew_product(get_pointer_vectorf(work->s, 0),
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
