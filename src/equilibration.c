#include "equilibration.h"

void ruiz_equilibration(QOCOProblemData* data, QOCOScaling* scaling,
                        QOCOInt ruiz_iters)
{
  // This function runs on the CPU.
  set_cpu_mode(1);
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

    // --- Step 1: Compute D = diag(delta[0..n-1]) ---
    // d(j) = 1 / sqrt(max(||col j of P||_inf, ||col j of A||_inf, ||col j of
    // G||_inf))
    for (QOCOInt j = 0; j < data->n; ++j) {
      set_element_vectorf(scaling->delta, j, 0.0);
    }
    if (data->P) {
      col_inf_norm_USymm_matrix(data->P, delta_data);
    }

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

    for (QOCOInt j = 0; j < data->n; ++j) {
      QOCOFloat temp = qoco_sqrt(get_element_vectorf(scaling->delta, j));
      temp = safe_div(1.0, temp);
      set_element_vectorf(scaling->delta, j, temp);
    }

    // --- Step 2: Compute E = diag(delta[n..n+p-1]) ---
    // e(k) = 1 / sqrt(||row k of A||_inf)
    if (get_nnz(data->A) > 0) {
      col_inf_norm_matrix(data->At, &delta_data[data->n]);
      for (QOCOInt k = 0; k < data->p; ++k) {
        QOCOFloat temp =
            qoco_sqrt(get_element_vectorf(scaling->delta, data->n + k));
        temp = safe_div(1.0, temp);
        set_element_vectorf(scaling->delta, data->n + k, temp);
      }
    }

    // --- Step 3: Compute F = diag(delta[n+p..n+p+m-1]) ---
    // f(k) = 1 / sqrt(||row k of G||_inf)
    if (get_nnz(data->G) > 0) {
      col_inf_norm_matrix(data->Gt, &delta_data[data->n + data->p]);
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
      QOCOInt qj = get_element_vectori(data->q, j);
      for (QOCOInt k = idx + 1; k < idx + qj; ++k) {
        F[k] = F[idx];
      }
      idx += qj;
    }

    // --- Step 4: Apply Ruiz step (D, E, F scaling) to all matrices ---
    // Scale P by D*P*D.
    if (data->P) {
      row_col_scale_matrix(data->P, D, D);
    }

    // Scale c by D.
    ew_product(cdata, D, cdata, data->n);

    // Scale A and G.
    row_col_scale_matrix(data->A, E, D);
    row_col_scale_matrix(data->G, F, D);
    row_col_scale_matrix(data->At, D, E);
    row_col_scale_matrix(data->Gt, D, F);

    // Update cumulative scaling matrices.
    ew_product(Druiz_data, D, Druiz_data, data->n);
    ew_product(Eruiz_data, E, Eruiz_data, data->p);
    ew_product(Fruiz_data, F, Fruiz_data, data->m);

    // --- Step 5: Compute cost scaling g from the now-D-scaled P and c ---
    // As per Algorithm 2 of the OSQP paper, gamma is computed after the Ruiz
    // step so that it normalises the already-scaled quantities.
    // delta[0..n-1] is no longer needed for D, so reuse it as scratch.
    QOCOFloat Pinf_mean = 0.0;
    if (data->P) {
      col_inf_norm_USymm_matrix(data->P, delta_data);
      for (QOCOInt j = 0; j < data->n; ++j) {
        Pinf_mean += delta_data[j];
      }
      Pinf_mean /= data->n;
    }

    // g = 1 / max(mean(||P_i||_inf), ||c||_inf)
    g = qoco_max(Pinf_mean, inf_norm(cdata, data->n));
    g = safe_div(1.0, g);
    scaling->k *= g;

    // --- Step 6: Apply cost scaling g to P and c ---
    if (data->P) {
      QOCOCscMatrix* Pcsc = get_csc_matrix(data->P);
      scale_arrayf(Pcsc->x, Pcsc->x, g, get_nnz(data->P));
    }
    scale_arrayf(cdata, cdata, g, data->n);
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
  set_cpu_mode(0);

  // Sync updated scaling vectors and scaled data to device (CUDA backend).
  if (data->P) {
    sync_matrix_to_device(data->P);
  }
  if (data->p > 0) {
    sync_matrix_to_device(data->A);
    sync_matrix_to_device(data->At);
  }
  if (data->m > 0) {
    sync_matrix_to_device(data->G);
    sync_matrix_to_device(data->Gt);
  }
  sync_vector_to_device(scaling->Druiz);
  sync_vector_to_device(scaling->Eruiz);
  sync_vector_to_device(scaling->Fruiz);
  sync_vector_to_device(scaling->Dinvruiz);
  sync_vector_to_device(scaling->Einvruiz);
  sync_vector_to_device(scaling->Finvruiz);
}

void unscale_variables(QOCOWorkspace* work)
{
  ew_product(get_data_vectorf(work->x), get_data_vectorf(work->scaling->Druiz),
             get_data_vectorf(work->x), work->data->n);

  ew_product(get_data_vectorf(work->s),
             get_data_vectorf(work->scaling->Finvruiz),
             get_data_vectorf(work->s), work->data->m);

  ew_product(get_data_vectorf(work->y), get_data_vectorf(work->scaling->Eruiz),
             get_data_vectorf(work->y), work->data->p);
  scale_arrayf(get_data_vectorf(work->y), get_data_vectorf(work->y),
               work->scaling->kinv, work->data->p);

  ew_product(get_data_vectorf(work->z), get_data_vectorf(work->scaling->Fruiz),
             get_data_vectorf(work->z), work->data->m);
  scale_arrayf(get_data_vectorf(work->z), get_data_vectorf(work->z),
               work->scaling->kinv, work->data->m);
}
