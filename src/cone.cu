#include "cone.h"
#include "kkt.h"
#include "qoco_linalg.h"
#include "qoco_utils.h"
#include "structs.h"
#include "definitions.h"
#include "backend.h"
#include <math.h>
#include <stdio.h>

// This file mirrors cone.c but is compiled with nvcc for the CUDA backend.

void cone_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                  QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  cone_product_qoco(u, v, p, l, nsoc, q);
}

void cone_division(const QOCOFloat* lambda, const QOCOFloat* v, QOCOFloat* d,
                   QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  cone_division_qoco(lambda, v, d, l, nsoc, q);
}

void nt_multiply(QOCOFloat* W, QOCOFloat* x, QOCOFloat* z, QOCOInt l, QOCOInt m,
                 QOCOInt nsoc, const QOCOInt* q)
{
  nt_multiply_qoco(W, x, z, l, m, nsoc, q);
}

void compute_nt_scaling(QOCOWorkspace* work)
{
#ifdef QOCO_ALGEBRA_BACKEND_CUDA
  // Use device implementation when s/z are on device.
  QOCOFloat* s_ptr = get_data_vectorf(work->s);
  QOCOFloat* z_ptr = get_data_vectorf(work->z);
  cudaPointerAttributes attr_s, attr_z;
  int s_dev = (cudaPointerGetAttributes(&attr_s, s_ptr) == cudaSuccess &&
               attr_s.type == cudaMemoryTypeDevice);
  int z_dev = (cudaPointerGetAttributes(&attr_z, z_ptr) == cudaSuccess &&
               attr_z.type == cudaMemoryTypeDevice);
  if (s_dev && z_dev) {
    compute_nt_scaling_cuda(work->s, work->z, work->W, work->WtW, work->Wfull,
                            work->Winv, work->Winvfull, work->lambda,
                            work->sbar, work->zbar, work->data->l,
                            work->data->nsoc, work->data->q);
    return;
  }
#endif

  // Fallback to host implementation (same as cone.c logic).
  sync_vector_to_host(work->s);
  sync_vector_to_host(work->z);
  sync_vector_to_host(work->W);
  sync_vector_to_host(work->Wfull);
  sync_vector_to_host(work->Winv);
  sync_vector_to_host(work->Winvfull);
  sync_vector_to_host(work->WtW);
  sync_vector_to_host(work->lambda);
  sync_vector_to_host(work->sbar);
  sync_vector_to_host(work->zbar);

  QOCOFloat* W = get_pointer_vectorf(work->W, 0);
  QOCOFloat* WtW = get_pointer_vectorf(work->WtW, 0);
  QOCOFloat* Wfull = get_pointer_vectorf(work->Wfull, 0);
  QOCOFloat* Winv = get_pointer_vectorf(work->Winv, 0);
  QOCOFloat* Winvfull = get_pointer_vectorf(work->Winvfull, 0);
  QOCOFloat* sbar = get_pointer_vectorf(work->sbar, 0);
  QOCOFloat* zbar = get_pointer_vectorf(work->zbar, 0);

  QOCOInt idx;
  for (idx = 0; idx < work->data->l; ++idx) {
    WtW[idx] = safe_div(get_element_vectorf(work->s, idx),
                        get_element_vectorf(work->z, idx));
    W[idx] = qoco_sqrt(WtW[idx]);
    Wfull[idx] = W[idx];
    Winv[idx] = safe_div(1.0, W[idx]);
    Winvfull[idx] = Winv[idx];
  }

  QOCOInt nt_idx = idx;
  QOCOInt nt_idx_full = idx;
  for (QOCOInt i = 0; i < work->data->nsoc; ++i) {
    QOCOFloat s_scal =
        soc_residual2(get_pointer_vectorf(work->s, idx), work->data->q[i]);
    s_scal = qoco_sqrt(s_scal);
    QOCOFloat f = safe_div(1.0, s_scal);
    scale_arrayf(get_pointer_vectorf(work->s, idx), sbar, f,
                 work->data->q[i]);

    QOCOFloat z_scal =
        soc_residual2(get_pointer_vectorf(work->z, idx), work->data->q[i]);
    z_scal = qoco_sqrt(z_scal);
    f = safe_div(1.0, z_scal);
    scale_arrayf(get_pointer_vectorf(work->z, idx), zbar, f,
                 work->data->q[i]);

    QOCOFloat gamma =
        qoco_sqrt(0.5 * (1 + qoco_dot(sbar, zbar, work->data->q[i])));

    f = safe_div(1.0, (2 * gamma));

    sbar[0] = f * (sbar[0] + zbar[0]);
    for (QOCOInt j = 1; j < work->data->q[i]; ++j) {
      sbar[j] = f * (sbar[j] - zbar[j]);
    }

    f = safe_div(1.0, qoco_sqrt(2 * (sbar[0] + 1)));
    zbar[0] = f * (sbar[0] + 1.0);
    for (QOCOInt j = 1; j < work->data->q[i]; ++j) {
      zbar[j] = f * sbar[j];
    }

    QOCOInt shift = 0;
    f = qoco_sqrt(safe_div(s_scal, z_scal));
    QOCOFloat finv = safe_div(1.0, f);
    for (QOCOInt j = 0; j < work->data->q[i]; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        QOCOInt full_idx1 = nt_idx_full + j * work->data->q[i] + k;
        QOCOInt full_idx2 = nt_idx_full + k * work->data->q[i] + j;
        W[nt_idx + shift] = 2 * (zbar[k] * zbar[j]);
        if (j != 0 && k == 0) {
          Winv[nt_idx + shift] = -W[nt_idx + shift];
        }
        else {
          Winv[nt_idx + shift] = W[nt_idx + shift];
        }
        if (j == k && j == 0) {
          W[nt_idx + shift] -= 1;
          Winv[nt_idx + shift] -= 1;
        }
        else if (j == k) {
          W[nt_idx + shift] += 1;
          Winv[nt_idx + shift] += 1;
        }
        W[nt_idx + shift] *= f;
        Winv[nt_idx + shift] *= finv;
        Wfull[full_idx1] = W[nt_idx + shift];
        Wfull[full_idx2] = W[nt_idx + shift];
        Winvfull[full_idx1] = Winv[nt_idx + shift];
        Winvfull[full_idx2] = Winv[nt_idx + shift];
        shift += 1;
      }
    }

    shift = 0;
    for (QOCOInt j = 0; j < work->data->q[i]; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        WtW[nt_idx + shift] = qoco_dot(
            &Wfull[nt_idx_full + j * work->data->q[i]],
            &Wfull[nt_idx_full + k * work->data->q[i]], work->data->q[i]);
        shift += 1;
      }
    }

    idx += work->data->q[i];
    nt_idx += (work->data->q[i] * work->data->q[i] + work->data->q[i]) / 2;
    nt_idx_full += work->data->q[i] * work->data->q[i];
  }

  nt_multiply(Wfull, get_pointer_vectorf(work->z, 0),
              get_pointer_vectorf(work->lambda, 0),
              work->data->l, work->data->m, work->data->nsoc, work->data->q);

  sync_vector_to_device(work->W);
  sync_vector_to_device(work->Wfull);
  sync_vector_to_device(work->Winv);
  sync_vector_to_device(work->Winvfull);
  sync_vector_to_device(work->WtW);
  sync_vector_to_device(work->lambda);
}

