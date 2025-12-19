/**
 * @file cone.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */

#include "cone.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


__device__ __forceinline__
QOCOFloat qoco_max_dev(QOCOFloat a, QOCOFloat b)
{
    return a > b ? a : b;
}

__device__ __forceinline__
QOCOFloat soc_residual_dev(const QOCOFloat* u, QOCOInt n)
{
    QOCOFloat sum = 0.0;
    for (QOCOInt i = 1; i < n; ++i) {
        sum += u[i] * u[i];
    }
    return sqrt(sum) - u[0];
}

__global__ void set_Wfull_linear(QOCOFloat* W, QOCOInt Wnnzfull, QOCOInt l)
{
    QOCOInt i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < Wnnzfull) {
        W[i] = 0.0;
    }
    if (i < l) {
        W[i] = 1.0;
    }
}

__global__ void set_Wfull_soc(QOCOFloat* W, QOCOInt* q, QOCOInt nsoc, QOCOInt l)
{
    QOCOInt soc = blockIdx.x;
    if (soc >= nsoc) return;

    QOCOInt dim = q[soc];
    QOCOInt k = threadIdx.x;

    if (k >= dim) return;

    // compute starting offset of this SOC block
    QOCOInt idx = l;
    for (QOCOInt i = 0; i < soc; ++i) {
        idx += q[i] * q[i];
    }

    // diagonal element
    W[idx + k * dim + k] = 1.0;
}

__global__ void cone_residual_kernel(const QOCOFloat* u,
                          QOCOInt l,
                          QOCOInt nsoc,
                          const QOCOInt* q,
                          QOCOFloat* out)
{
  // single thread
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  QOCOFloat res = -1e7;
  QOCOInt idx = 0;

  // LP cone
  for (; idx < l; ++idx) {
      res = qoco_max_dev(res, -u[idx]);
  }

  // SOC cones
  for (QOCOInt i = 0; i < nsoc; ++i) {
      res = qoco_max_dev(res, soc_residual_dev(&u[idx], q[i]));
      idx += q[i];
  }

  *out = res;
}

__global__ void bring2cone_kernel(QOCOFloat* u, QOCOInt* q, QOCOInt l, QOCOInt nsoc)
{
    // Single-thread kernel
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    QOCOFloat a = 0.0;
    QOCOInt idx = 0;

    /* ---------- LP cone ---------- */
    for (idx = 0; idx < l; ++idx) {
        a = qoco_max(a, -u[idx]);
    }
    a = qoco_max(a, (QOCOFloat)0.0);

    /* ---------- SOC cones ---------- */
    for (QOCOInt i = 0; i < nsoc; ++i) {
        QOCOInt qi = q[i];
        QOCOFloat soc_res = soc_residual_dev(&u[idx], qi);
        if (soc_res > a) {
            a = soc_res;
        }
        idx += qi;
    }

    QOCOFloat shift = (QOCOFloat)(1.0) + a;

    /* ---------- Update LP cone ---------- */
    for (idx = 0; idx < l; ++idx) {
        u[idx] += shift;
    }

    /* ---------- Update SOC cones ---------- */
    for (QOCOInt i = 0; i < nsoc; ++i) {
        u[idx] += shift;
        idx += q[i];
    }
}

void set_Wfull_identity(QOCOVectorf* Wfull, QOCOInt Wnnzfull, QOCOProblemData* data)
{
  CUDA_CHECK(cudaGetLastError()); 
  QOCOFloat* W = get_data_vectorf(Wfull);

  const int threads = 256;
  const int blocks  = (Wnnzfull + threads - 1) / threads;

  // kernel 1: zero + linear cone
  set_Wfull_linear<<<blocks, threads>>>(
      W,
      Wnnzfull,
      data->l
  );
  CUDA_CHECK(cudaGetLastError());

  // kernel 2: SOC blocks
  const int blocks2 = data->nsoc;
  if (data->nsoc > 0)
  {
    set_Wfull_soc<<<blocks2, 256>>>(
        W,
        get_data_vectori(data->q),
        data->nsoc,
        data->l
    );
  }
  CUDA_CHECK(cudaGetLastError());
}

QOCOFloat cone_residual(const QOCOFloat* d_u, QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  QOCOFloat* d_out = nullptr;
  QOCOFloat h_out;

  // Allocate output
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(QOCOFloat)));

  cone_residual_kernel<<<1,1>>>(d_u, l, nsoc, q, d_out);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&h_out, d_out,
                        sizeof(QOCOFloat),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_out));
  return h_out;
}

void soc_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                 QOCOInt n)
{
  p[0] = qoco_dot(u, v, n);
  for (QOCOInt i = 1; i < n; ++i) {
    p[i] = u[0] * v[i] + v[0] * u[i];
  }
}

void soc_division(const QOCOFloat* lam, const QOCOFloat* v, QOCOFloat* d,
                  QOCOInt n)
{
  QOCOFloat f = lam[0] * lam[0] - qoco_dot(&lam[1], &lam[1], n - 1);
  QOCOFloat finv = safe_div(1.0, f);
  QOCOFloat lam0inv = safe_div(1.0, lam[0]);
  QOCOFloat lam1dv1 = qoco_dot(&lam[1], &v[1], n - 1);

  d[0] = finv * (lam[0] * v[0] - qoco_dot(&lam[1], &v[1], n - 1));
  for (QOCOInt i = 1; i < n; ++i) {
    d[i] = finv *
           (-lam[i] * v[0] + lam0inv * f * v[i] + lam0inv * lam1dv1 * lam[i]);
  }
}

QOCO_HD QOCOFloat soc_residual(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat res = 0;
  for (QOCOInt i = 1; i < n; ++i) {
    res += u[i] * u[i];
  }
  res = qoco_sqrt(res) - u[0];
  return res;
}

QOCOFloat soc_residual2(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat res = u[0] * u[0];
  for (QOCOInt i = 1; i < n; ++i) {
    res -= u[i] * u[i];
  }
  return res;
}

void cone_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                  QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  QOCOInt idx;
  // Compute LP cone product.
  for (idx = 0; idx < l; ++idx) {
    p[idx] = u[idx] * v[idx];
  }

  // Compute second-order cone product.
  for (QOCOInt i = 0; i < nsoc; ++i) {
    soc_product(&u[idx], &v[idx], &p[idx], q[i]);
    idx += q[i];
  }
}

void cone_division(const QOCOFloat* lambda, const QOCOFloat* v, QOCOFloat* d,
                   QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  QOCOInt idx;
  // Compute LP cone division.
  for (idx = 0; idx < l; ++idx) {
    d[idx] = safe_div(v[idx], lambda[idx]);
  }

  // Compute second-order cone division.
  for (QOCOInt i = 0; i < nsoc; ++i) {
    soc_division(&lambda[idx], &v[idx], &d[idx], q[i]);
    idx += q[i];
  }
}

void bring2cone(QOCOFloat* u, QOCOProblemData* data)
{
  QOCOFloat res = cone_residual(u, data->l, data->nsoc, get_data_vectori(data->q));
  if (res >= 0) {
    bring2cone_kernel<<<1,1>>>(u, get_data_vectori(data->q), data->l, data->nsoc);
  }
}

void nt_multiply(QOCOFloat* W, QOCOFloat* x, QOCOFloat* z, QOCOInt l, QOCOInt m,
                 QOCOInt nsoc, QOCOInt* q)
{
  // Compute product for LP cone part of W.
  for (QOCOInt i = 0; i < l; ++i) {
    z[i] = (W[i] * x[i]);
  }

  // Compute product for second-order cones.
  QOCOInt nt_idx = l;
  QOCOInt idx = l;

  // Zero out second-order cone block of result z.
  for (QOCOInt i = l; i < m; ++i) {
    z[i] = 0;
  }

  // Loop over all second-order cones.
  for (QOCOInt i = 0; i < nsoc; ++i) {
    // Loop over elements within a second-order cone.
    for (QOCOInt j = 0; j < q[i]; ++j) {
      z[idx + j] += qoco_dot(&W[nt_idx + j * q[i]], &x[idx], q[i]);
    }
    idx += q[i];
    nt_idx += q[i] * q[i];
  }
}

void compute_nt_scaling(QOCOWorkspace* work)
{
  QOCOFloat* W = get_data_vectorf(work->W);
  QOCOFloat* WtW = get_data_vectorf(work->WtW);
  QOCOFloat* Wfull = get_data_vectorf(work->Wfull);
  QOCOFloat* Winv = get_data_vectorf(work->Winv);
  QOCOFloat* Winvfull = get_data_vectorf(work->Winvfull);
  QOCOFloat* sbar = get_data_vectorf(work->sbar);
  QOCOFloat* zbar = get_data_vectorf(work->zbar);
  QOCOFloat* lambda = get_data_vectorf(work->lambda);
  // Compute Nesterov-Todd scaling for LP cone.
  QOCOInt idx;
  for (idx = 0; idx < work->data->l; ++idx) {
    WtW[idx] =
        safe_div(get_element_vectorf(work->s, idx), get_element_vectorf(work->z, idx));
    W[idx] = qoco_sqrt(WtW[idx]);
    Wfull[idx] = W[idx];
    Winv[idx] = safe_div(1.0, W[idx]);
    Winvfull[idx] = Winv[idx];
  }

  // Compute Nesterov-Todd scaling for second-order cones.
  QOCOInt nt_idx = idx;
  QOCOInt nt_idx_full = idx;
  for (QOCOInt i = 0; i < work->data->nsoc; ++i) {
    QOCOInt qi = get_element_vectori(work->data->q, i);
    // Compute normalized vectors.
    QOCOFloat s_scal =
        soc_residual2(get_pointer_vectorf(work->s, idx), qi);
    s_scal = qoco_sqrt(s_scal);
    QOCOFloat f = safe_div(1.0, s_scal);
    scale_arrayf(get_pointer_vectorf(work->s, idx), sbar, f,
                 qi);

    QOCOFloat z_scal =
        soc_residual2(get_pointer_vectorf(work->z, idx), qi);
    z_scal = qoco_sqrt(z_scal);
    f = safe_div(1.0, z_scal);
    scale_arrayf(get_pointer_vectorf(work->z, idx), zbar, f,
                 qi);

    QOCOFloat gamma = qoco_sqrt(
        0.5 * (1 + qoco_dot(sbar, zbar, qi)));

    f = safe_div(1.0, (2 * gamma));

    // Overwrite sbar with wbar.
    sbar[0] = f * (sbar[0] + zbar[0]);
    for (QOCOInt j = 1; j < qi; ++j) {
      sbar[j] = f * (sbar[j] - zbar[j]);
    }

    // Overwrite zbar with v.
    f = safe_div(1.0, qoco_sqrt(2 * (sbar[0] + 1)));
    zbar[0] = f * (sbar[0] + 1.0);
    for (QOCOInt j = 1; j < qi; ++j) {
      zbar[j] = f * sbar[j];
    }

    // Compute W for second-order cones.
    QOCOInt shift = 0;
    f = qoco_sqrt(safe_div(s_scal, z_scal));
    QOCOFloat finv = safe_div(1.0, f);
    for (QOCOInt j = 0; j < qi; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        QOCOInt full_idx1 = nt_idx_full + j * qi + k;
        QOCOInt full_idx2 = nt_idx_full + k * qi + j;
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

    // Compute WtW for second-order cones.
    shift = 0;
    for (QOCOInt j = 0; j < qi; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        WtW[nt_idx + shift] = qoco_dot(
            &Wfull[nt_idx_full + j * qi],
            &Wfull[nt_idx_full + k * qi], qi);
        shift += 1;
      }
    }

    idx += qi;
    nt_idx += (qi * qi + qi) / 2;
    nt_idx_full += qi * qi;
  }

  // Compute scaled variable lambda. lambda = W * z.
  nt_multiply(Wfull, get_pointer_vectorf(work->z, 0), lambda,
              work->data->l, work->data->m, work->data->nsoc, get_data_vectori(work->data->q));
}

void compute_centering(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOFloat* xyz = get_data_vectorf(work->xyz);
  QOCOFloat* Ds = get_data_vectorf(work->Ds);
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
  QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
  QOCOFloat* Dzaff = &xyz[work->data->n + work->data->p];
  QOCOFloat a = qoco_min(
      linesearch(get_pointer_vectorf(work->z, 0), Dzaff, 1.0, solver),
      linesearch(get_pointer_vectorf(work->s, 0), Ds, 1.0, solver));

  // Compute rho. rho = ((s + a * Ds)'*(z + a * Dz)) / (s'*z).
  qoco_axpy(Dzaff, get_pointer_vectorf(work->z, 0), ubuff1, a,
            work->data->m);
  qoco_axpy(Ds, get_pointer_vectorf(work->s, 0), ubuff2, a,
            work->data->m);
  QOCOFloat rho = qoco_dot(ubuff1, ubuff2, work->data->m) /
                  qoco_dot(get_pointer_vectorf(work->z, 0),
                           get_pointer_vectorf(work->s, 0), work->data->m);

  // Compute sigma. sigma = max(0, min(1, rho))^3.
  QOCOFloat sigma = qoco_min(1.0, rho);
  sigma = qoco_max(0.0, sigma);
  sigma = sigma * sigma * sigma;
  solver->work->sigma = sigma;
}

QOCOFloat linesearch(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                     QOCOSolver* solver)
{
  if (solver->work->data->nsoc == 0) {
    return exact_linesearch(u, Du, f, solver);
  }
  else {
    return bisection_search(u, Du, f, solver);
  }
}

QOCOFloat bisection_search(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                           QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);

  QOCOFloat al = 0.0;
  QOCOFloat au = 1.0;
  QOCOFloat a = 0.0;
  for (QOCOInt i = 0; i < solver->settings->bisect_iters; ++i) {
    a = 0.5 * (al + au);
    qoco_axpy(Du, u, ubuff1, safe_div(a, f), work->data->m);
    if (cone_residual(ubuff1, work->data->l, work->data->nsoc,
                      get_data_vectori(work->data->q)) >= 0) {
      au = a;
    }
    else {
      al = a;
    }
  }
  return al;
}

QOCOFloat exact_linesearch(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                           QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;

  QOCOFloat a = 1.0;
  QOCOFloat minval = 0;

  // Compute a for LP cones.
  for (QOCOInt i = 0; i < work->data->l; ++i) {
    if (Du[i] < minval * u[i])
      minval = Du[i] / u[i];
  }

  if (-f < minval)
    a = f;
  else
    a = -f / minval;

  return a;
}
