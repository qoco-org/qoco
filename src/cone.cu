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

__device__ __forceinline__ QOCOFloat qoco_max_dev(QOCOFloat a, QOCOFloat b)
{
  return a > b ? a : b;
}

__device__ QOCOFloat soc_residual(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat sum = 0.0;
  for (QOCOInt i = 1; i < n; ++i) {
    sum += u[i] * u[i];
  }
  return sqrt(sum) - u[0];
}

__device__ QOCOFloat soc_residual2(const QOCOFloat* u, QOCOInt n)
{
  QOCOFloat res = u[0] * u[0];
  for (QOCOInt i = 1; i < n; ++i) {
    res -= u[i] * u[i];
  }
  return res;
}

__device__ void scale_arrayf_dev(const QOCOFloat* x, QOCOFloat* y, QOCOFloat s,
                                 QOCOInt n)
{
  for (QOCOInt i = 0; i < n; ++i) {
    y[i] = s * x[i];
  }
}

__device__ QOCOFloat qoco_dot_dev(const QOCOFloat* u, const QOCOFloat* v,
                                  QOCOInt n)
{
  QOCOFloat x = 0.0;
  for (QOCOInt i = 0; i < n; ++i) {
    x += u[i] * v[i];
  }
  return x;
}

__device__ void soc_product(const QOCOFloat* u, const QOCOFloat* v,
                            QOCOFloat* p, QOCOInt n)
{
  p[0] = qoco_dot_dev(u, v, n);
  for (QOCOInt i = 1; i < n; ++i) {
    p[i] = u[0] * v[i] + v[0] * u[i];
  }
}

__device__ void soc_division(const QOCOFloat* lam, const QOCOFloat* v,
                             QOCOFloat* d, QOCOInt n)
{
  QOCOFloat f = lam[0] * lam[0] - qoco_dot_dev(&lam[1], &lam[1], n - 1);
  QOCOFloat finv = safe_div(1.0, f);
  QOCOFloat lam0inv = safe_div(1.0, lam[0]);
  QOCOFloat lam1dv1 = qoco_dot_dev(&lam[1], &v[1], n - 1);

  d[0] = finv * (lam[0] * v[0] - qoco_dot_dev(&lam[1], &v[1], n - 1));
  for (QOCOInt i = 1; i < n; ++i) {
    d[i] = finv *
           (-lam[i] * v[0] + lam0inv * f * v[i] + lam0inv * lam1dv1 * lam[i]);
  }
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
  if (soc >= nsoc)
    return;

  QOCOInt dim = q[soc];
  QOCOInt k = threadIdx.x;

  if (k >= dim)
    return;

  // compute starting offset of this SOC block
  QOCOInt idx = l;
  for (QOCOInt i = 0; i < soc; ++i) {
    idx += q[i] * q[i];
  }

  // diagonal element
  W[idx + k * dim + k] = 1.0;
}

__global__ void cone_residual_kernel(const QOCOFloat* u, QOCOInt l,
                                     QOCOInt nsoc, const QOCOInt* q,
                                     QOCOFloat* out)
{
  // single thread
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  QOCOFloat res = -1e7;
  QOCOInt idx = 0;

  // LP cone
  for (; idx < l; ++idx) {
    res = qoco_max_dev(res, -u[idx]);
  }

  // SOC cones
  for (QOCOInt i = 0; i < nsoc; ++i) {
    res = qoco_max_dev(res, soc_residual(&u[idx], q[i]));
    idx += q[i];
  }

  *out = res;
}

__global__ void bring2cone_kernel(QOCOFloat* u, QOCOInt* q, QOCOInt l,
                                  QOCOInt nsoc)
{
  // Single-thread kernel
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

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
    QOCOFloat soc_res = soc_residual(&u[idx], qi);
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

__global__ void compute_nt_scaling_kernel(QOCOFloat* W, QOCOFloat* WtW,
                                          QOCOFloat* Wfull, QOCOFloat* Winv,
                                          QOCOFloat* Winvfull, QOCOFloat* s,
                                          QOCOFloat* z, QOCOFloat* sbar,
                                          QOCOFloat* zbar, QOCOInt l,
                                          QOCOInt nsoc, QOCOInt* q)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  QOCOInt idx;

  /* ================= LP cone ================= */
  for (idx = 0; idx < l; ++idx) {
    WtW[idx] = safe_div(s[idx], z[idx]);
    W[idx] = qoco_sqrt(WtW[idx]);
    Wfull[idx] = W[idx];

    Winv[idx] = safe_div((QOCOFloat)1.0, W[idx]);
    Winvfull[idx] = Winv[idx];
  }

  /* ================= SOC cones ================= */
  QOCOInt nt_idx = idx;
  QOCOInt nt_idx_full = idx;

  for (QOCOInt i = 0; i < nsoc; ++i) {

    QOCOInt qi = q[i];

    /* --- normalize s --- */
    QOCOFloat s_scal = soc_residual2(&s[idx], qi);
    s_scal = qoco_sqrt(s_scal);
    QOCOFloat f = safe_div((QOCOFloat)1.0, s_scal);
    scale_arrayf_dev(&s[idx], sbar, f, qi);

    /* --- normalize z --- */
    QOCOFloat z_scal = soc_residual2(&z[idx], qi);
    z_scal = qoco_sqrt(z_scal);
    f = safe_div((QOCOFloat)1.0, z_scal);
    scale_arrayf_dev(&z[idx], zbar, f, qi);

    /* --- compute gamma --- */
    QOCOFloat gamma = qoco_sqrt(
        (QOCOFloat)0.5 * ((QOCOFloat)1.0 + qoco_dot_dev(sbar, zbar, qi)));

    // For some unknown reason, when I replace the line below with
    // safe_div(1.0, 2.0 * gamma), when gamma=1.001301, we expect f=0.499350,
    // but we get f=0.500650, so safe_div is not used here. When safe_div is
    // used, all SOCP unit tests fail. Likely some GPU weirdness.
    f = 1.0 / (2.0 * gamma);

    /* overwrite sbar with wbar */
    sbar[0] = f * (sbar[0] + zbar[0]);
    for (QOCOInt j = 1; j < qi; ++j) {
      sbar[j] = f * (sbar[j] - zbar[j]);
    }

    /* overwrite zbar with v */
    f = safe_div((QOCOFloat)1.0,
                 qoco_sqrt((QOCOFloat)2.0 * (sbar[0] + (QOCOFloat)1.0)));

    zbar[0] = f * (sbar[0] + (QOCOFloat)1.0);
    for (QOCOInt j = 1; j < qi; ++j) {
      zbar[j] = f * sbar[j];
    }

    /* --- build W and Winv --- */
    QOCOInt shift = 0;
    QOCOFloat fwd = qoco_sqrt(safe_div(s_scal, z_scal));
    QOCOFloat finv = safe_div((QOCOFloat)1.0, fwd);

    for (QOCOInt j = 0; j < qi; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {

        QOCOInt full1 = nt_idx_full + j * qi + k;
        QOCOInt full2 = nt_idx_full + k * qi + j;

        QOCOFloat val = (QOCOFloat)2.0 * zbar[k] * zbar[j];

        QOCOFloat winv_val = val;
        if (j != 0 && k == 0)
          winv_val = -val;

        if (j == 0 && k == 0) {
          val -= (QOCOFloat)1.0;
          winv_val -= (QOCOFloat)1.0;
        }
        else if (j == k) {
          val += (QOCOFloat)1.0;
          winv_val += (QOCOFloat)1.0;
        }

        val *= fwd;
        winv_val *= finv;

        W[nt_idx + shift] = val;
        Winv[nt_idx + shift] = winv_val;
        Wfull[full1] = val;
        Wfull[full2] = val;
        Winvfull[full1] = winv_val;
        Winvfull[full2] = winv_val;

        shift++;
      }
    }

    /* --- compute WtW --- */
    shift = 0;
    for (QOCOInt j = 0; j < qi; ++j) {
      for (QOCOInt k = 0; k <= j; ++k) {
        WtW[nt_idx + shift] = qoco_dot_dev(&Wfull[nt_idx_full + j * qi],
                                           &Wfull[nt_idx_full + k * qi], qi);
        shift++;
      }
    }

    idx += qi;
    nt_idx += (qi * qi + qi) / 2;
    nt_idx_full += qi * qi;
  }
}

__global__ void nt_multiply_kernel(const QOCOFloat* W, const QOCOFloat* x,
                                   QOCOFloat* z, QOCOInt l, QOCOInt m,
                                   QOCOInt nsoc, const QOCOInt* q)
{
  QOCOInt i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m)
    return;

  /* ================= LP cone ================= */
  if (i < l) {
    z[i] = W[i] * x[i];
    return;
  }

  /* ================= SOC cones ================= */
  /* Find which SOC block index i belongs to */
  QOCOInt idx = l;    // start of SOC variables
  QOCOInt nt_idx = l; // start of SOC NT blocks

  for (QOCOInt soc = 0; soc < nsoc; ++soc) {
    QOCOInt qi = q[soc];

    if (i < idx + qi) {
      /* i is inside this SOC */
      QOCOInt j = i - idx;

      /* j-th row of qi x qi NT block times x[idx:idx+qi] */
      z[i] = qoco_dot_dev(&W[nt_idx + j * qi], &x[idx], qi);
      return;
    }

    idx += qi;
    nt_idx += qi * qi;
  }

  /* Safety (should not happen) */
  z[i] = 0.0;
}

__global__ void cone_product_kernel(const QOCOFloat* u, const QOCOFloat* v,
                                    QOCOFloat* p, QOCOInt l, QOCOInt nsoc,
                                    const QOCOInt* q)
{
  QOCOInt tid = blockIdx.x * blockDim.x + threadIdx.x;

  /* ================= LP cone ================= */
  if (tid < l) {
    p[tid] = u[tid] * v[tid];
    return;
  }

  /* ================= SOC cones ================= */
  QOCOInt soc_tid = tid - l;
  if (soc_tid >= nsoc)
    return;

  /* compute starting index of SOC cone soc_tid */
  QOCOInt idx = l;
  for (QOCOInt k = 0; k < soc_tid; ++k) {
    idx += q[k];
  }

  /* one thread computes one SOC cone product */
  soc_product(&u[idx], &v[idx], &p[idx], q[soc_tid]);
}

__global__ void cone_division_kernel(const QOCOFloat* lambda,
                                     const QOCOFloat* v, QOCOFloat* d,
                                     QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  QOCOInt tid = blockIdx.x * blockDim.x + threadIdx.x;

  /* ================= LP cone ================= */
  if (tid < l) {
    d[tid] = safe_div(v[tid], lambda[tid]);
    return;
  }

  /* ================= SOC cones ================= */
  QOCOInt soc_tid = tid - l;
  if (soc_tid >= nsoc)
    return;

  /* compute starting index of SOC cone soc_tid */
  QOCOInt idx = l;
  for (QOCOInt k = 0; k < soc_tid; ++k) {
    idx += q[k];
  }

  /* one thread handles one SOC cone */
  soc_division(&lambda[idx], &v[idx], &d[idx], q[soc_tid]);
}

__global__ void add_e_kernel(QOCOFloat* x, QOCOFloat a, QOCOInt l, QOCOInt nsoc,
                             const QOCOInt* q)
{
  QOCOInt tid = blockIdx.x * blockDim.x + threadIdx.x;

  /* ================= LP cone ================= */
  if (tid < l) {
    x[tid] -= a;
    return;
  }

  /* ================= SOC cones ================= */
  QOCOInt soc_tid = tid - l;
  if (soc_tid >= nsoc)
    return;

  /* compute starting index of SOC cone soc_tid */
  QOCOInt idx = l;
  for (QOCOInt k = 0; k < soc_tid; ++k) {
    idx += q[k];
  }

  /* subtract a from the cone "scalar" entry */
  x[idx] -= a;
}

void set_Wfull_identity(QOCOVectorf* Wfull, QOCOInt Wnnzfull,
                        QOCOProblemData* data)
{
  CUDA_CHECK(cudaGetLastError());
  QOCOFloat* W = get_data_vectorf(Wfull);

  const int threads = 256;
  const int blocks = (Wnnzfull + threads - 1) / threads;

  // kernel 1: zero + linear cone
  if (data->l > 0) {
    set_Wfull_linear<<<blocks, threads>>>(W, Wnnzfull, data->l);
  }
  CUDA_CHECK(cudaGetLastError());

  // kernel 2: SOC blocks
  const int blocks2 = data->nsoc;
  if (data->nsoc > 0) {
    set_Wfull_soc<<<blocks2, 256>>>(W, get_data_vectori(data->q), data->nsoc,
                                    data->l);
  }
  CUDA_CHECK(cudaGetLastError());
}

QOCOFloat cone_residual(const QOCOFloat* d_u, QOCOInt l, QOCOInt nsoc,
                        const QOCOInt* q)
{
  QOCOFloat* d_out = nullptr;
  QOCOFloat h_out = -1e7;
  // Allocate output
  if (l > 0 || nsoc > 0) {
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(QOCOFloat)));
    cone_residual_kernel<<<1, 1>>>(d_u, l, nsoc, q, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(
        cudaMemcpy(&h_out, d_out, sizeof(QOCOFloat), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
  }
  return h_out;
}

void cone_product(const QOCOFloat* u, const QOCOFloat* v, QOCOFloat* p,
                  QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  QOCOInt total_threads = l + nsoc;
  QOCOInt block = 256;
  QOCOInt grid = (total_threads + block - 1) / block;

  cone_product_kernel<<<grid, block>>>(u, v, p, l, nsoc, q);
}

void cone_division(const QOCOFloat* lambda, const QOCOFloat* v, QOCOFloat* d,
                   QOCOInt l, QOCOInt nsoc, const QOCOInt* q)
{
  QOCOInt total_threads = l + nsoc;
  QOCOInt block = 256;
  QOCOInt grid = (total_threads + block - 1) / block;

  cone_division_kernel<<<grid, block>>>(lambda, v, d, l, nsoc, q);
}

void bring2cone(QOCOFloat* u, QOCOProblemData* data)
{
  CUDA_CHECK(cudaGetLastError());
  QOCOFloat res =
      cone_residual(u, data->l, data->nsoc, get_data_vectori(data->q));
  if (res >= 0) {
    bring2cone_kernel<<<1, 1>>>(u, get_data_vectori(data->q), data->l,
                                data->nsoc);
    CUDA_CHECK(cudaGetLastError());
  }
}

void nt_multiply(QOCOFloat* W, QOCOFloat* x, QOCOFloat* z, QOCOInt l, QOCOInt m,
                 QOCOInt nsoc, QOCOInt* q)
{
  int threads = 256;
  int blocks = (m + threads - 1) / threads;

  nt_multiply_kernel<<<blocks, threads>>>(W, x, z, l, m, nsoc, q);
}

void compute_nt_scaling(QOCOWorkspace* work)
{
  QOCOFloat* W = get_data_vectorf(work->W);
  QOCOFloat* WtW = get_data_vectorf(work->WtW);
  QOCOFloat* Wfull = get_data_vectorf(work->Wfull);
  QOCOFloat* Winv = get_data_vectorf(work->Winv);
  QOCOFloat* Winvfull = get_data_vectorf(work->Winvfull);
  QOCOFloat* s = get_data_vectorf(work->s);
  QOCOFloat* z = get_data_vectorf(work->z);
  QOCOFloat* sbar = get_data_vectorf(work->sbar);
  QOCOFloat* zbar = get_data_vectorf(work->zbar);
  QOCOFloat* lambda = get_data_vectorf(work->lambda);
  QOCOInt* q = get_data_vectori(work->data->q);

  compute_nt_scaling_kernel<<<1, 1>>>(W, WtW, Wfull, Winv, Winvfull, s, z, sbar,
                                      zbar, work->data->l, work->data->nsoc, q);

  /* ================= lambda = W * z ================= */
  nt_multiply(Wfull, z, lambda, work->data->l, work->data->m, work->data->nsoc,
              q);
}

void compute_centering(QOCOSolver* solver)
{
  QOCOWorkspace* work = solver->work;
  QOCOFloat* xyz = get_data_vectorf(work->xyz);
  QOCOFloat* Ds = get_data_vectorf(work->Ds);
  QOCOFloat* ubuff1 = get_data_vectorf(work->ubuff1);
  QOCOFloat* ubuff2 = get_data_vectorf(work->ubuff2);
  QOCOFloat* Dzaff = &xyz[work->data->n + work->data->p];
  QOCOFloat a =
      qoco_min(linesearch(get_pointer_vectorf(work->z, 0), Dzaff, 1.0, solver),
               linesearch(get_pointer_vectorf(work->s, 0), Ds, 1.0, solver));

  // Compute rho. rho = ((s + a * Ds)'*(z + a * Dz)) / (s'*z).
  qoco_axpy(Dzaff, get_pointer_vectorf(work->z, 0), ubuff1, a, work->data->m);
  qoco_axpy(Ds, get_pointer_vectorf(work->s, 0), ubuff2, a, work->data->m);
  QOCOFloat rho = qoco_dot(ubuff1, ubuff2, work->data->m) /
                  qoco_dot(get_pointer_vectorf(work->z, 0),
                           get_pointer_vectorf(work->s, 0), work->data->m);

  // Compute sigma. sigma = max(0, min(1, rho))^3.
  QOCOFloat sigma = qoco_min(1.0, rho);
  sigma = qoco_max(0.0, sigma);
  sigma = sigma * sigma * sigma;
  solver->work->sigma = sigma;
}

/**
 * @brief Conducts linesearch by bisection to compute a \in (0, 1] such that
 * u + (a / f) * Du \in C
 * Warning: linesearch overwrites ubuff1. Do not pass in ubuff1 into u or Du.
 * Consider a dedicated buffer for linesearch.
 */
QOCOFloat bisection_search(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                           QOCOFloat* ubuff, QOCOInt m, QOCOInt l, QOCOInt nsoc,
                           QOCOInt* q, QOCOFloat* out_a, QOCOInt bisect_iters)
{
  // Single thread only
  QOCOFloat al = 0.0;
  QOCOFloat au = 1.0;
  QOCOFloat a = 0.0;

  for (QOCOInt i = 0; i < bisect_iters; ++i) {
    a = 0.5 * (al + au);

    qoco_axpy(Du, u, ubuff, safe_div(a, f), m);

    if (cone_residual(ubuff, l, nsoc, q) >= 0) {
      au = a;
    }
    else {
      al = a;
    }
  }
  return al;
}

__global__ void exact_linesearch_stage1(const QOCOFloat* u, const QOCOFloat* Du,
                                        QOCOInt l, QOCOFloat* block_mins)
{
  extern __shared__ QOCOFloat sdata[];

  QOCOInt tid = threadIdx.x;
  QOCOInt gid = blockIdx.x * blockDim.x + tid;

  /* Initialize with neutral element */
  QOCOFloat local_min = 0.0;

  if (gid < l && Du[gid] < 0.0) {
    local_min = Du[gid] / u[gid]; // negative
  }

  sdata[tid] = local_min;
  __syncthreads();

  /* Block-level min reduction */
  for (QOCOInt s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] < sdata[tid]) {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }

  /* Write block result */
  if (tid == 0) {
    block_mins[blockIdx.x] = sdata[0];
  }
}

__global__ void exact_linesearch_stage2(const QOCOFloat* block_mins,
                                        QOCOInt nblocks, QOCOFloat f,
                                        QOCOFloat* out_a)
{
  extern __shared__ QOCOFloat sdata[];

  QOCOInt tid = threadIdx.x;

  QOCOFloat local_min = 0.0;
  if (tid < nblocks) {
    local_min = block_mins[tid];
  }

  sdata[tid] = local_min;
  __syncthreads();

  for (QOCOInt s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] < sdata[tid]) {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    QOCOFloat minval = sdata[0];

    if (-f < minval) {
      *out_a = f;
    }
    else {
      *out_a = -f / minval;
    }
  }
}

/**
 * @brief Conducts exact linesearch to compute the largest a \in (0, 1] such
 * that u + (a / f) * Du \in C. Currently only works for LP cone.
 */
__global__ void exact_linesearch(const QOCOFloat* u, const QOCOFloat* Du,
                                 QOCOFloat f, QOCOInt l, QOCOFloat* out_a)
{
  // Only one thread executes
  QOCOFloat minval = 0.0;

  for (QOCOInt i = 0; i < l; ++i) {
    if (Du[i] < minval * u[i]) {
      minval = Du[i] / u[i];
    }
  }

  if (-f < minval) {
    *out_a = f;
  }
  else {
    *out_a = -f / minval;
  }
}

QOCOFloat linesearch(QOCOFloat* u, QOCOFloat* Du, QOCOFloat f,
                     QOCOSolver* solver)
{
  QOCOFloat a_host;
  QOCOFloat* d_linesearch_a;
  cudaMalloc(&d_linesearch_a, sizeof(QOCOFloat));

  if (solver->work->data->nsoc == 0) {
    int threads = 256;
    int blocks = (solver->work->data->l + threads - 1) / threads;
    QOCOFloat* block_mins;
    cudaMalloc(&block_mins, blocks * sizeof(QOCOFloat));

    exact_linesearch_stage1<<<blocks, threads, threads * sizeof(QOCOFloat)>>>(
        u, Du, solver->work->data->l, block_mins);

    int threads2 = 1;
    while (threads2 < blocks)
      threads2 <<= 1;

    exact_linesearch_stage2<<<1, threads2, threads2 * sizeof(QOCOFloat)>>>(
        block_mins, blocks, f, d_linesearch_a);

    cudaMemcpy(&a_host, d_linesearch_a, sizeof(QOCOFloat),
               cudaMemcpyDeviceToHost);
    cudaFree(d_linesearch_a);
    cudaFree(block_mins);
  }
  else {
    QOCOFloat* ubuff = get_data_vectorf(solver->work->ubuff1);
    QOCOInt m = solver->work->data->m;
    QOCOInt l = solver->work->data->l;
    QOCOInt nsoc = solver->work->data->nsoc;
    QOCOInt* q = get_data_vectori(solver->work->data->q);
    QOCOInt bisect_iters = solver->settings->bisect_iters;
    a_host = bisection_search(u, Du, f, ubuff, m, l, nsoc, q, d_linesearch_a,
                              bisect_iters);
  }
  return a_host;
}

void add_e(QOCOFloat* x, QOCOFloat a, QOCOFloat l, QOCOInt nsoc, QOCOVectori* q)
{
  QOCOInt total_threads = l + nsoc;
  QOCOInt block = 256;
  QOCOInt grid = (total_threads + block - 1) / block;

  add_e_kernel<<<grid, block>>>(x, a, l, nsoc, get_data_vectori(q));
}
