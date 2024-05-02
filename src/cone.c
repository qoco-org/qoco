/**
 * @file cone.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 2-Clause License
 */

#include "cone.h"
#include "utils.h"

void cone_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p,
                  QCOSProblemData* data)
{
  QCOSInt idx;
  // Compute LP cone product.
  for (idx = 0; idx < data->l; ++idx) {
    p[idx] = u[idx] * v[idx];
  }

  // Compute second-order cone product.
  for (QCOSInt i = 0; i < data->ncones; ++i) {
    soc_product(&u[idx], &v[idx], &p[idx], data->q[i]);
    idx += data->q[i];
  }
}

void cone_division(QCOSFloat* lambda, QCOSFloat* v, QCOSFloat* d,
                   QCOSProblemData* data)
{
  QCOSInt idx;

  // Compute LP cone division.
  for (idx = 0; idx < data->l; ++idx) {
    d[idx] = safe_div(v[idx], lambda[idx]);
  }

  // Compute second-order cone division.
  for (QCOSInt i = 0; i < data->ncones; ++i) {
    soc_division(&lambda[idx], &v[idx], &d[idx], data->q[i]);
    idx += data->q[i];
  }
}

void soc_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p, QCOSInt n)
{
  p[0] = dot(u, v, n);
  for (QCOSInt i = 1; i < n; ++i) {
    p[i] = u[0] * v[i] + v[0] * u[i];
  }
}

void soc_division(QCOSFloat* lam, QCOSFloat* v, QCOSFloat* d, QCOSInt n)
{
  QCOSFloat f = lam[0] * lam[0] - dot(&lam[1], &lam[1], n - 1);
  QCOSFloat finv = safe_div(1.0, f);
  QCOSFloat lam0inv = safe_div(1.0, lam[0]);
  QCOSFloat lam1dv1 = dot(&lam[1], &v[1], n - 1);

  d[0] = finv * (lam[0] * v[0] - dot(&lam[1], &v[1], n - 1));
  for (QCOSInt i = 1; i < n; ++i) {
    d[i] = finv *
           (-lam[i] * v[0] + lam0inv * f * v[i] + lam0inv * lam1dv1 * lam[i]);
  }
}

QCOSFloat soc_residual(QCOSFloat* u, QCOSInt n)
{
  QCOSFloat res = 0;
  for (QCOSInt i = 1; i < n; ++i) {
    res += u[i] * u[i];
  }
  res = qcos_sqrt(res) - u[0];
  return res;
}

QCOSFloat soc_residual2(QCOSFloat* u, QCOSInt n)
{
  QCOSFloat res = u[0] * u[0];
  for (QCOSInt i = 1; i < n; ++i) {
    res -= u[i] * u[i];
  }
  return res;
}

QCOSFloat cone_residual(QCOSFloat* u, QCOSProblemData* data)
{
  QCOSFloat res = -1e7;

  // Compute LP cone residuals.
  QCOSInt idx;
  for (idx = 0; idx < data->l; ++idx) {
    res = qcos_max(-u[idx], res);
    // If res is positive can just return here.
  }

  // Compute second-order cone residual.
  for (QCOSInt i = 0; i < data->ncones; ++i) {
    res = qcos_max(res, soc_residual(&u[idx], data->q[i]));
    idx += data->q[i];
  }

  return res;
}

void bring2cone(QCOSFloat* u, QCOSProblemData* data)
{
  if (cone_residual(u, data) >= 0) {
    QCOSFloat a = 0.0;

    // Get a for for LP cone.
    QCOSInt idx;
    for (idx = 0; idx < data->l; ++idx) {
      a = qcos_max(a, -u[idx]);
    }
    a = qcos_max(a, 0.0);

    // Get a for second-order cone.
    for (QCOSInt i = 0; i < data->ncones; ++i) {
      QCOSFloat soc_res = soc_residual(&u[idx], data->q[i]);
      if (soc_res > 0 && soc_res > a) {
        a = soc_res;
      }
      idx += data->q[i];
    }

    // Compute u + (1 + a) * e for LP cone.
    for (idx = 0; idx < data->l; ++idx) {
      u[idx] += (1 + a);
    }

    // Compute u + (1 + a) * e for second-order cones.
    for (QCOSInt i = 0; i < data->ncones; ++i) {
      u[idx] += (1 + a);
      idx += data->q[i];
    }
  }
}

void nt_multiply(QCOSFloat* W, QCOSFloat* x, QCOSFloat* z, QCOSInt l, QCOSInt m,
                 QCOSInt ncones, QCOSInt* q)
{
  // Compute product for LP cone part of W.
  for (QCOSInt i = 0; i < l; ++i) {
    z[i] = (W[i] * x[i]);
  }

  // Compute product for second-order cones.
  QCOSInt nt_idx = l;
  QCOSInt idx = l;

  // Zero out second-order cone block of result z.
  for (QCOSInt i = l; i < m; ++i) {
    z[i] = 0;
  }

  // Loop over all second-order cones.
  for (QCOSInt i = 0; i < ncones; ++i) {
    // Loop over elements within a second-order cone.
    for (QCOSInt j = 0; j < q[i]; ++j) {
      z[idx + j] += dot(&W[nt_idx + j * q[i]], &x[idx], q[i]);
    }
    idx += q[i];
    nt_idx += q[i] * q[i];
  }
}

void compute_mu(QCOSWorkspace* work)
{
  work->mu = safe_div(dot(work->s, work->z, work->data->m), work->data->m);
}

void compute_nt_scaling(QCOSWorkspace* work)
{
  // Compute Nesterov-Todd scaling for LP cone.
  QCOSInt idx;
  for (idx = 0; idx < work->data->l; ++idx) {
    work->WtW[idx] = safe_div(work->s[idx], work->z[idx]);
    work->W[idx] = qcos_sqrt(work->WtW[idx]);
    work->Wfull[idx] = work->W[idx];
    work->Winv[idx] = safe_div(1.0, work->W[idx]);
    work->Winvfull[idx] = work->Winv[idx];
  }

  // Compute Nesterov-Todd scaling for second-order cones.
  QCOSInt nt_idx = idx;
  QCOSInt nt_idx_full = idx;
  for (QCOSInt i = 0; i < work->data->ncones; ++i) {
    // Compute normalized vectors.
    QCOSFloat s_scal = soc_residual2(&work->s[idx], work->data->q[i]);
    s_scal = qcos_sqrt(s_scal);
    QCOSFloat f = safe_div(1.0, s_scal);
    scale_arrayf(&work->s[idx], work->sbar, f, work->data->q[i]);

    QCOSFloat z_scal = soc_residual2(&work->z[idx], work->data->q[i]);
    z_scal = qcos_sqrt(z_scal);
    f = safe_div(1.0, z_scal);
    scale_arrayf(&work->z[idx], work->zbar, f, work->data->q[i]);

    QCOSFloat gamma =
        qcos_sqrt(0.5 * (1 + dot(work->sbar, work->zbar, work->data->q[i])));

    f = safe_div(1.0, (2 * gamma));

    // Overwrite sbar with wbar.
    work->sbar[0] = f * (work->sbar[0] + work->zbar[0]);
    for (QCOSInt j = 1; j < work->data->q[i]; ++j) {
      work->sbar[j] = f * (work->sbar[j] - work->zbar[j]);
    }

    // Overwrite zbar with v.
    f = safe_div(1.0, qcos_sqrt(2 * (work->sbar[0] + 1)));
    work->zbar[0] = f * (work->sbar[0] + 1.0);
    for (QCOSInt j = 1; j < work->data->q[i]; ++j) {
      work->zbar[j] = f * work->sbar[j];
    }

    // Compute W for second-order cones.
    QCOSInt shift = 0;
    f = qcos_sqrt(safe_div(s_scal, z_scal));
    QCOSFloat finv = safe_div(1.0, f);
    for (QCOSInt j = 0; j < work->data->q[i]; ++j) {
      for (QCOSInt k = 0; k <= j; ++k) {
        QCOSInt full_idx1 = nt_idx_full + j * work->data->q[i] + k;
        QCOSInt full_idx2 = nt_idx_full + k * work->data->q[i] + j;
        work->W[nt_idx + shift] = 2 * (work->zbar[k] * work->zbar[j]);
        if (j != 0 && k == 0) {
          work->Winv[nt_idx + shift] = -work->W[nt_idx + shift];
        }
        else {
          work->Winv[nt_idx + shift] = work->W[nt_idx + shift];
        }
        if (j == k && j == 0) {
          work->W[nt_idx + shift] -= 1;
          work->Winv[nt_idx + shift] -= 1;
        }
        else if (j == k) {
          work->W[nt_idx + shift] += 1;
          work->Winv[nt_idx + shift] += 1;
        }
        work->W[nt_idx + shift] *= f;
        work->Winv[nt_idx + shift] *= finv;
        work->Wfull[full_idx1] = work->W[nt_idx + shift];
        work->Wfull[full_idx2] = work->W[nt_idx + shift];
        work->Winvfull[full_idx1] = work->Winv[nt_idx + shift];
        work->Winvfull[full_idx2] = work->Winv[nt_idx + shift];
        shift += 1;
      }
    }

    // Compute WtW for second-order cones.
    shift = 0;
    for (QCOSInt j = 0; j < work->data->q[i]; ++j) {
      for (QCOSInt k = 0; k <= j; ++k) {
        work->WtW[nt_idx + shift] =
            dot(&work->Wfull[nt_idx + j * work->data->q[i]],
                &work->Wfull[nt_idx + k * work->data->q[i]], work->data->q[i]);
        shift += 1;
      }
    }

    idx += work->data->q[i];
    nt_idx += (work->data->q[i] * work->data->q[i] + work->data->q[i]) / 2;
    nt_idx_full += work->data->q[i] * work->data->q[i];
  }

  // Compute scaled variable lambda. lambda = W * z.
  nt_multiply(work->Wfull, work->z, work->lambda, work->data->l, work->data->m,
              work->data->ncones, work->data->q);
}

void compute_centering(QCOSSolver* solver)
{
  QCOSWorkspace* work = solver->work;
  QCOSFloat* Dzaff = &work->kkt->xyz[work->data->n + work->data->p];
  QCOSFloat a = qcos_min(linesearch(work->z, Dzaff, 1.0, solver),
                         linesearch(work->s, work->Ds, 1.0, solver));

  // Compute rho. rho = ((s + a * Ds)'*(z + a * Dz)) / (s'*z).
  axpy(Dzaff, work->z, work->ubuff1, a, work->data->m);
  axpy(work->Ds, work->s, work->ubuff2, a, work->data->m);
  QCOSFloat rho = dot(work->ubuff1, work->ubuff2, work->data->m) /
                  dot(work->z, work->s, work->data->m);

  // Compute sigma. sigma = max(0, min(1, rho))^3.
  QCOSFloat sigma = qcos_min(1.0, rho);
  sigma = qcos_max(0.0, sigma);
  sigma = sigma * sigma * sigma;
  solver->work->sigma = sigma;
}

QCOSFloat linesearch(QCOSFloat* u, QCOSFloat* Du, QCOSFloat f,
                     QCOSSolver* solver)
{
  QCOSWorkspace* work = solver->work;

  QCOSFloat al = 0.0;
  QCOSFloat au = 1.0;
  QCOSFloat a = 0.0;
  for (QCOSInt i = 0; i < solver->settings->max_iter_bisection; ++i) {
    a = 0.5 * (al + au);
    axpy(Du, u, work->ubuff1, safe_div(a, f), work->data->m);
    if (cone_residual(work->ubuff1, work->data) >= 0) {
      au = a;
    }
    else {
      al = a;
    }
  }
  return al;
}