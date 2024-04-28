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

void soc_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p, QCOSInt n)
{
  p[0] = dot(u, v, n);
  for (QCOSInt i = 1; i < n; ++i) {
    p[i] = u[0] * v[i] + v[0] * u[i];
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

void compute_mu(QCOSWorkspace* work)
{
  work->mu = dot(work->s, work->z, work->data->m) / work->data->m;
}

void compute_nt_scaling(QCOSWorkspace* work)
{
  // Compute Nesterov-Todd scaling and scaled variables for LP cone.
  QCOSInt idx;
  for (idx = 0; idx < work->data->l; ++idx) {
    work->WtW[idx] = safe_div(work->s[idx], work->z[idx]);
    work->W[idx] = qcos_sqrt(work->WtW[idx]);
    work->Wfull[idx] = work->W[idx];
    work->lambda[idx] = qcos_sqrt(work->s[idx] * work->z[idx]);
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
    for (QCOSInt j = 0; j < work->data->q[i]; ++j) {
      for (QCOSInt k = 0; k <= j; ++k) {
        QCOSInt full_idx1 = nt_idx_full + j * work->data->q[i] + k;
        QCOSInt full_idx2 = nt_idx_full + k * work->data->q[i] + j;
        work->W[nt_idx + shift] = 2 * (work->zbar[k] * work->zbar[j]);
        if (j == k && j == 0) {
          work->W[nt_idx + shift] -= 1;
        }
        else if (j == k) {
          work->W[nt_idx + shift] += 1;
        }
        work->W[nt_idx + shift] *= f;
        work->Wfull[full_idx1] = work->W[nt_idx + shift];
        work->Wfull[full_idx2] = work->W[nt_idx + shift];
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

    // Compute lambda for second-order cones. lambda(soc_idx) = Wsoc * z_soc
    for (QCOSInt j = 0; j < work->data->q[i]; ++j) {
      work->lambda[nt_idx + j] =
          dot(&work->Wfull[nt_idx + j * work->data->q[i]], &work->z[idx],
              work->data->q[i]);
    }

    idx += work->data->q[i];
    nt_idx += (work->data->q[i] * work->data->q[i] + work->data->q[i]) / 2;
    nt_idx_full += work->data->q[i] * work->data->q[i];
  }
}