#include "validation.h"

QCOSInt qcos_validate_settings(QCOSSettings* settings)
{
  if (settings->tol < 0)
    return 1;
  return 0;
}

QCOSInt qcos_validate_data(const QCOSCscMatrix* P, const QCOSFloat* c,
                           const QCOSCscMatrix* A, const QCOSFloat* b,
                           const QCOSCscMatrix* G, const QCOSFloat* h,
                           const QCOSInt l, const QCOSInt ncones,
                           const QCOSInt* q)
{
  if (!P)
    return 1;
  if (!c)
    return 1;
  if (!A)
    return 1;
  if (!b)
    return 1;
  if (!G)
    return 1;
  if (!h)
    return 1;
  if (l < 0)
    return 1;
  if (ncones < 0)
    return 1;
  if (!q)
    return 1;

  return 0;
}
