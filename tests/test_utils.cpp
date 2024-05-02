#include "test_utils.h"

void expect_eq_vectorf(QCOSFloat* x, QCOSFloat* y, QCOSInt n, QCOSFloat tol)
{
  for (QCOSInt i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], y[i], tol);
  }
}
