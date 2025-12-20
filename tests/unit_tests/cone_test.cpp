#include "test_utils.h"
#include "gtest/gtest.h"

#include "qoco.h"

// TEST(cone, soc_product_test)
// {
//   constexpr QOCOInt n = 5;
//   QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
//   QOCOFloat y[] = {6.0, 7.0, 8.0, 9.0, 10.0};
//   QOCOFloat p[n];
//   QOCOFloat pexpected[] = {130.0, 19.0, 26.0, 33.0, 40.0};
//   QOCOFloat tol = 1e-12;

//   soc_product(x, y, p, n);
//   expect_eq_vectorf(p, pexpected, n, tol);
// }

// TEST(cone, soc_division_test)
// {
//   constexpr QOCOInt n = 5;
//   QOCOFloat x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
//   QOCOFloat y[] = {6.0, 7.0, 8.0, 9.0, 10.0};
//   QOCOFloat d[n];
//   QOCOFloat dexpected[] = {2.2264150943396226, 2.5471698113207544,
//                            1.3207547169811316, 0.09433962264150907,
//                            -1.1320754716981127};
//   QOCOFloat tol = 1e-12;

//   soc_division(x, y, d, n);
//   expect_eq_vectorf(d, dexpected, n, tol);
// }

// TEST(cone, soc_residual_test)
// {
//   constexpr QOCOInt n = 4;
//   QOCOFloat x[] = {2.2, 4.5, 3.2, 2.5};
//   QOCOFloat res_expected = 3.861352984276695;
//   QOCOFloat tol = 1e-12;

//   QOCOFloat res = soc_residual(x, n);
//   EXPECT_NEAR(res, res_expected, tol);
// }

// TEST(cone, soc_residual2_test)
// {
//   constexpr QOCOInt n = 4;
//   QOCOFloat x[] = {2.2, 4.5, 3.2, 2.5};
//   QOCOFloat res_expected = -31.9;
//   QOCOFloat tol = 1e-12;

//   QOCOFloat res = soc_residual2(x, n);
//   EXPECT_NEAR(res, res_expected, tol);
// }

TEST(cone, cone_product_test)
{
  constexpr QOCOInt n = 5;
  constexpr QOCOInt l = 3;
  constexpr QOCOInt nsoc = 2;
  QOCOInt q[] = {5, 5};

  QOCOFloat x[] = {6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0,
                   5.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  QOCOFloat y[] = {9.0,  10.0, 11.0, 6.0, 7.0, 8.0, 9.0,
                   10.0, 6.0,  7.0,  8.0, 9.0, 10.0};
  QOCOFloat p[13];
  QOCOFloat pexpected[] = {54.0, 70.0,  88.0, 130.0, 19.0, 26.0, 33.0,
                           40.0, 130.0, 19.0, 26.0,  33.0, 40.0};
  QOCOFloat tol = 1e-12;

  cone_product(x, y, p, l, nsoc, q);
  expect_eq_vectorf(p, pexpected, n, tol);
}

TEST(cone, cone_division_test)
{
  constexpr QOCOInt n = 5;
  constexpr QOCOInt l = 3;
  constexpr QOCOInt nsoc = 2;
  QOCOInt q[] = {5, 5};
  QOCOFloat x[] = {6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0,
                   5.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  QOCOFloat y[] = {9.0,  10.0, 11.0, 6.0, 7.0, 8.0, 9.0,
                   10.0, 6.0,  7.0,  8.0, 9.0, 10.0};
  QOCOFloat d[13];
  QOCOFloat dexpected[] = {1.5,
                           10.0 / 7.0,
                           1.375,
                           2.2264150943396226,
                           2.5471698113207544,
                           1.3207547169811316,
                           0.09433962264150907,
                           -1.1320754716981127,
                           2.2264150943396226,
                           2.5471698113207544,
                           1.3207547169811316,
                           0.09433962264150907,
                           -1.1320754716981127};
  QOCOFloat tol = 1e-12;

  cone_division(x, y, d, l, nsoc, q);
  expect_eq_vectorf(d, dexpected, n, tol);
}

// TEST(cone, cone_residual_test)
// {
//   constexpr QOCOInt n = 5;
//   constexpr QOCOInt l = 3;
//   constexpr QOCOInt nsoc = 2;
//   QOCOInt q[] = {5, 4};
//   QOCOFloat x[] =
//   {6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0};

//   QOCOFloat res_expected = ;

//   QOCOFloat res = cone_residual(x, l, nsoc, q);
//   EXPECT_NEAR(res, res_expected, tol);
//   EXPECT_EQ(1, 1);
// }