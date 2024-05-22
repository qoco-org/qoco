// This file was autogenerated by the QCOS test suite on 05/21/2024 18:41:24

#include "test_utils.h"
#include "gtest/gtest.h"
#include "double_integrator_data.h"

TEST(ocp_test, double_integrator)
{
    // Allocate and set sparse matrix data.
    QCOSCscMatrix* P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
    QCOSCscMatrix* A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
    QCOSCscMatrix* G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));

    qcos_set_csc(P, double_integrator_n, double_integrator_n, double_integrator_P_nnz, double_integrator_P_x, double_integrator_P_p, double_integrator_P_i);
    qcos_set_csc(A, double_integrator_p, double_integrator_n, double_integrator_A_nnz, double_integrator_A_x, double_integrator_A_p, double_integrator_A_i);
    qcos_set_csc(G, double_integrator_m, double_integrator_n, double_integrator_G_nnz, double_integrator_G_x, double_integrator_G_p, double_integrator_G_i);

    QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

    QCOSInt exit = qcos_setup(solver, double_integrator_n, double_integrator_m, double_integrator_p, P, double_integrator_c, A, double_integrator_b, G, double_integrator_h, double_integrator_l, double_integrator_ncones, double_integrator_q, settings);
    ASSERT_EQ(exit, QCOS_NO_ERROR);

    exit = qcos_solve(solver);
    ASSERT_EQ(exit, QCOS_SOLVED);

    // Cleanup memory allocations. 
    qcos_cleanup(solver);
    free(P);
    free(A);
    free(G);
}
