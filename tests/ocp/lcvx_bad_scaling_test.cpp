#include "test_utils.h"
#include "gtest/gtest.h"
#include "lcvx_bad_scaling_data.h"

TEST(ocp_test, lcvx_bad_scaling)
{
    // Allocate and set sparse matrix data.
    QCOSCscMatrix* P;
    QCOSCscMatrix* A;
    QCOSCscMatrix* G;
    if(lcvx_bad_scaling_P_nnz > 0) {
        P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(P, lcvx_bad_scaling_n, lcvx_bad_scaling_n, lcvx_bad_scaling_P_nnz, lcvx_bad_scaling_P_x, lcvx_bad_scaling_P_p, lcvx_bad_scaling_P_i);
    }
    else {
        P = nullptr;
    }
    if(lcvx_bad_scaling_A_nnz > 0) {
        A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(A, lcvx_bad_scaling_p, lcvx_bad_scaling_n, lcvx_bad_scaling_A_nnz, lcvx_bad_scaling_A_x, lcvx_bad_scaling_A_p, lcvx_bad_scaling_A_i);
    }
    else {
        A = nullptr;
    }
    if(lcvx_bad_scaling_G_nnz > 0) {
        G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(G, lcvx_bad_scaling_m, lcvx_bad_scaling_n, lcvx_bad_scaling_G_nnz, lcvx_bad_scaling_G_x, lcvx_bad_scaling_G_p, lcvx_bad_scaling_G_i);
    }
    else {
        G = nullptr;
    }
    QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

    QCOSInt exit = qcos_setup(solver, lcvx_bad_scaling_n, lcvx_bad_scaling_m, lcvx_bad_scaling_p, P, lcvx_bad_scaling_c, A, lcvx_bad_scaling_b, G, lcvx_bad_scaling_h, lcvx_bad_scaling_l, lcvx_bad_scaling_nsoc, lcvx_bad_scaling_q, settings);
    ASSERT_EQ(exit, QCOS_NO_ERROR);

    exit = qcos_solve(solver);
    ASSERT_EQ(exit, QCOS_SOLVED);

    // Expect relative error of objective to be less than tolerance.
    expect_rel_error(solver->sol->obj, lcvx_bad_scaling_objopt, 0.0005);

    // Cleanup memory allocations. 
    qcos_cleanup(solver);
    free(P);
    free(A);
    free(G);
}
