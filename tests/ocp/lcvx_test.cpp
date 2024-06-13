#include "test_utils.h"
#include "gtest/gtest.h"
#include "lcvx_data.h"

TEST(ocp_test, lcvx)
{
    // Allocate and set sparse matrix data.
    QCOSCscMatrix* P;
    QCOSCscMatrix* A;
    QCOSCscMatrix* G;
    if(lcvx_P_nnz > 0) {
        P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(P, lcvx_n, lcvx_n, lcvx_P_nnz, lcvx_P_x, lcvx_P_p, lcvx_P_i);
    }
    else {
        P = nullptr;
    }
    if(lcvx_A_nnz > 0) {
        A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(A, lcvx_p, lcvx_n, lcvx_A_nnz, lcvx_A_x, lcvx_A_p, lcvx_A_i);
    }
    else {
        A = nullptr;
    }
    if(lcvx_G_nnz > 0) {
        G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(G, lcvx_m, lcvx_n, lcvx_G_nnz, lcvx_G_x, lcvx_G_p, lcvx_G_i);
    }
    else {
        G = nullptr;
    }
    QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

    QCOSInt exit = qcos_setup(solver, lcvx_n, lcvx_m, lcvx_p, P, lcvx_c, A, lcvx_b, G, lcvx_h, lcvx_l, lcvx_nsoc, lcvx_q, settings);
    ASSERT_EQ(exit, QCOS_NO_ERROR);

    exit = qcos_solve(solver);
    ASSERT_EQ(exit, QCOS_SOLVED);

    // Expect relative error of objective to be less than tolerance.
    expect_rel_error(solver->sol->obj, lcvx_objopt, 0.0001);

    // Cleanup memory allocations. 
    qcos_cleanup(solver);
    free(P);
    free(A);
    free(G);
}