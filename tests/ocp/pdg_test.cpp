#include "test_utils.h"
#include "gtest/gtest.h"
#include "pdg_data.h"

TEST(ocp_test, pdg)
{
    // Allocate and set sparse matrix data.
    QCOSCscMatrix* P;
    QCOSCscMatrix* A;
    QCOSCscMatrix* G;
    if(pdg_P_nnz > 0) {
        P = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(P, pdg_n, pdg_n, pdg_P_nnz, pdg_P_x, pdg_P_p, pdg_P_i);
    }
    else {
        P = nullptr;
    }
    if(pdg_A_nnz > 0) {
        A = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(A, pdg_p, pdg_n, pdg_A_nnz, pdg_A_x, pdg_A_p, pdg_A_i);
    }
    else {
        A = nullptr;
    }
    if(pdg_G_nnz > 0) {
        G = (QCOSCscMatrix*)malloc(sizeof(QCOSCscMatrix));
        qcos_set_csc(G, pdg_m, pdg_n, pdg_G_nnz, pdg_G_x, pdg_G_p, pdg_G_i);
    }
    else {
        G = nullptr;
    }
    QCOSSettings* settings = (QCOSSettings*)malloc(sizeof(QCOSSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QCOSSolver* solver = (QCOSSolver*)malloc(sizeof(QCOSSolver));

    QCOSInt exit = qcos_setup(solver, pdg_n, pdg_m, pdg_p, P, pdg_c, A, pdg_b, G, pdg_h, pdg_l, pdg_nsoc, pdg_q, settings);
    ASSERT_EQ(exit, QCOS_NO_ERROR);

    exit = qcos_solve(solver);
    ASSERT_EQ(exit, QCOS_SOLVED);

    // Expect relative error of objective to be less than tolerance.
    expect_rel_error(solver->sol->obj, pdg_objopt, 0.0001);

    // Cleanup memory allocations. 
    qcos_cleanup(solver);
    free(P);
    free(A);
    free(G);
}
