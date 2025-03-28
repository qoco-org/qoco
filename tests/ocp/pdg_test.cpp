#include "test_utils.h"
#include "gtest/gtest.h"
#include "pdg_data.h"

TEST(ocp_test, pdg)
{
    // Allocate and set sparse matrix data.
    QOCOCscMatrix* P;
    QOCOCscMatrix* A;
    QOCOCscMatrix* G;
    if(pdg_P_nnz > 0) {
        P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(P, pdg_n, pdg_n, pdg_P_nnz, pdg_P_x, pdg_P_p, pdg_P_i);
    }
    else {
        P = nullptr;
    }
    if(pdg_A_nnz > 0) {
        A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(A, pdg_p, pdg_n, pdg_A_nnz, pdg_A_x, pdg_A_p, pdg_A_i);
    }
    else {
        A = nullptr;
    }
    if(pdg_G_nnz > 0) {
        G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
        qoco_set_csc(G, pdg_m, pdg_n, pdg_G_nnz, pdg_G_x, pdg_G_p, pdg_G_i);
    }
    else {
        G = nullptr;
    }
    QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
    set_default_settings(settings);
    settings->verbose = 1;
    QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));

    QOCOInt exit = qoco_setup(solver, pdg_n, pdg_m, pdg_p, P, pdg_c, A, pdg_b, G, pdg_h, pdg_l, pdg_nsoc, pdg_q, settings);
    ASSERT_EQ(exit, QOCO_NO_ERROR);

    exit = qoco_solve(solver);
    ASSERT_EQ(exit, QOCO_SOLVED);

    // Expect relative error of objective to be less than tolerance.
    expect_rel_error(solver->sol->obj, pdg_objopt, 0.0001);

    // Cleanup memory allocations. 
    qoco_cleanup(solver);
    free(settings);
    free(P);
    free(A);
    free(G);
}
