import utils.codegen_functions as cgen
from scipy import sparse
import numpy as np
import cvxpy as cp


def generate_markowitz():

    # Seed rng.
    rng = np.random.default_rng(12345)

    # Number of factors.
    f = 50

    # Number of assets.
    n = 100

    # Factor loading matrix.
    F = sparse.random(f, n, density=0.25, random_state=rng)
    Pfull = F.T @ F + 0.1 * sparse.eye(n)
    P = sparse.triu(Pfull, format='csc')
    c = -rng.random(n)

    A = np.ones((1, n))
    A = sparse.csc_matrix(A)
    b = np.array([1.])

    G = -sparse.eye(n)
    G = sparse.csc_matrix(G)
    h = np.zeros(n)

    l = n
    nsoc = 0
    q = None

    # Solve with cvxpy.
    xvar = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(xvar, Pfull) + c.T @ xvar),
                      [G @ xvar <= h,
                       A @ xvar == b])
    prob.solve(verbose=False)

    # Generate data file for unit test.
    cgen.generate_data(n, n, 1, P, c, A, b, G, h, l, nsoc, q,
                       prob.value, "portfolio", "markowitz")
    cgen.generate_test("portfolio", "markowitz")


generate_markowitz()
