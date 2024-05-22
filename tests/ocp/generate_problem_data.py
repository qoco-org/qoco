import utils.codegen_functions as cgen
from scipy import sparse
import numpy as np

def generate_double_integrator():
    P = sparse.diags([1., 2., 3., 4., 5., 6.])
    P = sparse.triu(P, format='csc')
    c = np.array([1., 2., 3., 4., 5., 6.])

    A = np.array([[1., 1., 0., 0., 0., 0.], [0., 1., 2., 0., 0., 0.]])
    A = sparse.csc_matrix(A)
    b = np.array([1., 2.])

    G = -sparse.eye(6)
    G = sparse.csc_matrix(G)
    h = np.zeros((6, 1))

    l = 1
    ncones = 2
    q = np.array([2, 3])

    cgen.generate_data(P, c, A, b, G, h, l, ncones, q, "ocp", "double_integrator")
    cgen.generate_test("ocp", "double_integrator")

generate_double_integrator()