import utils.codegen_functions as cgen
from matplotlib import pyplot as plt
from scipy import sparse
import numpy as np
import cvxpy as cp


def generate_pdg():

    # Number of timesteps.
    N = 15

    # Discretization interval.
    dt = 1

    # Gravitational acceleration.
    g0 = 9.8

    # Initial condition.
    zi = np.array([10,10,100,0,0,0])

    # Terminal condition.
    zf = np.zeros(6)

    # State cost matrix.
    Q = 1.0 * sparse.eye(6)

    # Input cost matrix.
    R = 5.0 * sparse.eye(3)

    # Max inf norm on velocity.
    vmax = 10.0

    # Maximum thrust
    umax = 12.0

    n = 9 * N - 2

    # Parse cost function.
    Qfull = sparse.kron(sparse.eye(N), Q)
    Rfull = sparse.kron(sparse.eye(N - 1), R)
    P = sparse.block_diag((Qfull, Rfull, 0.0*sparse.eye(1)))
    c = np.zeros(n)

    # Double integrator dynamics.
    Ad = np.block([[np.eye(3), dt*np.eye(3)],[np.zeros((3,3)), np.eye(3)]])
    Bd = np.block([[0.5*dt**2*np.eye(3)],[dt*np.eye(3)]])
    g = np.array([0, 0, -0.5*g0*dt**2, 0, 0, -g0*dt])

    # Parse dynamics constraint.
    Azdyn = np.block([np.kron(np.eye(N-1), Ad), np.zeros((6*(N - 1), 6))]) - np.block([np.zeros((6*(N-1), 6)), np.eye(6*(N - 1))])
    Audyn = np.kron(np.eye(N-1), Bd)
    Axidyn = np.zeros((6*(N - 1), 1))

    # Parse boundary conditions.
    Azbc = np.block([[np.eye(6), np.zeros((6, 6*(N-1)))],[np.zeros((6, 6*(N-1))), np.eye(6)]])
    Aubc = np.zeros((12, 3 * (N - 1)))
    Axibc = np.zeros((12, 1))

    # Parse slack variable.
    Azslack = np.zeros((1, 6*N))
    Auslack = np.zeros((1, 3*(N - 1)))
    Axislack = np.array([1.0])

    # Combine dynamics and boundary conditions into equality constraint matrix A, and vector b.
    A = np.block([[Azdyn, Audyn, Axidyn],[Azbc, Aubc, Axibc],[Azslack, Auslack, Axislack]])
    b = np.hstack((np.kron(np.ones(N - 1), -g), zi, zf, umax))
    p = 6 * N + 7

    # Parse velocity constraint.
    Gzvelocity = np.block([[np.kron(np.eye(N), np.block([np.zeros((3,3)), np.eye(3)]))], \
                           [np.kron(np.eye(N), np.block([np.zeros((3,3)), -np.eye(3)]))]])
    Guvelocity = np.zeros((6 * N, 3*(N-1)))
    Gxivelocity = np.zeros((6*N, 1))

    # Parse thrust constraint.
    Gzthrust = np.zeros((4*(N-1), 6*N))
    Guthrust = np.kron(np.eye(N - 1), np.block([[np.zeros((1,3))],[-np.eye(3)]]))
    Gxithrust = np.kron(np.ones(N-1), np.array([-1,0,0,0]))
    Gxithrust = np.asmatrix(Gxithrust).T
    G = np.block([[Gzvelocity, Guvelocity, Gxivelocity], [Gzthrust, Guthrust, Gxithrust]])
    h = np.hstack((vmax * np.ones((6 * N)), np.zeros(4*(N-1))))

    l = 6 * N
    m, _ = G.shape

    q = 4 * np.ones(N - 1)
    nsoc = N - 1

    # Convert to sparse data type.
    P = sparse.triu(P, format='csc')
    A = sparse.csc_matrix(A)
    G = sparse.csc_matrix(G)

    # Solve with cvxpy.
    zvar = cp.Variable((6, N))
    uvar = cp.Variable((3, N - 1))
    obj = 0
    con = [zvar[:, 0] == zi, zvar[:,N - 1] == zf]
    for i in range(N - 1):
        obj += (1/2)*(cp.quad_form(zvar[:,i], Q) + cp.quad_form(uvar[:,i], R))
        con += [zvar[:, i + 1] == Ad @ zvar[:, i] + Bd @ uvar[:, i] + g]
        con += [cp.norm_inf(zvar[4:6,i]) <= vmax]
        con += [cp.norm(uvar[:,i]) <= umax]
    obj += (1/2)*(cp.quad_form(zvar[:,N - 1], Q))
    prob = cp.Problem(cp.Minimize(obj), con)
    prob.solve(verbose=True)

    # data, chain, inverse_data = prob.get_problem_data(cp.SCS)
    # unorm = np.zeros(N - 1)
    # for i in range(N - 1):
    #     unorm[i] = np.linalg.norm(uvar.value[:, i])
    # fig = plt.figure()
    # # syntax for 3-D projection
    # ax = plt.axes(projection ='3d')
    # # plotting
    # ax.plot3D(zvar.value[0,:], zvar.value[1,:], zvar.value[2,:], 'green')
    # plt.figure()
    # plt.plot(unorm)
    # plt.title("Thrust magnitude")

    # plt.figure()
    # plt.plot(zvar.value[3,:])
    # plt.plot(zvar.value[4,:])
    # plt.plot(zvar.value[5,:])

    # plt.show()

    # Generate data file for unit test.
    cgen.generate_data(n, m, p, P, c, A, b, G, h, l, nsoc, q,
                       prob.value, "ocp", "pdg")
    cgen.generate_test("ocp", "pdg")


generate_pdg()
