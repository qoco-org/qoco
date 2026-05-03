import numpy as np
from scipy import sparse
import os
import cvxpy as cp

def dump(prob, filename="problem.bin"):
    """
    Canonicalize a CVXPY problem via the QOCO solver interface and serialize it
    to a binary file.

    The problem is assumed to have the form:
        min  0.5 x'Px + c'x
        s.t. Ax = b
             Gx in K

    where K = {0} x R+^l x SOC(q[0]) x ... x SOC(q[nsoc-1]).

    Binary layout:
        header : n, m, p, l, nsoc, Pnnz, Annz, Gnnz  (int32)
        c      : n float64
        b      : p float64
        h      : m float64
        q      : nsoc int32
        P, A, G: data (float64), indices (int32), indptr (int32)  [CSC format]
    """
    data, _, _ = prob.get_problem_data(cp.QOCO)
    p = data["dims"].zero
    l = data["dims"].nonneg
    q = data["dims"].soc
    m = l + sum(q)
    nsoc = len(q)

    c = data["c"]
    try:
        P = data["P"]
        P = sparse.triu(P, format="csc")
    except:
        P = None

    n = len(c)
    A = data["A"][0:p, :].tocsc()
    b = data["b"][0:p]
    G = data["A"][p:, :].tocsc()
    h = data["b"][p:]

    P = sparse.triu(P).tocsc()

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename, "wb") as f:
        np.array([n, m, p, l, nsoc, P.nnz, A.nnz, G.nnz], dtype=np.int32).tofile(f)

        c.astype(np.float64).tofile(f)
        b.astype(np.float64).tofile(f)
        h.astype(np.float64).tofile(f)
        np.array(q, dtype=np.int32).tofile(f)

        def dump_csc(M):
            M.data.astype(np.float64).tofile(f)
            M.indices.astype(np.int32).tofile(f)
            M.indptr.astype(np.int32).tofile(f)

        dump_csc(P)
        dump_csc(A)
        dump_csc(G)
