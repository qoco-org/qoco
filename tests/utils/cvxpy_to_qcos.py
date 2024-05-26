import cvxpy as cp
from scipy import sparse

def convert(prob):
    data, _, _ = prob.get_problem_data(cp.SCS)
    p = data['dims'].zero
    l = data['dims'].nonneg
    q = data['dims'].soc
    m = l + sum(q)
    nsoc = len(q)

    c = data['c']
    P = data['P']
    P = sparse.triu(P, format='csc')
    n = len(c)

    A = data['A'][0:p,:]
    b = data['b'][0:p]
    
    G = data['A'][p::,:]
    h = data['b'][p::]

    return  n, m, p, P, c, A, b, G, h, l, nsoc, q

