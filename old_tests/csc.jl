using SparseArrays, LinearAlgebra

function toCSC(A)
    A = sparse(A)
    println("m: ", A.m)
    println("n: ", A.n)
    println("nnz: ", length(A.nzval))
    println("x: ", A.nzval)
    println("i: ", A.rowval .- 1)
    println("p: ", A.colptr .- 1)
end

function toMatrix(m, n, Ax, Ap, Ai)
    Ai .= Ai .+ 1
    Ap .= Ap .+ 1
    A = SparseMatrixCSC(m, n, Ap, Ai, Ax)
    display(Array(A))
    return A
end

begin
    p = 2
    m = 6
    n = 6
    P = diagm([1, 2, 3, 4, 5, 6])
    A = [1 1 0 0 0 0; 0 1 2 0 0 0]
    G = -I(6)
    K = [P A' G'; zeros(p, m) zeros(p, p) zeros(p, m); zeros(m, n) zeros(m, p) zeros(m, m)]
    K = sparse(K)

    x = [1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 1.000, 1.000, 1.000, 2.000, -1.000, -2.164, -1.000, -3.745, -1.000, -3.218, -1.000, -1.385, -1.000, 0.779, -1.093, -1.000, 0.802, -0.284, -1.109]
    i = [0, 1, 2, 3, 4, 5, 0, 1, 1, 2, 0, 8, 1, 9, 2, 10, 3, 11, 4, 11, 12, 5, 11, 12, 13]
    p = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 25]
    KKT = toMatrix(14, 14, x, p, i)
end