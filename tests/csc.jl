using SparseArrays, LinearAlgebra

function toCSC(A)
    A = sparse(A)
    println("m: ", A.m)
    println("n: ", A.n)
    println("nnz: ", length(A.nzval))
    println("x: ", A.nzval)
    println("p: ", A.colptr .- 1)
    println("i: ", A.rowval .- 1)
end

function toMatrix(m, n, Ax, Ap, Ai)
    Ai .= Ai .+ 1
    Ap .= Ap .+ 1
    A = SparseMatrixCSC(m, n, Ap, Ai, Ax)
    display(Array(A))
end

let
    A = [1 1 0 0 0 0; 0 1 2 0 0 0]
    toCSC(A)
end