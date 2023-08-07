abstract type Program end 

struct QuadraticProgram <: Program 
    #=
    min 1/2 x'Qx + q'x
    s.t.    Ax = b
            Gx <= h
            lb <= x <= ub

    =#
    Q::SparseMatrixCSC{Float64,Int64}
    q::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int64}
    b::Vector{Float64}
    G::SparseMatrixCSC{Float64,Int64}
    h::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
end