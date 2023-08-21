abstract type Program end 

struct LinearProgram <: Program
    #=
    min c'x
    s.t.    Ax = b
            x >= 0
    =#
    A::AbstractMatrix{Float64}
    b::AbstractArray{Float64}
    c::AbstractArray{Float64}
end

struct QuadraticProgram <: Program 
    #=
    min 1/2 x'Qx + q'x
    s.t.    Ax = b
            Gx <= h
    =#
    Q::AbstractMatrix{Float64}
    q::AbstractArray{Float64}
    A::AbstractMatrix{Float64}
    b::AbstractArray{Float64}
    G::AbstractMatrix{Float64}
    h::AbstractArray{Float64}
end