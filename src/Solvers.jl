module Solvers

import ForwardDiff as FD
using Printf
using LinearAlgebra
using SparseArrays
using DocStringExtensions

include("types.jl")

include("qp/qp.jl")
include("qp/augmented_lagrangian.jl")

export QuadraticProgram
export solve_augmented_lagrangian

end
