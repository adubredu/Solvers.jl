module Solvers

import ForwardDiff as FD
import FiniteDiff as FnD
using Printf
using LinearAlgebra
using SparseArrays
using DocStringExtensions

include("types.jl")

include("qp/qp.jl")
include("qp/augmented_lagrangian.jl")
include("qp/barrier_method.jl")

export QuadraticProgram
export solve_augmented_lagrangian,
       solve_barrier_method

end
