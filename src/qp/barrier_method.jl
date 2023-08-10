"""
$(SIGNATURES)
Return cost augmented with barrier function 

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `x::Vector`: The current solution.
- `θ::Real`: The barrier parameter
"""
function barrier_augmented_cost(qp::QuadraticProgram, x::Vector, θ::Real)
    return cost(qp, x) - θ*sum([log(-σ) for σ in c_ineq(qp, x)])
end

function 