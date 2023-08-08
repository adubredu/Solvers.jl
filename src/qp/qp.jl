function cost(qp::QuadraticProgram, x::Vector)
    return 0.5 * dot(x, qp.Q * x) + dot(qp.q, x)
end

function c_eq(qp::QuadraticProgram, x::Vector)
    return qp.A * x - qp.b
end

function c_ineq(qp::QuadraticProgram, x::Vector)
    return qp.G * x - qp.h
end

function grad_cost(qp::QuadraticProgram, x::Vector)
    return qp.Q * x + qp.q
end

function grad_c_eq(qp::QuadraticProgram, x::Vector)
    return qp.A
end

function grad_c_ineq(qp::QuadraticProgram, x::Vector)
    return qp.G
end

function hess_cost(qp::QuadraticProgram, x::Vector)
    return qp.Q
end

function hess_c_eq(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.A, 2), size(qp.A, 2))
end

function hess_c_ineq(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.G, 2), size(qp.G, 2))
end

"""
$(SIGNATURES)
Logs the current iteration of the augmented Lagrangian method.

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `iter::Int`: The current iteration.
- `L_gradient::Vector`: The gradient of the Lagrangian.
- `x::Vector`: The current solution.
- `μ::Vector`: The inequality constraint dual variable.
- `λ::Vector`: The equality constraint dual variable.
- `ρ::Real`: The penalty parameter.
"""
function log_iteration(qp::QuadraticProgram, iter::Int, L_gradient::Vector, x::Vector, μ::Vector, λ::Vector, ρ::Real)
    @printf("%3d   % 7.2e   % 7.2e   % 7.2e   % 7.2e   % 7.2e   %5.0e\n", iter, cost(qp, x), norm(c_eq(qp, x)), maximum(c_ineq(qp, x)), ρ, augmented_lagrangian(qp, x, μ, λ, ρ), norm(L_gradient))
end

