"""
$(SIGNATURES)
Returns the Lagrangian augmented with a barrier function 

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `x::Vector`: The current solution.
- `μ::Vector`: The inequality constraint dual variable.
- `λ::Vector`: The equality constraint dual variable.
- `ρ::Real`: The barrier gain parameter.
"""
function barrier_augmented_lagrangian(qp::QuadraticProgram, x::Vector, μ::Vector, λ::Vector, ρ::Real)
    return cost(qp, x) + λ' * c_eq(qp, x) + ρ*sum(log.(c_ineq(qp, x)))
end

"""
$(SIGNATURES)
Solves the given quadratic program using the Interior Point method.

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `verbose::Bool`: Whether to print the progress of the solver.
- `max_iter::Int`: The maximum number of iterations.
- `tol::Real`: The tolerance for the stopping criterion.
"""
function solve_interior_point(qp::QuadraticProgram; verbose=true, max_iter=100, tol=1e-6)
    x = zeros(size(qp.Q, 1))
    λ = zeros(size(qp.A, 1))
    μ = zeros(size(qp.G, 1))
    ρ = 1.0
    ϕ = 10.0

    if verbose
        @printf "Iter     Cost     |Eq. C.|     |Ineq. C.|     Barrier     L     |∇ₓL|\n"
    end

    for iter in 1:max_iter
        g = FD.gradient(_x -> barrier_augmented_lagrangian(qp, _x, μ, λ, ρ), x)
        if norm(g) < tol
            λ += ρ * c_eq(qp, x)
            μ = max.(zero(μ), μ + ρ * c_ineq(qp, x))
            ρ *= ϕ
        else
            H = FD.hessian(_x -> barrier_augmented_lagrangian(qp, _x, μ, λ, ρ), x)
            Δx = -H \ g
            α = backtrack(_x -> barrier_augmented_lagrangian(qp, _x, μ, λ, ρ), x, Δx, g)
            x += α * Δx
            λ += ρ * c_eq(qp, x)
            μ = max.(zero(μ), μ + ρ * c_ineq(qp, x))
        end

        if verbose
            log_iteration(qp, iter, g, x, μ, λ, ρ)
        end
    end

end
