"""
$(SIGNATURES)
Returns the mask matrix for the given quadratic program.

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `x::Vector`: The current solution.
- `μ::Vector`: The inequality constraint dual variable.
- `ρ::Real`: The penalty parameter.
"""
function mask_matrix(qp::QuadraticProgram, x::Vector, μ::Vector, ρ::Real)
    d = zero(μ)
    h = c_ineq(qp, x)
    for i in eachindex(d)
        if h[i] < 0.0 && μ[i] == 0.0
            d[i] = 0.0
        else
            d[i] = ρ
        end
    end
    return diagm(d)
end

"""
$(SIGNATURES)
Returns the KKT conditions for the given quadratic program.

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `x::Vector`: The current solution.
- `μ::Vector`: The inequality constraint dual variable.
- `λ::Vector`: The equality constraint dual variable.
"""
function KKT_conditions(qp::QuadraticProgram, x::Vector, μ::Vector, λ::Vector)
    return [grad_cost(qp, x) + grad_c_eq(qp, x)' * λ + grad_c_ineq(qp, x)' * μ;
        c_eq(qp, x);
        c_ineq(qp, x);
        μ;
        μ .* c_ineq(qp, x)]
end

"""
$(SIGNATURES)
Returns the augmented Lagrangian for the given quadratic program.

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `x::Vector`: The current solution.
- `μ::Vector`: The inequality constraint dual variable.
- `λ::Vector`: The equality constraint dual variable.
- `ρ::Real`: The penalty parameter.
"""
function augmented_lagrangian(qp::QuadraticProgram, x::Vector, μ::Vector, λ::Vector, ρ::Real)
    Iρ = mask_matrix(qp, x, μ, ρ)
    L = cost(qp, x) + λ' * c_eq(qp, x) + μ' * c_ineq(qp, x) + 0.5
    Lρ = L + 0.5ρ * c_eq(qp, x)' * c_eq(qp, x) + 0.5ρ * c_ineq(qp, x)' * Iρ * c_ineq(qp, x)
    return Lρ
end

"""
$(SIGNATURES)
Logs the current iteration of the augmented Lagrangian method.

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `iter::Int`: The current iteration.
- `AL_gradient::Vector`: The gradient of the augmented Lagrangian.
- `x::Vector`: The current solution.
- `μ::Vector`: The inequality constraint dual variable.
- `λ::Vector`: The equality constraint dual variable.
- `ρ::Real`: The penalty parameter.
"""
function log_iteration(qp::QuadraticProgram, iter::Int, AL_gradient::Vector, x::Vector, μ::Vector, λ::Vector, ρ::Real)
    @printf("%3d   % 7.2e   % 7.2e   % 7.2e   % 7.2e   % 7.2e   %5.0e\n", iter, cost(qp, x), norm(c_eq(qp, x)), maximum(c_ineq(qp, x)), ρ, augmented_lagrangian(qp, x, μ, λ, ρ), norm(AL_gradient))
end

"""
$(SIGNATURES)
Solves the given quadratic program using the augmented Lagrangian method.

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `verbose::Bool`: Whether to print the progress of the solver.
- `max_iter::Int`: The maximum number of iterations.
- `tol::Real`: The tolerance for the stopping criterion.
"""
function solve_augmented_lagrangian(qp::QuadraticProgram; verbose=true, max_iter=100, tol=1e-6)
    x = zeros(size(qp.Q, 1))
    λ = zeros(size(qp.A, 1))
    μ = zeros(size(qp.G, 1))
    ρ = 1.0
    ϕ = 10.0

    if verbose
        @printf "Iter     Cost     |Eq. C.|     |Ineq. C.|     Penalty     AL     |∇ₓAL|\n"
    end

    for iter in 1:max_iter
        g = FD.gradient(_x -> augmented_lagrangian(qp, _x, μ, λ, ρ), x)
        if norm(g) < tol
            λ += ρ * c_eq(qp, x)
            μ = max.(zero(μ), μ + ρ * c_ineq(qp, x))
            ρ *= ϕ
        else
            H = FD.hessian(_x -> augmented_lagrangian(qp, _x, μ, λ, ρ), x)
            x += -H \ g
        end

        if verbose
            log_iteration(qp, iter, g, x, μ, λ, ρ)
        end

        if maximum(c_eq(qp, x)) < tol && maximum(c_ineq(qp, x)) < tol
            verbose && @info "Solver converged in $iter iterations."
            return x, λ, μ
        end
    end
    error("Solver did not converge in $max_iter iterations.")

end
