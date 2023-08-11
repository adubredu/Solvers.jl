"""
$(SIGNATURES)
Return cost augmented with barrier function 

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `x::Vector`: The current solution.
- `θ::Real`: The barrier parameter
"""
function barrier(qp::QuadraticProgram, x::Vector, θ::Real)
    return -θ*sum(log.(-c_ineq(qp, x)))
end 

function barrier_gradient(qp::QuadraticProgram, x::Vector, θ::Real)
    return -θ*qp.G'*c_ineq(qp, x).^(-1)
end

function barrier_hessian(qp::QuadraticProgram, x::Vector, θ::Real)
    return θ*qp.G'*diagm(c_ineq(qp, x).^(-2))*qp.G
end

function solve_newton(qp::QuadraticProgram, x0::Vector, λ0::Vector, θ::Real; verbose=true, max_iter=100, tol=1e-6)
    z = [x0; λ0]
    ρ=1e-3
    for iter=1:max_iter 
        x = z[1:size(qp.Q, 1)] 
        g = [grad_cost(qp, x) + barrier_gradient(qp, x, θ); zero(λ0)]
        if norm(g) > tol 
            H = [hess_cost(qp, x)+barrier_hessian(qp, x, θ)+ρ*I     qp.A'; 
                qp.A    -ρ*I]
            z -= H\g
        else
            verbose && @info "Newton converged to $(norm(g)) after $iter iterations."
            return z
        end
    end
    error("Newton did not converge in $max_iter iterations.")
end 

"""
$(SIGNATURES)
Solves the given quadratic program using the barrier method

# Arguments
- `qp::QuadraticProgram`: The quadratic program to solve.
- `x0::Vector`: The initial solution.
- `verbose::Bool`: Whether to print the progress of the solver.
- `max_iter::Int`: The maximum number of iterations.
- `tol::Real`: The tolerance for the stopping criterion.
"""
function solve_barrier_method(qp::QuadraticProgram, x0::Vector; verbose=true, max_iter=100, tol=1e-6)
    x = x0
    λ = zeros(size(qp.A, 1))
    θ = 1.0
    ϕ = 10.0
    m = size(qp.G, 1)

    if verbose
        @printf "Iter     |Eq. C.|     |Ineq. C.|     Barrier_Param\n"
    end

    for iter in 1:max_iter
        z = solve_newton(qp, x, λ, θ, verbose=true)
        x = z[1:size(qp.Q, 1)]
        λ = z[size(qp.Q, 1)+1:end]
        # θ = max(θ/ϕ, 1e-6)
        θ = θ/ϕ
        if verbose
            println("$iter     $(norm(c_eq(qp, x)))      $(norm(c_ineq(qp, x)))        $θ")
        end
        # if maximum(c_eq(qp, x)) < tol && maximum(c_ineq(qp, x)) < tol
        if m*θ < tol
            verbose && @info "Solver converged in $iter iterations."
            return x, λ
        end
    end
    # error("Solver did not converge in $max_iter iterations.")
    return x, λ
end