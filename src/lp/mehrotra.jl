"""
$(SIGNATURES)
Returns the initial primal and dual values for the given linear program.
    Nocedal and Wright (2006. page 410) 

# Arguments
- `lp::LinearProgram`: The linear program to solve.

""" 
function initialize_starting_point(lp::LinearProgram)
    x_tilde = lp.A'*pinv(lp.A*lp.A')*lp.b 
    λ_tilde = pinv(lp.A*lp.A')*lp.A*lp.c
    s_tilde = lp.c - lp.A'*λ_tilde

    δx = max(-1.5*minimum(x_tilde), 0.0)
    δs = max(-1.5*minimum(s_tilde), 0.0)

    x_hat = x_tilde + δx*ones(length(x_tilde))
    s_hat = s_tilde + δs*ones(length(s_tilde))

    δx_hat = 0.5 * dot(x_hat, s_hat) / sum(s_hat)
    δs_hat = 0.5 * dot(x_hat, s_hat) / sum(x_hat)

    x₀ = x_hat + δx_hat*ones(length(x_hat))
    λ₀ = λ_tilde
    s₀ = s_hat + δs_hat*ones(length(s_hat))

    return x₀, λ₀, s₀
end

"""
$(SIGNATURES)
Returns the affine scaling direction 

# Arguments
- `lp::LinearProgram`: The linear program to solve.
- `x::Vector`: Current primal solution
- `s::Vector`: Current dual solution
- `rb::Vector`: The primal residuals
- `rc::Vector`: The dual residuals
"""
function compute_affine_scaling_direction(lp::LinearProgram, x::Vector, s::Vector, rb::Vector, rc::Vector)
    m, n = size(lp.A)
    ρ = 1e-6
    M = [zeros(n, n)+I*ρ    lp.A'   I;
        lp.A zeros(m, m)+I*ρ zeros(m, n)
        diagm(s) zeros(n, m) diagm(x)+I*ρ]
    rhs = [-rc; -rb; -x .* s]
    Δ = M\rhs 
    Δx = Δ[1:n]
    Δλ = Δ[n+1:n+m]
    Δs = Δ[n+m+1:end]
    return Δx, Δλ, Δs
end

"""
$(SIGNATURES)
Returns the primal and dual step lengths 

# Arguments 
`x::Vector`: The current primal solution
`s::Vector`: The current dual solution
`Δx::Vector`: The current primal step direction
`Δs::Vector`: The current dual step direction
"""
function compute_step_lengths(x::Vector, s::Vector, Δx::Vector, Δs::Vector)
    α_pri_max = minimum([-x[i] / Δx[i] for i in 1:length(x) if Δx[i] < 0])
    α_dual_max = minimum([-s[i] / Δs[i] for i in 1:length(s) if Δs[i] < 0])
    η = 0.95
    α_pri = min(1, η * α_pri_max)
    α_dual = min(1, η * α_dual_max)
    return α_pri, α_dual
end 

"""
$(SIGNATURES)
Returns the combined step direction of the input LP 

# Arguments
- `lp::LinearProgram`: The linear program to solve.
- `x::Vector`: Current primal solution
- `s::Vector`: Current dual solution
- `rb::Vector`: The primal residuals
- `rc::Vector`: The dual residuals
- `Δx_aff::Vector`: The affine scaling direction for the primal variables
- `Δs_aff::Vector`: The affine scaling direction for the dual variables
- `σ::Real`: The centering parameter
- `μ::Real`: The duality measure
"""
function solve_combined_direction(lp::LinearProgram, x::Vector, s::Vector, rb::Vector, rc::Vector, Δx_aff::Vector, Δs_aff::Vector, σ::Real, μ::Real)
    m, n = size(lp.A)
    ρ = 1e-6
    M = [zeros(n, n)+I*ρ lp.A' I
        lp.A zeros(m, m)+I*ρ zeros(m, n)
        diagm(s) zeros(n, m) diagm(x)+I*ρ]
    rhs = [-rc; -rb; -x .* s - Δx_aff .* Δs_aff + σ * μ * ones(n)]
    Δ = M \ rhs
    Δx = Δ[1:n]
    Δλ = Δ[n+1:n+m]
    Δs = Δ[n+m+1:end]
    return Δx, Δλ, Δs
end


"""
$(SIGNATURES)
Returns the solution to the input Linear Program solved using Mehrotra's Primal-Dual Interior Point Method

# Arguments
- `lp::LinearProgram`: The linear program to solve.

"""
function solve_mehrotra(lp::LinearProgram; verbose=true, max_iter=100, tol=1e-6)
    # Initialization
    x, λ, s = initialize_starting_point(lp)

    for k = 1:max_iter 
        rb = lp.A*x - lp.b
        rc = lp.A'*λ + s - lp.c
        μ = dot(x, s) / length(x)

        # Check for convergence
        norm_residual = norm([rb; rc])
        if verbose
            println("Iteration: $k, Residual: $norm_residual")
        end
        if norm_residual < tol
            return x, λ, s
        end

        # Compute affine scaling direction
        Δx_aff, Δλ_aff, Δs_aff = compute_affine_scaling_direction(lp, x, s, rb, rc)

        # Compute step lengths for affine scaling direction
        α_pri_aff, α_dual_aff = compute_step_lengths(x, s, Δx_aff, Δs_aff)

        # Compute centering parameter
        μ_aff = dot(x + α_pri_aff * Δx_aff, s + α_dual_aff * Δs_aff) / length(x)
        σ = (μ_aff / μ)^3

        # Compute combined direction 
        Δx, Δλ, Δs = solve_combined_direction(lp, x, s, rb, rc, Δx_aff, Δs_aff, σ, μ)

        # Compute step lengths for combined direction 
        α_pri, α_dual = compute_step_lengths(x, s, Δx, Δs)

        # Update variables
        x += α_pri * Δx
        λ += α_dual * Δλ
        s += α_dual * Δs
    end
    return x, λ, s
end
