using LinearAlgebra

function mehrotra_ipm(A, b, c)
    # Initialization
    x, λ, s = initialize_starting_point(A, b, c)
    max_iterations = 100
    tolerance = 1e-6
    
    for k in 1:max_iterations
        # Compute residuals
        r_b = A * x - b
        r_c = A' * λ + s - c
        μ = dot(x, s) / length(x)
        
        if norm(r_b) < tolerance && norm(r_c) < tolerance && μ < tolerance
            return x, λ, s
        end
        
        # Solve for affine scaling direction
        Δx_aff, Δλ_aff, Δs_aff = solve_affine_scaling_direction(A, x, s, r_b, r_c)
        
        # Compute step lengths for affine scaling direction
        α_pri_aff, α_dual_aff = compute_step_lengths(x, s, Δx_aff, Δs_aff)
        
        # Compute μ_aff
        μ_aff = dot(x + α_pri_aff * Δx_aff, s + α_dual_aff * Δs_aff) / length(x)
        
        # Centering parameter
        σ = (μ_aff / μ)^3
        
        # Solve for combined direction
        Δx, Δλ, Δs = solve_combined_direction(A, x, s, r_b, r_c, Δx_aff, Δs_aff, σ, μ)
        
        # Compute step lengths for combined direction
        α_pri, α_dual = compute_step_lengths(x, s, Δx, Δs)
        
        # Update variables
        x += α_pri * Δx
        λ += α_dual * Δλ
        s += α_dual * Δs
    end
    
    return x, λ, s
end

function initialize_starting_point(A, b, c)
    x_tilde = A' * pinv(A * A') * b
    λ_tilde = pinv(A * A') * A * c
    s_tilde = c - A' * λ_tilde

    δx = max(-1.5 * minimum(x_tilde), 0)
    δs = max(-1.5 * minimum(s_tilde), 0)

    x_hat = x_tilde + δx * ones(length(x_tilde))
    s_hat = s_tilde + δs * ones(length(s_tilde))

    δx_hat = 0.5 * dot(x_hat, s_hat) / sum(s_hat)
    δs_hat = 0.5 * dot(x_hat, s_hat) / sum(x_hat)

    x0 = x_hat + δx_hat * ones(length(x_hat))
    s0 = s_hat + δs_hat * ones(length(s_hat))
    λ0 = λ_tilde

    return x0, λ0, s0
end

function solve_affine_scaling_direction(A, x, s, r_b, r_c)
    m, n = size(A)
    M = [zeros(n, n) A' I
        A zeros(m, m) zeros(m, n)
        Diagonal(s) zeros(n, m) Diagonal(x)]
    rhs = [-r_c; -r_b; -x .* s]
    Δ = M \ rhs
    return Δ[1:n], Δ[n+1:2n], Δ[2n+1:end]
end


function compute_step_lengths(x, s, Δx, Δs)
    α_pri_max = minimum([-x[i] / Δx[i] for i in 1:length(x) if Δx[i] < 0])
    α_dual_max = minimum([-s[i] / Δs[i] for i in 1:length(s) if Δs[i] < 0])
    η = 0.95
    α_pri = min(1, η * α_pri_max)
    α_dual = min(1, η * α_dual_max)
    return α_pri, α_dual
end

function solve_combined_direction(A, x, s, r_b, r_c, Δx_aff, Δs_aff, σ, μ)
    n = length(x)
    M = [zeros(n, n) A' I; A zeros(n, n) zeros(n, n); Diagonal(s) zeros(n, n) Diagonal(x)]
    rhs = [-r_c; -r_b; -x .* s + σ * μ * ones(n) - Δx_aff .* Δs_aff]
    Δ = M \ rhs
    return Δ[1:n], Δ[n+1:2n], Δ[2n+1:end]
end


# Test
A = [1 0 0 1 0 0
    0 1 0 0 1 0
    0 0 1 0 0 1
    1 1 1 0 0 0
    0 0 0 1 1 1.0]
b = [8, 5, 2, 6, 9.0]
c = [5, 5, 3, 6, 4, 1.0]

x, λ, s = mehrotra_ipm(A, b, c)