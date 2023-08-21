using LinearAlgebra
using Printf

function mehrotraPCIPM(A, b, c)
    (m, n) = size(A)
    
    # Scaling
    row_scale = vec(sqrt.(sum(A.^2, dims=2)))
    col_scale = vec(sqrt.(sum(A.^2, dims=1)))
    A_scaled = A ./ row_scale ./ col_scale'
    c_scaled = c ./ col_scale
    b_scaled = b ./ row_scale
    
    # Initial feasible solution
    x = col_scale
    s = c_scaled
    λ = b_scaled / m
    
    # Tolerance, maximum iterations, and regularization parameter
    ϵ = 1e-8
    max_iter = 500
    δ = 1e-10
    β = 0.95  # Damping factor for step lengths
    
    for iter = 1:max_iter
        # Compute residuals
        r_dual = A_scaled' * λ + s - c_scaled
        r_pri = A_scaled * x - b_scaled
        μ = dot(x, s) / n
        
        # Check convergence
        @printf("iter: %d, norm(r_dual): %f, norm(r_pri): %f, μ: %f\n", iter, norm(r_dual), norm(r_pri), μ)
        if norm(r_dual) < ϵ && norm(r_pri) < ϵ && μ < ϵ
            return x .* col_scale, λ ./ row_scale, s ./ col_scale
        end
        
        # System matrix
        J = [zeros(n, n) A_scaled' I; A_scaled zeros(m, m) zeros(m, n); Diagonal(s) zeros(n, m) Diagonal(x)] + δ * I
        
        # Right-hand side for predictor step
        Δ_aff = [-r_dual; -r_pri; -μ * ones(n) - x .* s]
        
        # Solve for predictor step
        d_aff = J \ Δ_aff
        d_x_aff = d_aff[1:n]
        d_λ_aff = d_aff[n+1:n+m]
        d_s_aff = d_aff[n+m+1:end]
        
        # Compute step length for predictor step
        α_aff_pri_values = [x[i] / -d_x_aff[i] for i = 1:n if d_x_aff[i] < 0]
        α_aff_pri = isempty(α_aff_pri_values) ? 1 : β * minimum(α_aff_pri_values)

        α_aff_dual_values = [s[i] / -d_s_aff[i] for i = 1:n if d_s_aff[i] < 0]
        α_aff_dual = isempty(α_aff_dual_values) ? 1 : β * minimum(α_aff_dual_values)
        
        # Update for corrector step
        μ_aff = dot(x + α_aff_pri * d_x_aff, s + α_aff_dual * d_s_aff) / n
        σ = (μ_aff / μ)^2
        
        # Right-hand side for corrector step
        Δ_corr = [-r_dual; -r_pri; σ * μ * ones(n) - (x + α_aff_pri * d_x_aff) .* (s + α_aff_dual * d_s_aff)]
        
        # Solve for corrector step
        d_corr = J \ Δ_corr
        d_x_corr = d_corr[1:n]
        d_λ_corr = d_corr[n+1:n+m]
        d_s_corr = d_corr[n+m+1:end]
        
        # Compute step length for corrector step
        α_corr_pri_values = [x[i] / -d_x_corr[i] for i = 1:n if d_x_corr[i] < 0]
        α_corr_pri = isempty(α_corr_pri_values) ? 1 : β * minimum(α_corr_pri_values)

        α_corr_dual_values = [s[i] / -d_s_corr[i] for i = 1:n if d_s_corr[i] < 0]
        α_corr_dual = isempty(α_corr_dual_values) ? 1 : β * minimum(α_corr_dual_values)
        
        # Update solution
        x += α_corr_pri * d_x_corr
        λ += α_corr_dual * d_λ_corr
        s += α_corr_dual * d_s_corr
    end
    
    error("Maximum iterations reached")
end

# Test
A = [1 1; 1 0; 0 1]
b = [1, 0, 0]
c = [-1, -1]

x, λ, s = mehrotraPCIPM(A, b, c)
println("Optimal solution: ", x)


# using JuMP, OSQP
# model = JuMP.Model(OSQP.Optimizer)
# @variable(model, x[1:2])
# @objective(model, Min, c' * x)
# @constraint(model, A * x ≤ b)
# optimize!(model)
# value.(x)