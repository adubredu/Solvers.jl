using LinearAlgebra
using Plots
using Revise

function barrier_hessian(Q, G, h, x, t)
    H = 1.0 * Q
    for i in axes(G, 1)
        ai = G[i, :]
        H += (1 / t) * (ai * ai') / (G[i, :] ⋅ x - h[i])^2
    end
    return H
end

function barrier_gradient(Q, q, G, h, x, t)
    grad = Q * x + q
    for i in axes(G, 1)
        grad -= ((1 / t) * G[i, :]' / (G[i, :] ⋅ x - h[i]))'
    end
    return grad
end

function primal_dual_ipm(Q, q, A, b, G, h, x0, v0, λ0; t=10.0, tol=1e-6, max_iters=100)
    x = x0 
    v = v0
    λ = λ0

    for iter = 1 : max_iters
        ∇f0 = Q*x + q + barrier_gradient(Q, q, G, h, x, t)
        ∇²f0 = Q + barrier_hessian(Q, G, h, x, t)
        Df = G 
        rd = ∇f0 + Df'*λ + A'*v 
        rp = A*x - b
        rc = -diagm(λ)*(G*x-h) - (1/t)*ones(size(h,1))

        # Solve the Newton system
        # [∇²f0 Df' A'] [dx] = [-rd]
        # [-diag(λ)Df  -diag(Gx-h) 0]     [dλ] =  [-rc]
        # [A 0 0]     [dv]   = [-rp]
        ρ = 1e-10
        M = [∇²f0 Df' A'; #2x7
            -diagm(λ)*Df -diagm(G*x-h) zero(G); #3x7
            A zero(G') zero(A)] #2x7
        rhs = [-rd; -rc; -rp]
        sol = (M+ρ*I)\rhs
        Δx = sol[1:2]
        Δλ = sol[3:4]
        Δv = sol[5:6]

        # Backtracking line search 
        α = 1.0 

        # Update
        x = x + α*Δx
        λ = λ + α*Δλ
        v = v + α*Δv

        # Check for convergence
        if norm([rd; rp; rc]) < tol
            @show norm(rd), norm(rp), norm(rc)
            return x
        end

        # Update t
        t = t*100       
        @show iter, norm([rd; rp; rc])

    end


    error("Max iterations reached")
end

# Test data
# Q = [2.0 0.0; 0.0 2.0]
# q = [-2.0; -6.0]
# A = zeros(2,2)
# b = zeros(2)
# G = [1.0 1.0; -1.0 2.0; 2.0 3.0]
# h = [2.0; 2.0; 3.0]

# Q = [0.0 0.0; 0.0 0.0]
# q = [3.0; 2.0]
# G = [1.0 1.0; 2.0 1.0]
# h = [4.0; 6.0]
# A = zeros(2, 2)
# b = zeros(2)

Q =2 * [0.4 0; 0 1]
q = [-5.0; -6.0]
G = [1 -1; -0.3 -1]
h = [-2.0, -8]
A = [0.0 0; 0 0]
b = [0.0, 0.0]

x0 = [0,0]#zeros(size(A,2))
v0 = 0.5*ones(size(A,2)) # Equality dual variable
λ0 = 0.5*ones(size(G, 1)) # Inequality dual variable

result = primal_dual_ipm(Q, q, A, b, G, h, x0, v0, λ0)
expected_result = [4.615384602255649, 6.615384594713718]
println(result)

# Plot the result
plot()
x = range(-15, 15, length=100)
y = range(-15, 15, length=100)
f(x, y) = 0.5*(Q[1,1]*x^2 + Q[2,2]*y^2) + q[1]*x + q[2]*y
constraint1(x) = (h[1] - G[1,1]*x)/G[1,2]
constraint2(x) = (h[2] - G[2, 1] * x) / G[2, 2] 

contour(x, y, f, levels=20) 
plot!(x, constraint1, st=:line, color=:blue, label="Constraint 1")
plot!(x, constraint2, st=:line, color=:brown, label="Constraint 2") 
plot!([result[1]], [result[2]], marker=:circle, markersize=5, color=:red, label="Result")
plot!([expected_result[1]], [expected_result[2]], marker=:circle, markersize=5, color=:green, label="Expected result")
plot!() 

