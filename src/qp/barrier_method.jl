using LinearAlgebra
using Optim

function solve_barrier_method(qp::QuadraticProgram; μ=10, tol=1e-6, max_iter=100, newton_tol=1e-6, newton_max_iter=100)
    Q = qp.Q 
    q = qp.q
    G = qp.G
    h = qp.h
    A = qp.A
    b = qp.b
    m, n = size(G)
    
    # Define the barrier function and its gradient and Hessian
    function barrier_objective(x, t)
        sum = 0.5 * x' * Q * x + q' * x
        for i = 1:m
            if (G[i, :] ⋅ x - h[i]) >= 0
                return Inf
            else
                sum -= (1/t) * log(-G[i, :] ⋅ x + h[i])
            end
        end
        return sum
    end
    
    function barrier_gradient(x, t)
        grad = Q * x + q 
        for i = 1:m 
            grad -= ((1/t) * G[i, :]' / (G[i, :] ⋅ x - h[i]))'
        end
        return grad
    end
    
    function barrier_hessian(x, t)
        H = 1.0*Q
        for i = 1:m
            ai = G[i, :]
            H += (1/t) * (ai * ai') / (G[i, :] ⋅ x - h[i])^2
        end
        return H
    end
    
    # Newton's method
    function newton_method(x0, t; verbose=false)
        x = x0
        for iter = 1:newton_max_iter
            grad = barrier_gradient(x, t) 
            H = barrier_hessian(x, t) 
            Δx = H \ grad
            x -= Δx
            if norm(Δx) < newton_tol
                verbose && @info "Solved Newton in $iter iterations"
                break
            end
        end
        return x
    end
    
    x = zeros(n)
    t = 1.0
    
    println("Iter   Equality   Max_Inequality   t   Cost")
    for iter = 1:max_iter
        # x = newton_method(x, t)

        res = Optim.optimize(x -> barrier_objective(x, t), x)
        x = Optim.minimizer(res)
        
        # Check for convergence
        if m/t < tol
            @info "Solved Barrier Method in $iter iterations"
            break
        end

        # Log the current iteration
        @printf("%3d  % 7.2e   % 7.2e   % 7.2e   % 7.2e\n", iter, norm(A*x-b), maximum(G*x-h), t, barrier_objective(x, t))
        
        # Update t
        t *= μ
    end
    
    return x
end 