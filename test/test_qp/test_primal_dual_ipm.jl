@testset "Primal-Dual Interior Point Method QP problem 1" begin
    Q = [0.0 0.0; 0.0 0.0]
    q = [3.0; 2.0]
    G = [1.0 1.0; 2.0 1.0]
    h = [4.0; 6.0]
    A = zeros(2,2)
    b =  zeros(2)

    qp_problem = QuadraticProgram(Q, q, A, b, G, h)

    x = solve_barrier_method(qp_problem)
    x_expected = [0.0, 1.0] 
    @test x ≈ x_expected atol = 1e-3
end
