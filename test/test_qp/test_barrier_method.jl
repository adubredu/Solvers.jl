# @testset "Barrier Method QP problem 1" begin
#     @load joinpath(@__DIR__, "qp_data.jld2") qp
#     qp_problem = QuadraticProgram(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h)
#     x = solve_barrier_method(qp_problem)

#     x_expected = [-0.43550583177746516, 0.28372074622818044, -0.5251063315649425, -1.3839636733299523, -1.4220549543622256, 0.5880492269328014, -0.045150442643051895, 1.3474387567012018, 0.47370274986711924, -0.7176674961483989] 

#     @test x ≈ x_expected atol = 1e-6 
# end

@testset "Barrier Method QP problem 2" begin
    Q = [2.0 0.0; 0.0 2.0]
    q = [-2.0; -5.0]
    G = [1.0 2.0; 2.0 3.0]
    h = [3.0; 3.0]
    A = zeros(2,2)
    b =  zeros(2)

    qp_problem = QuadraticProgram(Q, q, A, b, G, h)

    x = solve_barrier_method(qp_problem)
    x_expected = [0.0, 1.0]

    min_val = 0.5 * x' * Q * x + q' * x
    min_val_exp = 0.5 * x_expected' * Q * x_expected + q' * x_expected
    @show min_val
    @show min_val_exp
    @test x ≈ x_expected atol = 1e-3
end
