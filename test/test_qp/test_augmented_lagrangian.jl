@testset "Augmented Lagrangian QP solver" begin
    @load joinpath(@__DIR__, "qp_data.jld2") qp
    qp_problem = QuadraticProgram(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h)
    x, λ, μ = solve_augmented_lagrangian(qp_problem)

    x_expected = [-0.43550583177746516, 0.28372074622818044, -0.5251063315649425, -1.3839636733299523, -1.4220549543622256, 0.5880492269328014, -0.045150442643051895, 1.3474387567012018, 0.47370274986711924, -0.7176674961483989]
    λ_expected = [-0.03814022771932146, -2.99391684081393, -0.951923551611948]
    μ_expected = [0.1013287893271988, 0.3256114505153771, 0.0, 0.5180654288865217, 0.0]

    @test x ≈ x_expected atol = 1e-6
    @test λ ≈ λ_expected atol = 1e-6
    @test μ ≈ μ_expected atol = 1e-6
end