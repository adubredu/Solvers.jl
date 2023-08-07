@testset "Augmented Lagrangian QP solver" begin
    @load joinpath(@__DIR__, "qp_data.jld2") qp_data
    qp = QuadraticProgram(qp_data.Q, qp_data.q, qp_data.A, qp_data.b, qp_data.G, qp_data.h)
    x, λ, μ = solve_qp_augmented_lagrangian(qp)

    x_expected = [-0.32623080431873486, 0.24943798756566352, -0.43226765471113954, -1.417224694812929, -1.3994527462892892, 0.609958243607347, -0.07312201788675675, 1.3031477492933286, 0.5389034765217047, -0.722581370760872]
    λ_expected = [-0.1282341950085557, -2.837717107754904, -0.8322858252247716]
    μ_expected = [0.03640903482216018, 0.0, 0.0, 1.0595504446962325, 0.0]

    @test x ≈ x_expected atol = 1e-6
    @test λ ≈ λ_expected atol = 1e-6
    @test μ ≈ μ_expected atol = 1e-6
end
