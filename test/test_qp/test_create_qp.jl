@testset "Create QP" begin
    @load joinpath(@__DIR__, "qp_data.jld2") qp
    @test_nowarn qp_problem = QuadraticProgram(qp.Q, qp.q, qp.A, qp.b, qp.G, qp.h) 
end
