@testset "Create QP" begin
    @load joinpath(@__DIR__, "qp_data.jld2") qp_data
    qp = QuadraticProgram(qp_data.Q, qp_data.q, qp_data.A, qp_data.b, qp_data.G, qp_data.h)

end
