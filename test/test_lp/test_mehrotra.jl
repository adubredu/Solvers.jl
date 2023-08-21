@testset "Mehrotra LP Problem 1" begin
    A = [1 0 0 1 0 0;
        0 1 0 0 1 0;
        0 0 1 0 0 1;
        1 1 1 0 0 0;
        0 0 0 1 1 1.0]
    b = [8, 5, 2, 6, 9.0]
    c = [5, 5, 3, 6, 4, 1.0]

    lp = LinearProgram(A, b, c)
    x, _, _ = solve_mehrotra(lp; verbose=false)

    using JuMP, GLPK 
    model = JuMP.Model(GLPK.Optimizer)
    set_silent(model)
    m,n = size(A)
    @variable(model, x_var[1:n])
    @objective(model, Min, c'*x_var)
    @constraint(model, A*x_var == b)
    @constraint(model, x_var .>= 0.0)
    optimize!(model)
    x_expected = value.(x_var)

    @show x_expected
    @test x ≈ x_expected atol = 1e-6
end