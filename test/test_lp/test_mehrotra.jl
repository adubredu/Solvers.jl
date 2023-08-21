using Revise 
using Solvers

A = [1 0 0 1 0 0;
     0 1 0 0 1 0;
     0 0 1 0 0 1;
     1 1 1 0 0 0;
     0 0 0 1 1 1.0]
b = [8, 5, 2, 6, 9.0]
c = [5, 5, 3, 6, 4, 1.0]

lp = LinearProgram(A, b, c)
initial_x, initial_λ, initial_s = solve_mehrotra(lp)