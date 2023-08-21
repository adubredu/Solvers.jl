function cost(lp::LinearProgram, x::Vector)
    return dot(lp.c, x)
end

function c_eq(lp::LinearProgram, x::Vector)
    return lp.A * x - lp.b
end

function c_ineq(lp::LinearProgram, x::Vector)
    return -x
end

function grad_cost(lp::LinearProgram, x::Vector)
    return lp.c
end

function grad_c_eq(lp::LinearProgram, x::Vector)
    return lp.A
end

function grad_c_ineq(lp::LinearProgram, x::Vector)
    return -eye(length(x))
end

function hess_cost(lp::LinearProgram, x::Vector)
    return zeros(length(x), length(x))
end

function hess_c_eq(lp::LinearProgram, x::Vector)
    return zeros(size(lp.A, 2), size(lp.A, 2))
end

function hess_c_ineq(lp::LinearProgram, x::Vector)
    return zeros(length(x), length(x))
end