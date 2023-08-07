function cost(qp::QuadraticProgram, x::Vector)
    return 0.5 * dot(x, qp.Q * x) + dot(qp.q, x)
end

function c_eq(qp::QuadraticProgram, x::Vector)
    return qp.A * x - qp.b
end

function c_ineq(qp::QuadraticProgram, x::Vector)
    return qp.G * x - qp.h
end

function c_ineq_lb(qp::QuadraticProgram, x::Vector)
    return qp.lb - x
end

function c_ineq_ub(qp::QuadraticProgram, x::Vector)
    return x - qp.ub
end

function grad_cost(qp::QuadraticProgram, x::Vector)
    return qp.Q * x + qp.q
end

function grad_c_eq(qp::QuadraticProgram, x::Vector)
    return qp.A
end

function grad_c_ineq(qp::QuadraticProgram, x::Vector)
    return qp.G
end

function grad_c_ineq_lb(qp::QuadraticProgram, x::Vector)
    return -I
end

function grad_c_ineq_ub(qp::QuadraticProgram, x::Vector)
    return I
end

function hess_cost(qp::QuadraticProgram, x::Vector)
    return qp.Q
end

function hess_c_eq(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.A, 2), size(qp.A, 2))
end

function hess_c_ineq(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.G, 2), size(qp.G, 2))
end

function hess_c_ineq_lb(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.lb, 1), size(qp.lb, 1))
end

function hess_c_ineq_ub(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.ub, 1), size(qp.ub, 1))
end

