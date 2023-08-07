function cost(qp::QuadraticProgram, x::Vector)
    return 0.5 * dot(x, qp.Q * x) + dot(qp.q, x)
end

function c_eq(qp::QuadraticProgram, x::Vector)
    return qp.A * x - qp.b
end

function c_ineq(qp::QuadraticProgram, x::Vector)
    return qp.G * x - qp.h
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

function hess_cost(qp::QuadraticProgram, x::Vector)
    return qp.Q
end

function hess_c_eq(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.A, 2), size(qp.A, 2))
end

function hess_c_ineq(qp::QuadraticProgram, x::Vector)
    return zeros(size(qp.G, 2), size(qp.G, 2))
end

