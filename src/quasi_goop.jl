struct ParametricQuasiGOOP{T1, T2, T3, T4, T5, T6, T7, T8}
    "Objective functions for all players"
    objectives::T1
    "Equality constraints for all players"
    private_inner_equality_constraints::T2
    "Inequality constraints for all players"
    private_inner_inequality_constraints::T3
    "Shared equality constraint"
    shared_equality_constraints::T4
    "Shared inequality constraint"
    shared_inequality_constraints::T5

    "Dimension of parameter vector"
    parameter_dimensions::T6
    "Dimension of primal variables for all players"
    primal_dimensions::T7
    "Dimension of equality constraints for all players"
    equality_dimensions::T7
    "Dimension of inequality constraints for all players"
    inequality_dimensions::T7
    "Dimension of shared equality constraint"
    shared_equality_dimension::T8
    "Dimension of shared inequality constraint"
    shared_inequality_dimension::T8

    "Corresponding Primal Dual System Representation."
    pd_system::PrimalDualSysEqn 
end

struct PrimalDualSysEqn{T1, T2, T3}
    "A callable function that computes F(x, λ; θ, μ)"
    x::T1
    "Dual variables"
    λ::Vector{Float64}
    "Lagrangian function"
    L::Function
    "Jacobian of the Lagrangian function"
    J::Function
end
