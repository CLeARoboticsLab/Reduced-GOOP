""" Practice: write a standard-form optimization problem as a MCP.
i.e., take a problem of the form
             min_{x ∈ Rⁿ} f(x, θ)
             s.t.            g(x, θ) = 0
                             h(x, θ) ≥ 0

where θ is a vector of parameters. Express it in the following form:
             find   z
             s.t.   F(z, θ) ⟂ z̲ ≤ z ≤ z̅
where we interpret z = (x, λ, μ), with λ and μ the Lagrange multipliers
for the constraints g and h, respectively. The expression F(z) ⟂ z̲ ≤ z ≤ z̅
should be read as the following three statements:
            - if z = z̲, then F(z, θ) ≥ 0
            - if z̲ < z < z̅, then F(z, θ) = 0
            - if z = z̅, then F(z, θ) ≤ 0

For more details, please consult the documentation for the package
`Complementarity.jl`, which may be found here:
https://github.com/chkwon/Complementarity.jl/tree/master
"""

"Generic description of a constrained optimization problem."
function ParametricOptimizationProblem(;
    objective,
    equality_constraint,
    inequality_constraint,
    parameter_dimension = 1,
    primal_dimension,
    equality_dimension,
    inequality_dimension,
    num_players,
)
    @assert primal_dimension % num_players == 0 "Primal dimension must be divisible by number of players."
    @assert parameter_dimension % num_players == 0 "Parameter dimension must be divisible by number of players."

    total_dimension = primal_dimension + equality_dimension + inequality_dimension

    # Define symbolic variables for this MCP.
    # z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
    # z = BlockArray(z̃, [primal_dimension, equality_dimension, inequality_dimension])

    # x = z[Block(1)]
    # x = BlockArray(x, fill(primal_dimension ÷ num_players, num_players))
    # λ = z[Block(2)]
    # μ = z[Block(3)]

    backend = SymbolicTracingUtils.SymbolicsBackend()
    x = SymbolicTracingUtils.make_variables(backend, :x, primal_dimension) |> 
        to_blockvector(fill(primal_dimension ÷ num_players, num_players))
    λ = SymbolicTracingUtils.make_variables(backend, :λ, equality_dimension)
    μ = SymbolicTracingUtils.make_variables(backend, :μ, inequality_dimension)
    ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
    η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

    symbolic_type = eltype(x)

    # Define a symbolic variable for the parameters.
    # Symbolics.@variables θ̃[1:(parameter_dimension)]
    # θ = Symbolics.scalarize(θ̃)
    # θ = BlockArray(θ, fill(parameter_dimension ÷ num_players, num_players))
    θ = SymbolicTracingUtils.make_variables(backend, :θ, parameter_dimension) |>
        to_blockvector(fill(parameter_dimension ÷ num_players, num_players))

    # Build symbolic expressions for objective and constraints.
    f = objective(x, θ)
    g = isnothing(equality_constraint) ? Symbolics.Num[] : equality_constraint(x, θ)
    h = isnothing(inequality_constraint) ? Symbolics.Num[] : inequality_constraint(x, θ)

    # Build Lagrangian.
    L = f - λ' * g - μ' * h

    # Build F = [∇ₓL, g, h]'.
    ∇ₓL = Symbolics.gradient(L, x)
    F = Vector{symbolic_type}([∇ₓL; g; h])

    primal_dims = 1:primal_dimension
    preference_slack_dims = 1:0
    interior_point_slack_dims = 1:0
    inequality_constraint_dual_dims = primal_dimension + equality_dimension + 1 : total_dimension

    z = Vector{symbolic_type}([x; λ; μ])
    θ = Vector{symbolic_type}(θ)

    BuildGOOPKKTSystem(
        F,
        z,
        θ,
        ϵ,
        η,
        primal_dims,
        preference_slack_dims,
        interior_point_slack_dims,
        inequality_constraint_dual_dims,
    )
end