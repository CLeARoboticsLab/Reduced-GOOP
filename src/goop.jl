Base.@kwdef struct ParametricGOOP
    "Vector of callable preference functions for each player, called via h(x, θ)."
    preferences::Vector{Vector{Function}}

    "Booleans to indicate if each preference is a constraint, i.e. h(x, θ) ≥ 0."
    is_prioritized_constraint::Vector{Vector{Bool}}

    "Shared equality and inequality constraints."
    shared_equality_constraint::Union{Nothing,Function} = nothing
    shared_inequality_constraint::Union{Nothing,Function} = nothing

    "Dimensions for all relevant quantities."
    primal_dims::Vector{Int}
    parameter_dims::Int
    shared_equality_dims::Int
    shared_inequality_dims::Int

    "Number of players."
    num_players::Any
end

"Construct a ParametricGOOP given a test point (x, θ)."
function ParametricGOOP(
    x,
    θ;
    preferences,
    is_prioritized_constraint,
    shared_equality_constraint,
    shared_inequality_constraint,
)
    primal_dims = BlockArrays.blocksizes(x)
    parameter_dims = length(θ)
    shared_equality_dims =
        isnothing(shared_equality_constraint):shared_equality_constraint(x, θ)
    shared_inequality_dims =
        isnothing(shared_inequality_constraint) ? 0 : shared_inequality_constraint(x, θ)

    ParametricGOOP(;
        preferences,
        is_prioritized_constraint,
        shared_equality_constraint,
        shared_inequality_constraint,
        primal_dims,
        parameter_dims,
        shared_equality_dims,
        shared_inequality_dims,
        num_players = length(preferences),
    )
end

"Construct the KKT system corresponding to a ParametricGOOP."
function generate_slacked_kkt_system(
    goop::ParametricGOOP;
    backend = SymbolicTracingUtils.SymbolicsBackend(),
)
    # Symbolic variables for all primals, parameters, and duals for shared constraints.
    x =
        SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
        to_blockvector(goop.primal_dims)
    θ = SymbolicTracingUtils.make_variables(backend, :θ, goop.parameter_dims)
    λ̃ = SymbolicTracingUtils.make_variables(backend, :λ̃, goop.shared_equality_dims)
    μ̃ = SymbolicTracingUtils.make_variables(backend, :μ̃, goop.shared_inequality_dims)
    symbolic_type = eltype(x)

    # Keep track of all the preference (s) and interior point (σ) slacks we create.
    s = []
    σ = []

    # Keep track of all inequality constraint duals that we create.
    μ = []

    # Recursive function to construct a player's KKT conditions.
    function construct_player_kkt_conditions(
        preferences,
        is_prioritized_constraint;
        inner_kkt_conditions = nothing,
    )
        # TODO! Incorporate below as base case.
    end

    # Handle the inner-most layer separately for each player.
    inner_kkt_systems = map(1:(goop.num_players)) do player
        h = last(goop.preferences[player])(x, θ)
        f =
            isnothing(goop.shared_equality_constraint) ? nothing :
            goop.shared_equality_constraint(x, θ)
        g =
            isnothing(goop.shared_inequality_constraint) ? nothing :
            goop.shared_inequality_constraint(x, θ)

        if last(goop.is_prioritized_constraint[player])
            # Highest priority is a constraint.
            preference_slack = only(
                SymbolicTracingUtils.make_variables(
                    backend,
                    :s($player)($length(goop.preferences[player])),
                    1,
                ),
            )
            push!(s, preference_slack)

            ip_slack = SymbolicTracingUtils.make_variables(
                backend,
                :σ($player)($length(goop.preferences[player])),
                1 + goop.shared_inequality_dims,
            )
            push!(σ, ip_slack)

            dual = only(
                SymbolicTracingUtils.make_variables(
                    backend,
                    :μ($player)($length(goop.preferences[player])),
                    1,
                ),
            )
            push!(μ, dual)

            L =
                preference_slack - dual * (h - preference_slack) -
                (isnothing(f) ? 0 : λ̃' * f) - (isnothing(g) ? 0 : μ̃' * g)
            ∇L = Symbolics.gradient(vcat(x[Block(player)], preference_slack))
            F = Vector{symbolic_type}(
                filter!(
                    !isnothing,
                    [
                        ∇L
                        h - preference_slack - first(ip_slack)
                        f
                        (isnothing(g) ? g : g - ip_slack[2:end])
                    ],
                ),
            )

            return F
        else
            # Highest priority is a cost.
            ip_slack = SymbolicTracingUtils.make_variables(
                backend,
                :σ($player)($length(goop.preferences[player])),
                goop.shared_inequality_dims,
            )
            push!(σ, ip_slack)

            L = h - (isnothing(f) ? 0 : λ̃' * f) - (isnothing(g) ? 0 : μ̃' * g)
            ∇L = Symbolics.gradient(x[Block(player)])
            F = Vector{symbolic_type}(
                filter!(
                    !isnothing,
                    [
                        ∇L
                        f
                        (isnothing(g) ? g : g - ip_slack)
                    ],
                ),
            )

            return F
        end
    end

    # Recursively generate the rest of the KKT conditions for each player.
    F = mapreduce(vcat, 1:goop.num_players) do player
        construct_player_kkt_conditions(
            goop.preferences[player],
            goop.is_prioritized_constraint[player];
            inner_kkt_conditions = nothing,
        )
    end


    GOOPKKTSystem(
        F,
        z,
        θ,
        preference_slack_dims,
        interior_point_slack_dims,
        inequality_constraint_dual_dims,
    )
end
