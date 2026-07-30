"""
Canonical selective primal--dual warm-start modes.

The repository's established spelling is `:primal_only` (singular).  The modes
are nested with respect to the mathematical partition
`(z, λ, ψ_out, ψ_in)`, where code-level `z` here means the primal block
`mcp.primal_dims`, not the full KKT vector named `z` inside the solver.
"""
const SELECTIVE_WARMSTART_MODES = (
    :primal_only,
    :equality_duals,
    :all_except_innermost_stationarity,
    :all_duals,
)

"""
    kkt_variable_blocks(mcp)

Return fresh index vectors for the selective warm-start partition:

- `z`: primal trajectory variables (`mcp.primal_dims`);
- `λ`: equality-constraint multipliers;
- `ψ_out`: multipliers attached to non-innermost stationarity equations;
- `ψ_in`: multipliers attached to innermost stationarity equations.

`ψ_in` contains multiplier coordinates, not KKT residual-row coordinates.
Every returned vector is newly allocated, so callers may manipulate a block
index list without mutating `mcp` or a later caller's partition.
"""
function kkt_variable_blocks(mcp)
    metadata =
        hasproperty(mcp, :metadata) &&
        getproperty(mcp, :metadata) isa GOOPKKTMetadata ?
        getproperty(mcp, :metadata) : nothing
    if isnothing(metadata)
        z = collect(Int, mcp.primal_dims)
        λ = collect(Int, mcp.equality_constraint_dual_dims)
        stationarity = collect(Int, mcp.stationarity_dual_dims)
        ψ_in = collect(Int, mcp.innermost_stationarity_dual_dims)
    else
        variables = sort(metadata.variables; by = record -> record.index)
        z = [
            record.index for record in variables if record.family === :primal
        ]
        λ = [
            record.index for record in variables if
            record.family === :equality_multiplier
        ]
        stationarity_records = filter(
            record -> record.family === :stationarity_multiplier,
            variables,
        )
        stationarity = getfield.(stationarity_records, :index)
        innermost_level = Dict{Int,Int}()
        for record in stationarity_records
            player = something(record.player)
            innermost_level[player] =
                max(get(innermost_level, player, 0), something(record.target_level))
        end
        ψ_in = [
            record.index for record in stationarity_records if
            record.target_level == innermost_level[something(record.player)]
        ]

        z == collect(Int, mcp.primal_dims) || throw(
            ArgumentError("Semantic primal coordinates disagree with primal_dims."),
        )
        λ == sort(collect(Int, mcp.equality_constraint_dual_dims)) || throw(
            ArgumentError(
                "Semantic equality-multiplier coordinates disagree with " *
                "equality_constraint_dual_dims.",
            ),
        )
        stationarity == sort(collect(Int, mcp.stationarity_dual_dims)) || throw(
            ArgumentError(
                "Semantic stationarity-multiplier coordinates disagree with " *
                "stationarity_dual_dims.",
            ),
        )
        ψ_in == sort(collect(Int, mcp.innermost_stationarity_dual_dims)) || throw(
            ArgumentError(
                "Semantic innermost-stationarity coordinates disagree with " *
                "innermost_stationarity_dual_dims.",
            ),
        )
    end

    missing_ψ_in = setdiff(ψ_in, stationarity)
    isempty(missing_ψ_in) || throw(
        ArgumentError(
            "Innermost-stationarity multiplier coordinates must be a subset of " *
            "stationarity_dual_dims; missing $(missing_ψ_in).",
        ),
    )
    ψ_out = setdiff(stationarity, ψ_in)

    (; z, λ, ψ_out, ψ_in)
end

"""
    build_selective_warmstart(
        shifted_primal,
        previous_z,
        mcp,
        mode;
        dual_transport = :identity_copy,
    )

Build a fresh, full-length KKT initial point for one selective warm-start
`mode`.  All modes use the same default initialization as `solve`: zero for
unselected coordinates and one for preference slacks, interior-point slacks,
and inequality multipliers.  Only the requested `(λ, ψ_out, ψ_in)` blocks are
copied from `previous_z`.

Neither input vector is mutated or aliased by the result.  In particular,
`:all_duals` means all equality and hierarchical-stationarity multipliers in
the stated partition; inequality multipliers and complementarity-equation
multipliers retain the common default initialization.

`dual_transport` is an explicit policy:

- `:identity_copy` (default) preserves the historical same-coordinate copy;
- `:stage_shift_zero_tail` shifts annotated duals by one horizon stage and
  resets coordinates without a successor;
- `:stage_shift_hold_tail` performs the same shift but retains the last
  available same-stage value at a successor tail.

The aliases `:identity` and `:receding` are accepted for compatibility;
`:receding` uses `transport_tail = :zero` or `:hold`.
"""
function build_selective_warmstart(
    shifted_primal::AbstractVector,
    previous_z::AbstractVector,
    mcp,
    mode::Symbol,
    ;
    dual_transport::Symbol = :identity_copy,
    transport_tail::Symbol = :zero,
)
    mode in SELECTIVE_WARMSTART_MODES ||
        throw(ArgumentError("Unknown selective warm-start mode $(mode)."))

    blocks = kkt_variable_blocks(mcp)
    length(shifted_primal) == length(blocks.z) || throw(
        DimensionMismatch(
            "Shifted primal has length $(length(shifted_primal)); " *
            "expected $(length(blocks.z)).",
        ),
    )
    length(previous_z) == mcp.variable_dimension || throw(
        DimensionMismatch(
            "Previous KKT point has length $(length(previous_z)); " *
            "expected $(mcp.variable_dimension).",
        ),
    )
    dual_source = if dual_transport in (:identity_copy, :identity)
        previous_z
    elseif dual_transport === :stage_shift_zero_tail
        transport_receding_duals(previous_z, mcp; tail = :zero)
    elseif dual_transport === :stage_shift_hold_tail
        transport_receding_duals(previous_z, mcp; tail = :hold)
    elseif dual_transport === :receding
        transport_tail in (:zero, :hold) || throw(
            ArgumentError(
                "Unknown transport_tail $(transport_tail); expected :zero or :hold.",
            ),
        )
        transport_receding_duals(previous_z, mcp; tail = transport_tail)
    else
        throw(
            ArgumentError(
                "Unknown dual_transport $(dual_transport); expected " *
                ":identity_copy, :stage_shift_zero_tail, or " *
                ":stage_shift_hold_tail.",
            ),
        )
    end

    T = promote_type(eltype(shifted_primal), eltype(previous_z))
    warmstart = zeros(T, mcp.variable_dimension)
    warmstart[mcp.preference_slack_dims] .= one(T)
    warmstart[mcp.interior_point_slack_dims] .= one(T)
    warmstart[mcp.inequality_constraint_dual_dims] .= one(T)
    warmstart[blocks.z] .= shifted_primal

    if mode !== :primal_only
        warmstart[blocks.λ] .= dual_source[blocks.λ]
    end
    if mode in (:all_except_innermost_stationarity, :all_duals)
        warmstart[blocks.ψ_out] .= dual_source[blocks.ψ_out]
    end
    if mode === :all_duals
        warmstart[blocks.ψ_in] .= dual_source[blocks.ψ_in]
    end

    warmstart
end

build_selective_warmstart(
    shifted_primal,
    previous_z,
    mcp;
    mode::Symbol,
    dual_transport::Symbol = :identity_copy,
    transport_tail::Symbol = :zero,
) = build_selective_warmstart(
    shifted_primal,
    previous_z,
    mcp,
    mode;
    dual_transport,
    transport_tail,
)

"""
    shift_receding_trajectories(strategies, dynamics, planning_horizon)

Shift each per-player trajectory forward by one stage without mutating the
source trajectory.  For an old trajectory `(x₁:T, u₁:T)`, the shifted stages
are `(x₂:T, u₂:T)`.  Complete the terminal state with
`step(x_T, u_T, planning_horizon - 1)`, using the last retained control so the
new last dynamics equation is consistent, then append a zero terminal control
which is unused by the horizon dynamics constraints.
"""
function shift_receding_trajectories(strategies, dynamics, planning_horizon::Integer)
    planning_horizon >= 2 ||
        throw(ArgumentError("A receding-horizon shift requires at least two stages."))
    length(strategies) == length(dynamics) || throw(
        DimensionMismatch(
            "Got $(length(strategies)) strategies for $(length(dynamics)) dynamics models.",
        ),
    )

    map(eachindex(strategies, dynamics)) do player
        strategy = strategies[player]
        player_dynamics = dynamics[player]
        length(strategy.xs) == planning_horizon || throw(
            DimensionMismatch(
                "Player $(player) has $(length(strategy.xs)) states; " *
                "expected $(planning_horizon).",
            ),
        )
        length(strategy.us) == planning_horizon || throw(
            DimensionMismatch(
                "Player $(player) has $(length(strategy.us)) controls; " *
                "expected $(planning_horizon).",
            ),
        )

        xs = [collect(x) for x in strategy.xs[2:end]]
        us = [collect(u) for u in strategy.us[2:end]]
        length(us[end]) == player_dynamics.control_dimension || throw(
            DimensionMismatch(
                "Player $(player)'s retained terminal control has length " *
                "$(length(us[end])); expected $(player_dynamics.control_dimension).",
            ),
        )

        terminal_state = collect(
            player_dynamics.step(xs[end], us[end], planning_horizon - 1),
        )
        length(terminal_state) == length(xs[end]) || throw(
            DimensionMismatch(
                "Player $(player)'s completed terminal state has length " *
                "$(length(terminal_state)); expected $(length(xs[end])).",
            ),
        )
        terminal_control = zero.(us[end])
        push!(xs, terminal_state)
        push!(us, terminal_control)
        (; xs, us)
    end
end
