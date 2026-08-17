Base.@kwdef struct ParametricGOOP{T1,T2,T3,T4,T5}
    "Preference functions, either Vector{Vector{Function}} or Vector{Function}."
    preferences::T1

    "Booleans to indicate if each preference is a constraint, i.e. h(x, θ) ≥ 0."
    is_prioritized_constraint::Vector{Vector{Bool}}

    "Equality and inequality constraints."
    equality_constraints::T2 = nothing
    inequality_constraints::T3 = nothing

    "Shared equality and inequality constraints."
    shared_equality_constraint::T4 = nothing
    shared_inequality_constraint::T5 = nothing

    "Dimensions for all relevant quantities."
    primal_dims::Vector{Int}
    parameter_dims::Vector{Int}
    equality_dims::Vector{Int}
    inequality_dims::Vector{Int}
    shared_equality_dims::Int = 0
    shared_inequality_dims::Int = 0

    "Number of players."
    num_players::Int
end

"Construct a ParametricGOOP given a test point (x, θ)."
function ParametricGOOP(
    x,
    θ;
    preferences,
    is_prioritized_constraint,
    equality_constraints,
    inequality_constraints,
    shared_equality_constraint,
    shared_inequality_constraint,
)
    # `collect` materializes the result: BlockArrays ≥ 1.10 returns a lazy
    # `BlockedUnitRangeLengths` view, which does not match the `::Vector{Int}` fields.
    primal_dims = collect(BlockArrays.blocklengths(axes(x, 1)))
    parameter_dims = collect(BlockArrays.blocklengths(axes(θ, 1)))
    equality_dims = map(equality_constraints) do f
        isnothing(f) ? 0 : length(f(x, θ))
    end
    inequality_dims = map(inequality_constraints) do g
        isnothing(g) ? 0 : length(g(x, θ))
    end
    shared_equality_dims =
        isnothing(shared_equality_constraint) ? 0 :
        length(shared_equality_constraint(x, θ))
    shared_inequality_dims =
        isnothing(shared_inequality_constraint) ? 0 :
        length(shared_inequality_constraint(x, θ))

    ParametricGOOP(;
        preferences,
        is_prioritized_constraint,
        equality_constraints,
        inequality_constraints,
        shared_equality_constraint,
        shared_inequality_constraint,
        primal_dims,
        parameter_dims,
        equality_dims,
        inequality_dims,
        shared_equality_dims,
        shared_inequality_dims,
        num_players = length(preferences),
    )
end

mutable struct QuasiLagrangianTerm{T<:Symbolics.Num}
    expr::Union{AbstractVector{T},T}
    duals::Union{AbstractVector{T},Nothing}
    deriv_order::Int
end

function Symbolics.gradient(f::QuasiLagrangianTerm, x::AbstractVector{<:Symbolics.Num})
    next_order = f.deriv_order + 1
    if f.deriv_order >= 2 # retain terms up to order 2
        return QuasiLagrangianTerm(zero.(x), f.duals, next_order)
    end

    expr = isnothing(f.duals) ? f.expr : -f.expr' * f.duals
    QuasiLagrangianTerm(Symbolics.gradient(expr, x), f.duals, next_order)
end

function _push_quasi_lagrangian_term!(
    terms::Vector{QuasiLagrangianTerm},
    expr,
    duals = nothing;
    deriv_order = 0,
)
    isnothing(expr) && return nothing
    push!(terms, QuasiLagrangianTerm(expr, duals, deriv_order))
    return nothing
end

function _append_quasi_policy_terms!(
    terms::Vector{QuasiLagrangianTerm},
    π_term_groups::Vector{Vector{QuasiLagrangianTerm}},
    ψ::AbstractVector{<:Symbolics.Num},
)
    offset = 0
    for group in π_term_groups
        isempty(group) && continue
        group_length = length(group[1].expr)
        ψ_block = ψ[(offset + 1):(offset + group_length)]
        for term in group
            push!(terms, QuasiLagrangianTerm(term.expr, ψ_block, term.deriv_order))
        end
        offset += group_length
    end

    @assert offset == length(ψ)
    return nothing
end

function _quasi_gradient_from_terms(
    terms::Vector{QuasiLagrangianTerm},
    x::AbstractVector{<:Symbolics.Num},
)
    ∇L = zero.(x)
    new_terms = QuasiLagrangianTerm[]
    for term in terms
        new_term = Symbolics.gradient(term, x)
        ∇L .+= new_term.expr
        push!(new_terms, new_term)
    end

    return ∇L, new_terms
end

"This is a new preference objective that avoids slack reformulation.
Preference is encoded as h(x) ≥ 0. 
    - If h(x) ≥ 0, preference is satisfied. 
    - If h(x) < 0, preference is violated by an amount -h(x), which is minimized.
Slack reformulation: min_{x,s} s such that s ≥ 0, s ≥ -h(x) <=> min_x max(0, -h(x)))
Equivalently, we consider min_x ifelse(h(x) ≥ 0, 0, (-h(x))^(level+2)), where level is the priority level of the preference.
"

function smooth_piecewise_preference_objective(preference, level; ϵ = 0.0)
    ifelse(preference ≥ ϵ, 0.0, (ϵ - preference)^(level + 2))
end

function smooth_piecewise_preference_objective(
    preference::SymbolicTracingUtils.FD.Node,
    level;
    ϵ = 0.0,
)
    # FastDifferentiation cannot differentiate through ifelse, so use the
    # algebraically identical branch-free form max(ϵ - preference, 0)^(level + 2).
    violation = ϵ - preference
    (0.5 * (violation + abs(violation)))^(level + 2)
end

"Construct the Reduced KKT system corresponding to a ParametricGOOP."
function generate_slacked_reduced_kkt_system(
    goop::ParametricGOOP;
    backend = SymbolicTracingUtils.SymbolicsBackend(),
    drop_higher_order_terms = false,
    backend_options = (;),
    codegen = :native,
    fd_codegen_chunk_size = nothing,
)
    if drop_higher_order_terms &&
       backend isa SymbolicTracingUtils.FastDifferentiationBackend
        error(
            "drop_higher_order_terms = true (the quasi variant) relies on Symbolics-only QuasiLagrangianTerm objects; use backend = SymbolicTracingUtils.SymbolicsBackend().",
        )
    end
    # Main.@infiltrate
    # Symbolic variables for all primals, parameters, and duals for shared constraints.
    @timeit TO "symbolic variable construction" begin
        x =
            SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
            to_blockvector(goop.primal_dims)
        θ =
            SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims)) |>
            to_blockvector(goop.parameter_dims)
        ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))

        η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

        λₛ = SymbolicTracingUtils.make_variables(backend, :λₛ, goop.shared_equality_dims)
        γₛ =
            SymbolicTracingUtils.make_variables(backend, :γₛ, goop.shared_inequality_dims)
        σₛ =
            SymbolicTracingUtils.make_variables(backend, :σₛ, goop.shared_inequality_dims)

        symbolic_type = eltype(x)
    end

    @timeit TO "shared constraint symbolic construction" begin
        fₛ =
            isnothing(goop.shared_equality_constraint) ? nothing :
            goop.shared_equality_constraint(x, θ)
        gₛ =
            isnothing(goop.shared_inequality_constraint) ? nothing :
            goop.shared_inequality_constraint(x, θ)
    end

    # Construct the innermost preference functions for each player and pass it to BuildGOOPKKTSystem as innermost_preference_symbolic.
    innermost_preference_symbolic = @timeit TO "innermost preference construction" begin
        preferences = symbolic_type[]
        for player in 1:goop.num_players
            if goop.is_prioritized_constraint[player][end]
                append!(preferences, goop.preferences[player][end](x, θ))
            end
        end
        isempty(preferences) ? nothing : Vector{symbolic_type}(preferences)
    end

    # Keep track of all the preference (s) and interior point (σ) slacks we create.
    s = symbolic_type[]
    Σ = symbolic_type[]

    # Keep track of all equality constraint duals (λ) that we create.
    Λ = symbolic_type[]
    Φ = symbolic_type[] # 10/25: store duals for complementarity slackness
    Φₛ = symbolic_type[] # store duals for complementarity slackness for shared constraints

    # Keep track of all inequality constraint duals (γ) that we create.
    Γ = symbolic_type[]
    Γ_by_player = [symbolic_type[] for _ in 1:goop.num_players] # 10/25: store duals for complementarity slackness
    Γ_cs_shared_by_player = [symbolic_type[] for _ in 1:goop.num_players] # store duals for complementarity slackness for shared constraints

    # Keep track of all lower level policy constraint duals (ψ) that we create.
    Ψ = symbolic_type[]
    innermost_stationarity_dual_offsets = Int[]

    # Recursive function to construct a player's KKT conditions.
    function construct_player_kkt_conditions(
        preferences,
        is_prioritized_constraint;
        player,
    )
        num_levels = length(goop.preferences[player]) # Kⁱ
        @assert length(preferences) == length(is_prioritized_constraint)
        level = 1 + num_levels - length(preferences)

        f =
            isnothing(goop.equality_constraints[player]) ? nothing :
            goop.equality_constraints[player](x, θ)
        g =
            isnothing(goop.inequality_constraints[player]) ? nothing :
            goop.inequality_constraints[player](x, θ)

        λ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("λ_$(player)_$(level)"),
            goop.equality_dims[player],
        )
        push!(Λ, λ...)

        γ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("γ_$(player)_$(level)"),
            goop.inequality_dims[player],
        )
        push!(Γ, γ...)
        push!(Γ_by_player[player], γ...) # 10/25

        γ̃ₛ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("γ̃ₛ_$(player)_$(level)"),
            goop.shared_inequality_dims,
        )
        push!(Γ, γ̃ₛ...)
        push!(Γ_cs_shared_by_player[player], γ̃ₛ...) # 10/25

        # # Shared constraints exist at every level. https://github.com/CLeARoboticsLab/Quasi-GOOP/issues/6
        # Option (1): Share the multipliers only at all players' innermost levels, but let successive outer levels have their own separate multipliers for all players.

        # (1015)TODO Discuss: interior point slacks only at innermost level. Otherwise,  γ₂ is not enforced?
        σ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("σ_$(player)_$(level)"),
            goop.inequality_dims[player],
        )
        push!(Σ, σ...)

        # Base case is the inner-most layer.
        if length(preferences) == 1
            h = only(preferences)(x, θ)

            if only(is_prioritized_constraint)
                # Highest priority is a constraint.
                # preference_slack = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("s_$(player)_$(level)"),
                # 	length(h), # get the right dimension
                # )
                # push!(s, preference_slack...)

                # σₚ = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("σₚ_$(player)_$(level)"),
                # 	length(h),
                # )
                # push!(Σ, σₚ...)

                # σₚₛ = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("σₚₛ_$(player)_$(level)"),
                # 	length(h),
                # )
                # push!(Σ, σₚₛ...)

                # γₚ = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("γₚ_$(player)_$(level)"),
                # 	length(h),
                # )
                # push!(Γ, γₚ...)

                # μₛ = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("μₛ_$(player)_$(level)"),
                # 	length(h),
                # )
                # push!(Γ, μₛ...)

                # L =
                # 	sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
                # 	(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                # 	(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)

                L =
                    sum(smooth_piecewise_preference_objective.(h, level)) -
                    # γₚ' * (h .+ preference_slack) -
                    (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                    (isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)

                # vars = vcat(x[Block(player)], preference_slack)
                vars = x[Block(player)]

                ∇L, π_terms = if drop_higher_order_terms
                    lagrangian_terms = QuasiLagrangianTerm[]
                    # _push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack .^ 2))
                    # _push_quasi_lagrangian_term!(
                    # 	lagrangian_terms,
                    # 	vcat(h .+ preference_slack, isnothing(P) ? symbolic_type[] : P),
                    # 	γₚ,
                    # )
                    _push_quasi_lagrangian_term!(
                        lagrangian_terms,
                        sum(smooth_piecewise_preference_objective.(h, level)),
                    )
                    _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, fₛ, λₛ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, gₛ, γₛ)
                    _quasi_gradient_from_terms(lagrangian_terms, vars)
                else
                    (@timeit TO "symbolic gradient construction" SymbolicTracingUtils.gradient(
                        L,
                        vars,
                    )),
                    QuasiLagrangianTerm[]
                end
                F = Vector{symbolic_type}(
                    filter!(
                        !isnothing,
                        [
                            ∇L .+ η * vars
                            isnothing(f) ? nothing : f
                            # h .+ preference_slack .- σₚ
                            # σₚ .* γₚ .- ϵ
                            # preference_slack .- σₚₛ
                            # σₚₛ .* μₛ .- ϵ
                            (isnothing(g) ? nothing : g .- σ)
                            (isnothing(g) ? nothing : σ .* γ .- ϵ)
                        ],
                    ),
                )

                return (; F, π = ∇L, π_term_groups = [π_terms])
            else
                @assert length(h) == 1 "Expected a single preference function at the base level, but got $(length(h))"
                # Highest priority is a cost.

                L =
                    h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                    (isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)
                ∇L, π_terms = if drop_higher_order_terms
                    lagrangian_terms = QuasiLagrangianTerm[]
                    _push_quasi_lagrangian_term!(lagrangian_terms, h)
                    _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, fₛ, λₛ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, gₛ, γₛ)
                    _quasi_gradient_from_terms(lagrangian_terms, x[Block(player)])
                else
                    (@timeit TO "symbolic gradient construction" SymbolicTracingUtils.gradient(
                        L,
                        x[Block(player)],
                    )),
                    QuasiLagrangianTerm[]
                end
                F = Vector{symbolic_type}(
                    filter!(
                        !isnothing,
                        [
                            ∇L .+ η * x[Block(player)]
                            f
                            (isnothing(g) ? nothing : g .- σ)
                            (isnothing(g) ? nothing : σ .* γ .- ϵ)
                        ],
                    ),
                )

                return (; F, π = ∇L, π_term_groups = [π_terms])
            end
        end

        # Handle higher levels via tail recursion.
        (; F, π, π_term_groups) = construct_player_kkt_conditions(
            preferences[2:end],
            is_prioritized_constraint[2:end];
            player,
        )
        h = first(preferences)(x, θ)

        ψ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("ψ_$(player)_$(level)"),
            length(π),
        )
        ψ_start = length(Ψ) + 1
        push!(Ψ, ψ...)
        append!(
            innermost_stationarity_dual_offsets,
            (ψ_start + length(ψ) - goop.primal_dims[player]):(ψ_start + length(ψ) - 1),
        )

        # 10/25: added duals for complementarity slackness for lower levels
        ϕ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("ϕ_$(player)_$(level)"),
            goop.inequality_dims[player] * (num_levels - level), # ℓ = 1 to Kⁱ - k
        )
        push!(Φ, ϕ...)

        ϕₛ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("ϕₛ_$(player)_$(level)"),
            goop.shared_inequality_dims * (num_levels - level), # ℓ = 1 to Kⁱ - k
        )
        push!(Φₛ, ϕₛ...)

        λ̃ₛ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("λ̃ₛ_$(player)_$(level)"),
            goop.shared_equality_dims,
        )
        push!(Λ, λ̃ₛ...)

        # γ̃ₛ = SymbolicTracingUtils.make_variables(
        # 	backend,
        # 	Symbol("γ̃ₛ_$(player)_$(level)"),
        # 	goop.shared_inequality_dims,
        # )
        # push!(Γ, γ̃ₛ...)
        # push!(Γ_cs_shared_by_player[player], γ̃ₛ...) # 10/25

        if first(is_prioritized_constraint)
            # Highest priority is a constraint.
            # preference_slack = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("s_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(s, preference_slack...)

            # σₚ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("σₚ_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(Σ, σₚ...)

            # σₚₛ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("σₚₛ_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(Σ, σₚₛ...)

            # γₚ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("γₚ_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(Γ, γₚ...)

            # μₛ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("μₛ_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(Γ, μₛ...)

            # Form reduced Lagrangian at this stage.
            blocked_Γ_cs =
                g === nothing ? nothing :
                make_blocks(Γ_by_player[player], goop.inequality_dims[player])
            blocked_Γ_cs_shared =
                gₛ === nothing ? nothing :
                make_blocks(Γ_cs_shared_by_player[player], goop.shared_inequality_dims)
            # L =
            # 	sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
            # 	ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
            # 	(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) -
            # 	(isnothing(g) ? 0 : ϕ' * (repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)])) -
            # 	(isnothing(gₛ) ? 0 : ϕₛ' * (repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)]))

            L =
                sum(smooth_piecewise_preference_objective.(h, level)) -
                # γₚ' * (h .+ (preference_slack)) -
                ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                (isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) - (
                    isnothing(g) ? 0 :
                    ϕ' * (
                        repeat(g, num_levels - level) .*
                        blocked_Γ_cs[Block(level + 1):Block(num_levels)]
                    )
                ) - (
                    isnothing(gₛ) ? 0 :
                    ϕₛ' * (
                        repeat(gₛ, num_levels - level) .*
                        blocked_Γ_cs_shared[Block(level + 1):Block(num_levels)]
                    )
                )

            # vars = vcat(x[Block(player)], preference_slack)
            vars = x[Block(player)]

            ∇L, π_terms = if drop_higher_order_terms
                lagrangian_terms = QuasiLagrangianTerm[]
                # _push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack .^ 2))
                # _push_quasi_lagrangian_term!(
                # 	lagrangian_terms,
                # 	vcat(h .+ preference_slack, isnothing(P) ? symbolic_type[] : P),
                # 	γₚ,
                # )
                _push_quasi_lagrangian_term!(
                    lagrangian_terms,
                    sum(smooth_piecewise_preference_objective.(h, level)),
                )
                _append_quasi_policy_terms!(lagrangian_terms, π_term_groups, ψ)
                _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                _push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
                _push_quasi_lagrangian_term!(lagrangian_terms, fₛ, λ̃ₛ)
                _push_quasi_lagrangian_term!(lagrangian_terms, gₛ, γ̃ₛ)
                _push_quasi_lagrangian_term!(
                    lagrangian_terms,
                    isnothing(g) ? nothing :
                    repeat(g, num_levels - level) .*
                    blocked_Γ_cs[Block(level + 1):Block(num_levels)],
                    ϕ,
                )
                _push_quasi_lagrangian_term!(
                    lagrangian_terms,
                    isnothing(gₛ) ? nothing :
                    repeat(gₛ, num_levels - level) .*
                    blocked_Γ_cs_shared[Block(level + 1):Block(num_levels)],
                    ϕₛ,
                )
                _quasi_gradient_from_terms(lagrangian_terms, vars)
            else
                (@timeit TO "symbolic gradient construction" SymbolicTracingUtils.gradient(
                    L,
                    vars,
                )),
                QuasiLagrangianTerm[]
            end

            F̃ = [
                ∇L .+ η * vars
                # h .+ preference_slack .- σₚ
                # σₚ .* γₚ .- ϵ
                # preference_slack .- σₚₛ
                # σₚₛ .* μₛ .- ϵ
                (isnothing(g) ? nothing : g .- σ)
                (isnothing(g) ? nothing : σ .* γ .- ϵ)
                (isnothing(gₛ) ? nothing : σₛ .* γ̃ₛ .- ϵ) # Note: same slacks (not duals) for all levels
                F
            ]

            return (;
                F = F̃,
                π = vcat(∇L, π),
                π_term_groups = vcat([π_terms], π_term_groups),
            )
        else
            @assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
            # Current priority is a cost.
            # TODO: dual variables for preference inequalities
            blocked_Γ_cs =
                g === nothing ? nothing :
                make_blocks(Γ_by_player[player], goop.inequality_dims[player])
            blocked_Γ_cs_shared =
                gₛ === nothing ? nothing :
                make_blocks(Γ_cs_shared_by_player[player], goop.shared_inequality_dims)
            L =
                h - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                (isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) - (
                    isnothing(g) ? 0 :
                    ϕ' * (
                        repeat(g, num_levels - level) .*
                        blocked_Γ_cs[Block(level + 1):Block(num_levels)]
                    )
                ) - (
                    isnothing(gₛ) ? 0 :
                    ϕₛ' * (
                        repeat(gₛ, num_levels - level) .*
                        blocked_Γ_cs_shared[Block(level + 1):Block(num_levels)]
                    )
                )

            ∇L, π_terms = if drop_higher_order_terms
                lagrangian_terms = QuasiLagrangianTerm[]
                _push_quasi_lagrangian_term!(lagrangian_terms, h)
                _append_quasi_policy_terms!(lagrangian_terms, π_term_groups, ψ)
                _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                _push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
                _push_quasi_lagrangian_term!(lagrangian_terms, fₛ, λ̃ₛ)
                _push_quasi_lagrangian_term!(lagrangian_terms, gₛ, γ̃ₛ)
                _push_quasi_lagrangian_term!(
                    lagrangian_terms,
                    isnothing(g) ? nothing :
                    repeat(g, num_levels - level) .*
                    blocked_Γ_cs[Block(level + 1):Block(num_levels)],
                    ϕ,
                )
                _push_quasi_lagrangian_term!(
                    lagrangian_terms,
                    isnothing(gₛ) ? nothing :
                    repeat(gₛ, num_levels - level) .*
                    blocked_Γ_cs_shared[Block(level + 1):Block(num_levels)],
                    ϕₛ,
                )
                _quasi_gradient_from_terms(lagrangian_terms, x[Block(player)])
            else
                (@timeit TO "symbolic gradient construction" SymbolicTracingUtils.gradient(
                    L,
                    x[Block(player)],
                )),
                QuasiLagrangianTerm[]
            end
            F̃ = Vector{symbolic_type}(
                filter!(
                    !isnothing,
                    [
                        ∇L .+ η * x[Block(player)]
                        (isnothing(g) ? nothing : g .- σ)
                        (isnothing(g) ? nothing : σ .* γ .- ϵ)
                        (isnothing(gₛ) ? nothing : σₛ .* γ̃ₛ .- ϵ)
                        F
                    ],
                ),
            )

            return (;
                F = F̃,
                π = vcat(∇L, π),
                π_term_groups = vcat([π_terms], π_term_groups),
            )
        end
    end

    # Recursively generate the rest of the KKT conditions for each player.
    F_π_pair = @timeit TO "symbolic expression construction" mapreduce(
        vcat,
        1:(goop.num_players),
    ) do player
        construct_player_kkt_conditions(
            goop.preferences[player],
            goop.is_prioritized_constraint[player];
            player,
        )
    end

    # Flatten the F and π vectors for all players.
    @timeit TO "symbolic KKT vector assembly" begin
        flattened_F = begin
            if length(goop.primal_dims) > 1
                mapreduce(vcat, F_π_pair) do pair
                    pair.F
                end
            else
                F_π_pair.F
            end
        end

        # Filter out zeros and add shared constraints.
        F = Vector{symbolic_type}(
            filter!(
                !isnothing,
                vcat(
                    drop_higher_order_terms ? filter!(!iszero, flattened_F) : flattened_F,
                    fₛ,
                    (isnothing(gₛ) ? nothing : gₛ .- σₛ),
                    (isnothing(gₛ) ? nothing : σₛ .* γₛ .- ϵ),
                ),
            ),
        )

        # 

        # Pack all variables together.
        z = Vector{symbolic_type}(
            vcat(x, s, Σ, Λ, Γ, Ψ, λₛ, γₛ, σₛ, Φ, Φₛ), # 10/25: added ϕ
            # vcat(x, s, Σ, Λ, Γ, Ψ, λₛ, γₛ, σₛ),
        )
        θ = Vector{symbolic_type}(θ)

        idx = blockedrange(length.([x, s, Σ, Λ, Γ, Ψ, λₛ, γₛ, σₛ, Φ, Φₛ])) # 10/25: added ϕ
        # idx = blockedrange(length.([x, s, Σ, Λ, Γ, Ψ, λₛ, γₛ, σₛ]))
        primal_dims = idx[Block(1)] # x
        preference_slack_dims = idx[Block(2)] # s
        interior_point_slack_dims = vcat(idx[Block(3)], idx[Block(9)]) # Σ, σₛ
        inequality_constraint_dual_dims = vcat(idx[Block(5)], idx[Block(8)]) # Γ, γₛ
        equality_constraint_dual_dims = idx[Block(4)] # Λ
        stationarity_dual_dims = idx[Block(6)] # Ψ
        all_equality_stationarity_dual_dims =
            vcat(equality_constraint_dual_dims, stationarity_dual_dims)
        stationarity_offset = sum(length, (x, s, Σ, Λ, Γ))
        innermost_stationarity_dual_dims =
            stationarity_offset .+ innermost_stationarity_dual_offsets
    end

    @timeit TO "Jacobian / KKT construction" BuildGOOPKKTSystem(
        F,
        z,
        θ,
        ϵ,
        η,
        primal_dims,
        preference_slack_dims,
        interior_point_slack_dims,
        inequality_constraint_dual_dims;
        equality_constraint_dual_dims,
        stationarity_dual_dims,
        all_equality_stationarity_dual_dims,
        innermost_stationarity_dual_dims,
        innermost_preference_symbolic,
        backend_options,
        codegen,
        fd_codegen_chunk_size,
    )
end

"Construct the Reduced Quasi-KKT system corresponding to a ParametricGOOP."
function generate_slacked_quasi_kkt_system(
    goop::ParametricGOOP;
    backend = SymbolicTracingUtils.SymbolicsBackend(),
    backend_options = (;),
    codegen = :native,
    fd_codegen_chunk_size = nothing,
)
    generate_slacked_reduced_kkt_system(
        goop;
        backend,
        drop_higher_order_terms = true,
        backend_options,
        codegen,
        fd_codegen_chunk_size,
    )
end

"Construct the Complete KKT system corresponding to a ParametricGOOP."
function generate_slacked_complete_kkt_system(
    goop::ParametricGOOP;
    backend = SymbolicTracingUtils.SymbolicsBackend(),
    backend_options = (;),
    codegen = :native,
    fd_codegen_chunk_size = nothing,
)
    # Symbolic variables for all primals, parameters, and duals for shared constraints.
    x =
        SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
        to_blockvector(goop.primal_dims)
    θ =
        SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims)) |>
        to_blockvector(goop.parameter_dims)
    ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))

    η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

    # λₛ = SymbolicTracingUtils.make_variables(backend, :λₛ, goop.shared_equality_dims)
    # γₛ = SymbolicTracingUtils.make_variables(backend, :γₛ, goop.shared_inequality_dims)
    # σₛ = SymbolicTracingUtils.make_variables(backend, :σₛ, goop.shared_inequality_dims)

    symbolic_type = eltype(x)

    fₛ =
        isnothing(goop.shared_equality_constraint) ? nothing :
        goop.shared_equality_constraint(x, θ)
    gₛ =
        isnothing(goop.shared_inequality_constraint) ? nothing :
        goop.shared_inequality_constraint(x, θ)

    # Keep track of all the preference (s) and interior point (σ) slacks we create.
    s = symbolic_type[]
    Σ = symbolic_type[]

    # Keep track of all equality constraint duals (λ) that we create.
    Λ = symbolic_type[]

    # Keep track of all inequality constraint duals (γ) that we create.
    Γ = symbolic_type[]

    # Recursive function to construct a player's KKT conditions.
    function construct_player_kkt_conditions(
        preferences,
        is_prioritized_constraint;
        player,
    )
        num_levels = length(goop.preferences[player]) # Kⁱ
        @assert length(preferences) == length(is_prioritized_constraint)
        level = 1 + num_levels - length(preferences)

        f =
            isnothing(goop.equality_constraints[player]) ? nothing :
            goop.equality_constraints[player](x, θ)
        g =
            isnothing(goop.inequality_constraints[player]) ? nothing :
            goop.inequality_constraints[player](x, θ)

        # Base case is the inner-most layer.
        if length(preferences) == 1
            λ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("λ_$(player)_$(level)"),
                goop.equality_dims[player],
            )
            push!(Λ, λ...)

            λₛ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("λₛ_$(player)_$(level)"),
                goop.shared_equality_dims,
            )
            push!(Λ, λₛ...)

            γ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("γ_$(player)_$(level)"),
                goop.inequality_dims[player],
            )
            push!(Γ, γ...)

            γₛ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("γₛ_$(player)_$(level)"),
                goop.shared_inequality_dims,
            )
            push!(Γ, γₛ...)

            # σ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("σ_$(player)_$(level)"),
            # 	goop.inequality_dims[player],
            # )
            # push!(Σ, σ...)

            σₛ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("σₛ_$(player)_$(level)"),
                goop.shared_inequality_dims,
            )
            push!(Σ, σₛ...)

            h = only(preferences)(x, θ)

            if only(is_prioritized_constraint)
                # Highest priority is a constraint.
                preference_slack = SymbolicTracingUtils.make_variables(
                    backend,
                    Symbol("s_$(player)_$(level)"),
                    length(h),
                )
                push!(s, preference_slack...)

                # σₚ = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("σₚ_$(player)_$(level)"),
                # 	length(h),
                # )
                # push!(Σ, σₚ...)

                # σₚₛ = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("σₚₛ_$(player)_$(level)"),
                # 	length(h),
                # )
                # push!(Σ, σₚₛ...)

                γₚ = SymbolicTracingUtils.make_variables(
                    backend,
                    Symbol("γₚ_$(player)_$(level)"),
                    length(h),
                )
                push!(Γ, γₚ...)

                # μₛ = SymbolicTracingUtils.make_variables(
                # 	backend,
                # 	Symbol("μₛ_$(player)_$(level)"),
                # 	length(h),
                # )
                # push!(Γ, μₛ...)

                # Form Lagrangian at this stage.
                (; xs, us) = unflatten_trajectory(
                    x[Block(player)],
                    4, #state_dimension,
                    2, # control_dimnesion
                )
                # L =
                # 	sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
                # 	(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                # 	(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)
                L =
                    sum(preference_slack .^ 2) - γₚ' * (h .+ preference_slack) -
                    (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                    (isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)
                # player == 2 && let  # TODO: make this non hard-coded
                # 	goal_deviation = xs[end][1:2] .- θ[Block(2)][5:6] # Player 2
                # 	L += 0.03 * sum(goal_deviation .^ 2)
                # end

                ∇L = SymbolicTracingUtils.gradient(
                    L,
                    vcat(x[Block(player)], preference_slack),
                )

                F = Vector{symbolic_type}(filter!(
                    !isnothing,
                    [
                        ∇L #.+ η * vcat(x[Block(player)], preference_slack)
                        f
                        # h .+ preference_slack .- σₚ
                        # σₚ .* γₚ .- ϵ
                        # preference_slack .- σₚₛ
                        # σₚₛ .* μₛ .- ϵ
                        # isnothing(g) ? nothing : g .- σ
                        # isnothing(g) ? nothing : σ .* γ .- ϵ
                        # isnothing(gₛ) ? nothing : gₛ .- σₛ
                        # isnothing(gₛ) ? nothing : σₛ .* γₛ .- ϵ
                    ],
                ))
                G = Vector{symbolic_type}(
                    filter!(
                        !isnothing,
                        [
                            h .+ preference_slack
                            γₚ
                            ϵ - γₚ' * (h .+ preference_slack)
                            # μₛ
                            isnothing(g) ? nothing : γ
                            isnothing(g) ? nothing : g
                            isnothing(g) ? nothing : ϵ - γ' * g
                        ],
                    ),
                )
                z = Vector{symbolic_type}(
                    vcat(
                        x[Block(player)],
                        preference_slack,
                        γₚ,
                        (isnothing(f) ? [] : λ),
                        (isnothing(fₛ) ? [] : λₛ),
                        (isnothing(g) ? [] : γ),
                        (isnothing(gₛ) ? [] : γₛ),
                    ),
                )
                return (; F, G, z)
            else
                @assert length(h) == 1 "Expected a single preference function at the base level, but got $(length(h))"
                # Highest priority is a cost. 
                L =
                    h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
                    (isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)
                ∇L = SymbolicTracingUtils.gradient(L, x[Block(player)])
                F = Vector{symbolic_type}(filter!(
                    !isnothing,
                    [
                        ∇L #.+ η * x[Block(player)]
                        f
                        # fₛ
                        # isnothing(g) ? nothing : g .- σ
                        # isnothing(g) ? nothing : σ .* γ .- ϵ
                        # isnothing(gₛ) ? nothing : gₛ .- σₛ
                        # isnothing(gₛ) ? nothing : σₛ .* γₛ .- ϵ
                    ],
                ))
                G = Vector{symbolic_type}(
                    filter!(
                        !isnothing,
                        [
                            isnothing(g) ? nothing : γ
                            # isnothing(gₛ) ? nothing : γₛ
                            isnothing(g) ? nothing : g
                            # isnothing(gₛ) ? nothing : gₛ
                            isnothing(g) ? nothing : ϵ - γ' * g
                        ],
                    ),
                )
                z = Vector{symbolic_type}(
                    vcat(
                        x[Block(player)],
                        (isnothing(f) ? [] : λ),
                        (isnothing(fₛ) ? [] : λₛ),
                        (isnothing(g) ? [] : γ),
                        (isnothing(gₛ) ? [] : γₛ),
                    ),
                )
                return (; F, G, z)
            end
        end

        # Handle higher levels via tail recursion.
        (; F, G, z) = construct_player_kkt_conditions(
            preferences[2:end],
            is_prioritized_constraint[2:end];
            player,
        )

        h = first(preferences)(x, θ)

        λ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("λ_$(player)_$(level)"),
            length(F),
        )
        push!(Λ, λ...)

        γ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("γ_$(player)_$(level)"),
            length(G),
        )
        push!(Γ, γ...)

        # σ = SymbolicTracingUtils.make_variables(
        # 	backend,
        # 	Symbol("σ_$(player)_$(level)"),
        # 	length(G),
        # )
        # push!(Σ, σ...)

        if first(is_prioritized_constraint)
            # Highest priority is a constraint.
            preference_slack = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("s_$(player)_$(level)"),
                length(h),
            )
            push!(s, preference_slack...)

            # σₚ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("σₚ_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(Σ, σₚ...)

            # σₚₛ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("σₚₛ_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(Σ, σₚₛ...)

            γₚ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("γₚ_$(player)_$(level)"),
                length(h),
            )
            push!(Γ, γₚ...)

            # μₛ = SymbolicTracingUtils.make_variables(
            # 	backend,
            # 	Symbol("μₛ_$(player)_$(level)"),
            # 	length(h),
            # )
            # push!(Γ, μₛ...)

            # Form Lagrangian at this stage.
            (; xs, us) = unflatten_trajectory(
                x[Block(player)],
                4, #state_dimension,
                2, # control_dimnesion
            )
            # L = sum(preference_slack) + 0.001sum(sum(u .^ 2) for u in us) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -λ' * F - γ' * G
            L =
                sum(preference_slack .^ 2) - γₚ' * (h .+ preference_slack) - λ' * F -
                γ' * G
            ∇L = SymbolicTracingUtils.gradient(L, vcat(z, preference_slack))

            F̃ = Vector{symbolic_type}(filter!(
                !isnothing,
                [
                    ∇L #.+ η * vcat(z, preference_slack)
                    # h .+ preference_slack .- σₚ
                    # σₚ .* γₚ .- ϵ
                    # preference_slack .- σₚₛ
                    # σₚₛ .* μₛ .- ϵ
                    # G .- σ
                    # σ .* γ .- ϵ
                    F
                ],
            ))
            G̃ = Vector{symbolic_type}(
                filter!(
                    !isnothing,
                    [
                        h .+ preference_slack
                        γₚ
                        ϵ - γₚ' * (h .+ preference_slack)
                        # μₛ
                        γ
                        G
                        ϵ - γ' * G
                    ],
                ),
            )
            if level == 1
                σ_g = SymbolicTracingUtils.make_variables(
                    backend,
                    Symbol("σ_g_$(player)_$(level)"),
                    length(G),
                )
                push!(Σ, σ_g...)
                σₚ = SymbolicTracingUtils.make_variables(
                    backend,
                    Symbol("σₚ_$(player)_$(level)"),
                    length(h),
                )
                push!(Σ, σₚ...)

                return (;
                    F = [
                        F̃;
                        h .+ preference_slack .- σₚ;
                        ϵ - γₚ' * (h .+ preference_slack);
                        G .- σ_g;
                        ϵ - γ' * G
                    ],
                    G = G̃,
                    z = Vector{symbolic_type}([z; preference_slack; λ; γ; γₚ; σ_g]),
                )
            end
            return (;
                F = F̃,
                G = G̃,
                z = Vector{symbolic_type}([z; preference_slack; λ; γ; γₚ]),
            )
        else
            @assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
            # Current priority is a cost.
            L = h - λ' * F - γ' * G
            ∇L = SymbolicTracingUtils.gradient(L, z)
            F̃ = Vector{symbolic_type}(filter!(
                !isnothing,
                [
                    ∇L
                    # G .- σ
                    # σ .* γ .- ϵ
                    F
                ],
            ))
            G̃ = Vector{symbolic_type}(filter!(
                !isnothing,
                [
                    γ
                    G
                    ϵ - γ' * G
                ],
            ))
            if level == 1
                σ_g = SymbolicTracingUtils.make_variables(
                    backend,
                    Symbol("σ_g_$(player)_$(level)"),
                    length(G),
                )
                push!(Σ, σ_g...)
                return (;
                    F = [F̃; G .- σ_g; ϵ - γ' * G],
                    G = G̃,
                    z = Vector{symbolic_type}([z; λ; γ; σ_g]),
                )
            end
            return (; F = F̃, G = G̃, z = Vector{symbolic_type}([z; λ; γ]))
        end
    end

    # Recursively generate the rest of the KKT conditions for each player.
    F_G_pair = mapreduce(vcat, 1:(goop.num_players)) do player
        construct_player_kkt_conditions(
            goop.preferences[player],
            goop.is_prioritized_constraint[player];
            player,
        )
    end

    # Flatten the F vectors for all players.
    flattened_F = begin
        if length(goop.primal_dims) > 1
            mapreduce(vcat, F_G_pair) do pair
                pair.F
            end
        else
            F_G_pair.F
        end
    end

    # Filter out zeros.
    F = Vector{symbolic_type}(filter!(!isnothing, vcat(filter!(!iszero, flattened_F))))

    # Pack all variables together.
    z = Vector{symbolic_type}(vcat(x, s, Σ, Λ, Γ))
    θ = Vector{symbolic_type}(θ)

    idx = blockedrange(length.([x, s, Σ, Λ, Γ]))
    primal_dims = idx[Block(1)] # x
    preference_slack_dims = idx[Block(2)] # s
    interior_point_slack_dims = vcat(idx[Block(3)]) # Σ
    inequality_constraint_dual_dims = vcat(idx[Block(5)]) # Γ

    BuildGOOPKKTSystem(
        F,
        z,
        θ,
        ϵ,
        η,
        primal_dims,
        preference_slack_dims,
        interior_point_slack_dims,
        inequality_constraint_dual_dims;
        backend_options,
        codegen,
        fd_codegen_chunk_size,
    )
end

# Helper functions
make_blocks(vec, b) =
    (@assert length(vec) % b == 0; BlockArray(vec, fill(b, length(vec) ÷ b)))

function unflatten_trajectory(z, state_dimension, control_dimension)
    Z = reshape(z, state_dimension + control_dimension, :)
    X = @view Z[1:state_dimension, :]
    U = @view Z[(state_dimension + 1):end, :]
    xs = eachcol(X) .|> collect
    us = eachcol(U) .|> collect
    (; xs, us)
end

function to_blockvector(block_dimensions)
    function (data)
        BlockArrays.BlockArray(data, block_dimensions)
    end
end
