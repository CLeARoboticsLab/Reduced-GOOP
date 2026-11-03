Base.@kwdef struct ParametricGOOP{T1,T2,T3}
    "Preference functions, either Vector{Vector{Function}} or Vector{Function}."
    preferences::T1

    "Booleans to indicate if each preference is a constraint, i.e. h(x, θ) ≥ 0."
    is_prioritized_constraint::Vector{Vector{Bool}}

    "Equality and inequality constraints."
    equality_constraints::T2 = nothing
    inequality_constraints::T3 = nothing

    "Dimensions for all relevant quantities."
    primal_dims::Vector{Int}
    parameter_dims::Vector{Int}
    equality_dims::Vector{Int}
    inequality_dims::Vector{Int}

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

    ParametricGOOP(;
        preferences,
        is_prioritized_constraint,
        equality_constraints,
        inequality_constraints,
        primal_dims,
        parameter_dims,
        equality_dims,
        inequality_dims,
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
    QuasiLagrangianTerm(_fast_gradient(expr, x), f.duals, next_order)
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

"""
Construct the Reduced KKT system corresponding to a ParametricGOOP.

`complementarity = :interior_point` (default) builds the interior-point system
(`σ ∘ γ = ρ`) and returns a `GOOPKKTSystem` for `InteriorPoint`. `complementarity =
:scholtes` builds the Scholtes-form system (explicit φ ≥ 0, relaxed `s ∘ γ + u = ρ`)
and returns a `ScholtesKKTSystem` for `Scholtes`; it is opt-in, supports only
`phi = true`, and generates code with FastDifferentiation unless `codegen` says
otherwise, split into functions of `fd_codegen_chunk_size` expressions (default
`SCHOLTES_FD_CHUNK_SIZE`; pass `typemax(Int)` for one function per block). See
src/scholtes_kkt.jl. In both modes, a prioritized constraint h ≥ 0 at level k introduces
explicit preference slacks s: level k minimizes `sum(s)` subject to `h + s ≥ 0` and
`s ≥ 0`, and every upper level keeps these inequalities with fresh multipliers.
"""
function generate_slacked_reduced_kkt_system(
    goop::ParametricGOOP;
    drop_higher_order_terms = false,
    backend_options = (;),
    codegen = nothing,
    fd_codegen_chunk_size = nothing,
    complementarity::Symbol = :interior_point,
    phi::Bool = true,
    quasi_order::Int = 2,
)
    complementarity in (:interior_point, :scholtes) || throw(
        ArgumentError(
            "complementarity must be :interior_point or :scholtes, got $complementarity",
        ),
    )
    if complementarity === :scholtes # TODO: we ultimately want generated_slacked_reduced_kkt_system() to do Scholtes HERE.
        return generate_scholtes_reduced_kkt_system(
            goop;
            quasi = drop_higher_order_terms,
            quasi_order,
            phi,
            codegen = something(codegen, :fast_differentiation),
            fd_codegen_chunk_size = something(fd_codegen_chunk_size, SCHOLTES_FD_CHUNK_SIZE),
            backend_options,
        )
    end
    codegen = something(codegen, :native)
    # Symbolic variables for all primals and parameters.
    backend = SymbolicTracingUtils.SymbolicsBackend()
    @timeit TO "symbolic variable construction" begin
        x =
            SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
            to_blockvector(goop.primal_dims)
        θ =
            SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims)) |>
            to_blockvector(goop.parameter_dims)
        ρ = only(SymbolicTracingUtils.make_variables(backend, :ρ, 1))

        η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

        symbolic_type = eltype(x)
    end

    # The number of rows of each prioritized constraint h ≥ 0, per player and level (0 for a cost).
    preference_dims = [
        [
            prioritized ? length(preference(x, θ)) : 0 for (preference, prioritized) in
            zip(goop.preferences[player], goop.is_prioritized_constraint[player])
        ] for player in 1:(goop.num_players)
    ]

    # Keep track of all the preference (s) and interior point (σ) slacks we create.
    s = symbolic_type[]
    Σ = symbolic_type[]

    # Keep track of all equality constraint duals (λ) and of the duals (ϕ) of the lower levels'
    # complementarity products that we create.
    Λ = symbolic_type[]
    Φ = symbolic_type[]

    # Keep track of all inequality constraint duals (γ) that we create.
    Γ = symbolic_type[]

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

        # This level's inequalities G are g and, for every prioritized level from this one
        # inward, h + s ≥ 0 and s ≥ 0; γ and σ have one entry per row of G (built below).
        num_inequalities =
            goop.inequality_dims[player] + 2 * sum(preference_dims[player][level:end])
        γ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("γ_$(player)_$(level)"),
            num_inequalities,
        )
        push!(Γ, γ...)

        σ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("σ_$(player)_$(level)"),
            num_inequalities,
        )
        push!(Σ, σ...)

        # A prioritized constraint at this level has its own preference slacks.
        preference_slack = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("s_$(player)_$(level)"),
            preference_dims[player][level],
        )
        push!(s, preference_slack...)

        # Base case is the inner-most layer.
        if length(preferences) == 1
            h = only(preferences)(x, θ)

            if only(is_prioritized_constraint)
                # Highest priority is a constraint h ≥ 0: minimize sum(preference_slack)
                # subject to h + preference_slack ≥ 0 and preference_slack ≥ 0.
                preference_inequalities = [h .+ preference_slack; preference_slack]
                G = [isnothing(g) ? symbolic_type[] : g; preference_inequalities]
                @assert length(G) == num_inequalities
                L = sum(preference_slack) - (isnothing(f) ? 0 : λ' * f) - γ' * G
                vars = [x[Block(player)]; preference_slack]

                ∇L, π_terms = if drop_higher_order_terms
                    lagrangian_terms = QuasiLagrangianTerm[]
                    _push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack))
                    _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, G, γ)
                    _quasi_gradient_from_terms(lagrangian_terms, vars)
                else
                    (@timeit TO "symbolic gradient construction" _fast_gradient(L, vars)),
                    QuasiLagrangianTerm[]
                end
                F = Vector{symbolic_type}(
                    filter!(
                        !isnothing,
                        [
                            ∇L .+ η * vars
                            f
                            G .- σ
                            σ .* γ .- ρ
                        ],
                    ),
                )

                return (;
                    F,
                    π = ∇L,
                    π_term_groups = [π_terms],
                    preference_inequalities,
                    complementary_slackness = G .* γ,
                )
            else
                @assert length(h) == 1 "Expected a single preference function at the base level, but got $(length(h))"
                # Highest priority is a cost.

                L = h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
                ∇L, π_terms = if drop_higher_order_terms
                    lagrangian_terms = QuasiLagrangianTerm[]
                    _push_quasi_lagrangian_term!(lagrangian_terms, h)
                    _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                    _push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
                    _quasi_gradient_from_terms(lagrangian_terms, x[Block(player)])
                else
                    (@timeit TO "symbolic gradient construction" _fast_gradient(L, x[Block(player)])),
                    QuasiLagrangianTerm[]
                end
                F = Vector{symbolic_type}(
                    filter!(
                        !isnothing,
                        [
                            ∇L .+ η * x[Block(player)]
                            f
                            (isnothing(g) ? nothing : g .- σ)
                            (isnothing(g) ? nothing : σ .* γ .- ρ)
                        ],
                    ),
                )

                return (;
                    F,
                    π = ∇L,
                    π_term_groups = [π_terms],
                    preference_inequalities = symbolic_type[],
                    complementary_slackness = isnothing(g) ? symbolic_type[] : g .* γ,
                )
            end
        end

        # Handle higher levels via tail recursion.
        (; F, π, π_term_groups, preference_inequalities, complementary_slackness) =
            construct_player_kkt_conditions(
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
        # π ends with the innermost level's stationarity rows: own x, then its preference slacks.
        append!(
            innermost_stationarity_dual_offsets,
            (ψ_start + length(ψ) - preference_dims[player][end] - goop.primal_dims[player]):(ψ_start + length(ψ) - preference_dims[player][end] - 1),
        )

        # Duals of the lower levels' complementarity products Gₘ ⊙ γₘ, m > level.
        ϕ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("ϕ_$(player)_$(level)"),
            length(complementary_slackness),
        )
        push!(Φ, ϕ...)

        if first(is_prioritized_constraint)
            # Current priority is a constraint h ≥ 0: minimize sum(preference_slack) subject to
            # h + preference_slack ≥ 0 and preference_slack ≥ 0.
            preference_inequalities =
                [h .+ preference_slack; preference_slack; preference_inequalities]
            G = [isnothing(g) ? symbolic_type[] : g; preference_inequalities]
            @assert length(G) == num_inequalities
            L =
                sum(preference_slack) - ψ' * π - (isnothing(f) ? 0 : λ' * f) - γ' * G -
                (isempty(complementary_slackness) ? 0 : ϕ' * complementary_slackness)
            vars = [x[Block(player)]; preference_slack]

            ∇L, π_terms = if drop_higher_order_terms
                lagrangian_terms = QuasiLagrangianTerm[]
                _push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack))
                _append_quasi_policy_terms!(lagrangian_terms, π_term_groups, ψ)
                _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                _push_quasi_lagrangian_term!(lagrangian_terms, G, γ)
                _push_quasi_lagrangian_term!(
                    lagrangian_terms,
                    isempty(complementary_slackness) ? nothing : complementary_slackness,
                    ϕ,
                )
                _quasi_gradient_from_terms(lagrangian_terms, vars)
            else
                (@timeit TO "symbolic gradient construction" _fast_gradient(L, vars)),
                QuasiLagrangianTerm[]
            end
        else
            @assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
            # Current priority is a cost.
            G = [isnothing(g) ? symbolic_type[] : g; preference_inequalities]
            @assert length(G) == num_inequalities
            L =
                h - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isempty(G) ? 0 : γ' * G) -
                (isempty(complementary_slackness) ? 0 : ϕ' * complementary_slackness)
            vars = x[Block(player)]

            ∇L, π_terms = if drop_higher_order_terms
                lagrangian_terms = QuasiLagrangianTerm[]
                _push_quasi_lagrangian_term!(lagrangian_terms, h)
                _append_quasi_policy_terms!(lagrangian_terms, π_term_groups, ψ)
                _push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
                _push_quasi_lagrangian_term!(lagrangian_terms, isempty(G) ? nothing : G, γ)
                _push_quasi_lagrangian_term!(
                    lagrangian_terms,
                    isempty(complementary_slackness) ? nothing : complementary_slackness,
                    ϕ,
                )
                _quasi_gradient_from_terms(lagrangian_terms, vars)
            else
                (@timeit TO "symbolic gradient construction" _fast_gradient(L, vars)),
                QuasiLagrangianTerm[]
            end
        end

        F̃ = Vector{symbolic_type}(
            filter!(
                !isnothing,
                [
                    ∇L .+ η * vars
                    G .- σ
                    σ .* γ .- ρ
                    F
                ],
            ),
        )

        return (;
            F = F̃,
            π = vcat(∇L, π),
            π_term_groups = vcat([π_terms], π_term_groups),
            preference_inequalities,
            complementary_slackness = [G .* γ; complementary_slackness],
        )
    end

    # Recursively generate the rest of the KKT conditions for each player.
    # One result per player, always a vector: `mapreduce(vcat, …)` over a single player
    # returns the bare NamedTuple on Julia ≤ 1.12 but a 1-element vector on 1.13.
    F_π_pair = @timeit TO "symbolic expression construction" [
        construct_player_kkt_conditions(
            goop.preferences[player],
            goop.is_prioritized_constraint[player];
            player,
        ) for player in 1:(goop.num_players)
    ]

    # Flatten the F and π vectors for all players.
    @timeit TO "symbolic KKT vector assembly" begin
        flattened_F = reduce(vcat, (pair.F for pair in F_π_pair))

        # Filter out zeros.
        F = Vector{symbolic_type}(
            filter!(
                !isnothing,
                vcat(drop_higher_order_terms ? filter!(!iszero, flattened_F) : flattened_F),
            ),
        )

        # Pack all variables together.
        z = Vector{symbolic_type}(vcat(x, s, Σ, Λ, Γ, Ψ, Φ))
        θ = Vector{symbolic_type}(θ)

        idx = blockedrange(length.([x, s, Σ, Λ, Γ, Ψ, Φ]))
        primal_dims = idx[Block(1)] # x
        preference_slack_dims = idx[Block(2)] # s
        interior_point_slack_dims = vcat(idx[Block(3)]) # Σ
        inequality_constraint_dual_dims = vcat(idx[Block(5)]) # Γ
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
        ρ,
        η,
        primal_dims,
        preference_slack_dims,
        interior_point_slack_dims,
        inequality_constraint_dual_dims;
        equality_constraint_dual_dims,
        stationarity_dual_dims,
        all_equality_stationarity_dual_dims,
        innermost_stationarity_dual_dims,
        backend_options,
        codegen,
        fd_codegen_chunk_size,
    )
end

"Construct the Reduced Quasi-KKT system corresponding to a ParametricGOOP."
function generate_slacked_quasi_kkt_system(
    goop::ParametricGOOP;
    backend_options = (;),
    codegen = nothing,
    fd_codegen_chunk_size = nothing,
    complementarity::Symbol = :interior_point,
    phi::Bool = true,
    quasi_order::Int = 2,
)
    generate_slacked_reduced_kkt_system(
        goop;
        drop_higher_order_terms = true,
        backend_options,
        codegen,
        fd_codegen_chunk_size,
        complementarity,
        phi,
        quasi_order,
    )
end

"""
Construct the complete KKT system corresponding to a ParametricGOOP, equations (2.12)–(2.17)
of manuscript/main.pdf: each level keeps the KKT conditions of every level below it, with
their multipliers. Every player needs at least two levels. As in the manuscript, the
complete system's variables carry a bar (s̄, σ̄, λ̄, γ̄, γ̄ₚ, μ̄ₛ).

For player i, levels k = Kⁱ (innermost) … 1 (outermost); below the innermost level, F = f,
G = g and z = xⁱ. Level k is stationary in [zₖ₊₁; sₖ] for

    Lₖ = cₖ − λₖᵀFₖ₊₁ − γₖᵀGₖ₊₁ − γₚₖᵀ(hₖ + sₖ) − μₛₖᵀsₖ

and passes up, with Cₖ = [Gₖ₊₁ ⊙ γₖ; (hₖ + sₖ) ⊙ γₚₖ; sₖ ⊙ μₛₖ] its complementary slackness rows,

    Fₖ = [∇Lₖ; Cₖ; Fₖ₊₁]        (innermost: [∇Lₖ; f; Cₖ])
    Gₖ = [g; hₖ + sₖ; sₖ; γₖ; γₚₖ; μₛₖ; Gₖ₊₁ without g]
    zₖ = [zₖ₊₁; sₖ; λₖ; γₖ; γₚₖ; μₛₖ].

A prioritized level minimizes cₖ = 1ᵀsₖ subject to hₖ + sₖ ≥ 0 and sₖ ≥ 0; otherwise cₖ is
the cost hₖ, and sₖ, γₚₖ and μₛₖ are empty. A player's residual is F₁, with its inequalities
g and hₘ + sₘ written as equalities with nonnegative slacks σ; s, σ and every γ, γₚ, μₛ are
sign-bounded coordinates.

`complementarity = :exact` is (2.12)–(2.17), with no relaxation. `complementarity =
:perturbed` is (6.1): the same recursion, but each level k gets its own slacks σₖ on g
(g − σₖ = 0), every product row is written with σₖ in place of g, and every product equals
ρ. At ρ = 0 and σₖ = g its residual is the exact system's. Solve it down a ρ schedule.

Returns a `CompleteKKTSystem` for `Scholtes()`.
"""
function generate_slacked_complete_kkt_system(
    goop::ParametricGOOP;
    complementarity::Symbol = :exact,
    backend_options = (;),
    codegen = :fast_differentiation,
    fd_codegen_chunk_size = FD_CODEGEN_CHUNK_SIZE,
)
    complementarity in (:exact, :perturbed) || throw(
        ArgumentError("complementarity must be :exact or :perturbed, got $complementarity"),
    )
    codegen in (:native, :fast_differentiation) ||
        throw(ArgumentError("unknown codegen option: $codegen"))
    for (player, preferences) in enumerate(goop.preferences)
        length(preferences) >= 2 || throw(
            ArgumentError(
                "Complete KKT construction requires K >= 2 for every player; player $player has K = $(length(preferences)).",
            ),
        )
    end
    perturbed = complementarity === :perturbed

    # Symbolic variables for all primals and parameters.
    backend = SymbolicTracingUtils.SymbolicsBackend()
    x =
        SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
        to_blockvector(goop.primal_dims)
    θ =
        SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims)) |>
        to_blockvector(goop.parameter_dims)
    ρ = only(SymbolicTracingUtils.make_variables(backend, :ρ, 1))
    symbolic_type = eltype(x)

    # Keep track of all the preference (s) and interior point (σ) slacks we create, and of the
    # expressions the slacks σ stand for.
    s = symbolic_type[]
    Σ = symbolic_type[]
    slack_expressions = symbolic_type[]

    # Keep track of all equality constraint duals (λ) that we create.
    Λ = symbolic_type[]

    # Keep track of all inequality constraint duals (γ, γₚ, μₛ) that we create.
    Γ = symbolic_type[]

    # Recursive function to construct a player's KKT conditions.
    function construct_player_kkt_conditions(
        preferences,
        is_prioritized_constraint;
        player,
    )
        num_levels = length(goop.preferences[player]) # Kⁱ
        level = 1 + num_levels - length(preferences)

        f =
            goop.equality_dims[player] == 0 ? symbolic_type[] :
            symbolic_type[goop.equality_constraints[player](x, θ)...]
        g =
            goop.inequality_dims[player] == 0 ? symbolic_type[] :
            symbolic_type[goop.inequality_constraints[player](x, θ)...]

        # Below the inner-most level: F = f, G = g and the variables are xⁱ. Above it, the
        # conditions of the levels below. F_ρ holds F's rows as the residual writes them; it
        # differs from F only with :perturbed.
        (; F, F_ρ, G, z, levels) =
            length(preferences) == 1 ?
            (; F = f, F_ρ = f, G = g, z = collect(x[Block(player)]), levels = Any[]) :
            construct_player_kkt_conditions(
                preferences[2:end],
                is_prioritized_constraint[2:end];
                player,
            )
        h = symbolic_type[first(preferences)(x, θ)...]

        λ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("λ̄_$(player)_$(level)"),
            length(F),
        )
        push!(Λ, λ...)

        γ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("γ̄_$(player)_$(level)"),
            length(G),
        )
        push!(Γ, γ...)

        if first(is_prioritized_constraint)
            # Current priority is a constraint: minimize sum(preference_slack) subject to
            # h + preference_slack ≥ 0 and preference_slack ≥ 0.
            preference_slack = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("s̄_$(player)_$(level)"),
                length(h),
            )
            push!(s, preference_slack...)

            γₚ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("γ̄ₚ_$(player)_$(level)"),
                length(h),
            )
            push!(Γ, γₚ...)

            μₛ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("μ̄ₛ_$(player)_$(level)"),
                length(h),
            )
            push!(Γ, μₛ...)

            L =
                sum(preference_slack) - γₚ' * (h .+ preference_slack) -
                μₛ' * preference_slack - λ' * F - γ' * G
            vars = [z; preference_slack]
            ∇L = _fast_gradient(L, vars)

            # Every inequality times its multiplier; the level above inherits this level's
            # inequalities and multipliers.
            complementary_slackness = [
                G .* γ
                (h .+ preference_slack) .* γₚ
                preference_slack .* μₛ
            ]
            G̃ = [
                g
                h .+ preference_slack
                preference_slack
                γ
                γₚ
                μₛ
                G[(length(g) + 1):end]
            ]

            # The residual writes h + preference_slack ≥ 0 as an equality with slacks σₚ ≥ 0.
            σₚ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("σ̄ₚ_$(player)_$(level)"),
                length(h),
            )
            push!(Σ, σₚ...)
            push!(slack_expressions, (h .+ preference_slack)...)
        else
            @assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
            # Current priority is a cost.
            preference_slack, γₚ, μₛ = symbolic_type[], symbolic_type[], symbolic_type[]

            L = only(h) - λ' * F - γ' * G
            vars = z
            ∇L = _fast_gradient(L, vars)

            # Every inequality times its multiplier; the level above inherits this level's
            # multipliers.
            complementary_slackness = G .* γ
            G̃ = [g; γ; G[(length(g) + 1):end]]
        end

        # The residual writes g ≥ 0 as g − σ = 0 with slacks σ ≥ 0: once (:exact), or once per
        # level (:perturbed), because each level's complementary slackness rows with g (the
        # first rows of G) then use its own σ and equal ρ, σ ⊙ γ = ρ; a σ shared by all levels
        # would force their multipliers on g to be equal. At ρ = 0 every σ equals g, and the
        # system is the exact one. 
        if perturbed || length(preferences) == 1 # innermost level
            σ = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("σ̄_$(player)_$(level)"),
                length(g),
            )
            push!(Σ, σ...)
            push!(slack_expressions, g...)
        end
        complementary_slackness_ρ =
            perturbed ?
            [σ .* γ[1:length(g)]; complementary_slackness[(length(g) + 1):end]] .- ρ :
            complementary_slackness

        if length(preferences) == 1
            F̃ = [∇L; F; complementary_slackness]
            F̃_ρ = [∇L; F_ρ; complementary_slackness_ρ]
        else
            F̃ = [∇L; complementary_slackness; F]
            F̃_ρ = [∇L; complementary_slackness_ρ; F_ρ]
        end

        level_record = (; 
            F = F̃,
            G = G̃,
            L,
            lambda = λ,
            gamma = γ,
            preference_slack,
            preference_gamma = γₚ,
            preference_mu = μₛ,
            complementary_slackness,
        )
        return (;
            F = F̃,
            F_ρ = F̃_ρ,
            G = G̃,
            z = [z; preference_slack; λ; γ; γₚ; μₛ],
            levels = vcat(Any[level_record], levels),
        )
    end

    # Recursively generate the KKT conditions of each player.
    player_kkt_conditions = @timeit TO "symbolic expression construction" [
        construct_player_kkt_conditions(
            goop.preferences[player],
            goop.is_prioritized_constraint[player];
            player,
        ) for player in 1:(goop.num_players)
    ]

    # Flatten the residual rows of all players, then write every slacked inequality as an
    # equality.
    F = [
        reduce(vcat, (conditions.F_ρ for conditions in player_kkt_conditions))
        slack_expressions .- Σ
    ]
    G = reduce(vcat, (conditions.G for conditions in player_kkt_conditions))

    # Pack all variables together.
    z = Vector{symbolic_type}(vcat(x, s, Σ, Λ, Γ))
    θ = Vector{symbolic_type}(θ)

    idx = blockedrange(length.([x, s, Σ, Λ, Γ]))
    primal_dims = idx[Block(1)] # x
    preference_slack_dims = idx[Block(2)] # s
    interior_point_slack_dims = vcat(idx[Block(3)]) # Σ
    equality_constraint_dual_dims = vcat(idx[Block(4)]) # Λ
    inequality_constraint_dual_dims = vcat(idx[Block(5)]) # Γ

    @timeit TO "Jacobian / KKT construction" BuildCompleteKKTSystem(
        F,
        z,
        θ,
        ρ,
        primal_dims,
        preference_slack_dims,
        interior_point_slack_dims,
        inequality_constraint_dual_dims;
        equality_constraint_dual_dims,
        complementarity,
        G_symbolic = G,
        slack_expressions,
        levels = [conditions.levels for conditions in player_kkt_conditions],
        backend_options,
        codegen,
        fd_codegen_chunk_size,
    )
end

# Helper functions
# Note: the reduced and complete generators differentiate with `_fast_gradient`, not with
# `SymbolicTracingUtils.gradient`, because it is faster on large systems: both give the same
# expressions (checked string for string, and the compiled residuals and Jacobians bitwise), and
# the intersection game's exact complete system builds in 2.8 s with it against 7.2 s (2026-11-02).

"""
`Symbolics.gradient(O, x)`, expression for expression, without walking all of `O` once per
variable: `executediff` is called only on the terms of a sum that can contain `x[j]` (found
once from each term's variables). Anything that is not a plain sum, or an `x` with repeated
entries, goes to `Symbolics.gradient` itself.
"""
function _fast_gradient(O, x::AbstractVector{Symbolics.Num})
    u = Symbolics.unwrap(O)
    (u isa SymbolicUtils.BasicSymbolic && SymbolicUtils.isadd(u)) ||
        return Symbolics.gradient(O, x)
    column = Dict(Symbolics.unwrap(v) => j for (j, v) in enumerate(x))
    length(column) == length(x) || return Symbolics.gradient(O, x)

    terms = SymbolicUtils.arguments(u)
    candidates = [Int[] for _ in x]
    for (t, term) in enumerate(terms)
        every = false
        for v in Symbolics.get_variables(term)
            j = get(column, v, 0)
            j > 0 ? push!(candidates[j], t) : (every |= !_provably_distinct_leaf(v))
        end
        every && foreach(c -> (isempty(c) || last(c) != t) && push!(c, t), candidates)
    end

    VT = Symbolics.VartypeT
    return map(eachindex(x)) do j
        D = Symbolics.Differential(x[j])
        summed = SymbolicUtils.ArgsT{VT}()
        for t in candidates[j]
            t2 = Symbolics.executediff(D, terms[t])
            Symbolics._iszero(t2) && continue
            push!(summed, t2)
        end
        !isempty(summed) && return Symbolics.Num(SymbolicUtils.add_worker(VT, summed))
        any(t -> Symbolics.occursin_info(D.x, terms[t]), candidates[j]) ?
        Symbolics.Num(SymbolicUtils.add_worker(VT, summed)) :
        Symbolics.Num(Symbolics.COMMON_ZERO)
    end
end

# A leaf that provably is not any `x[j]` when it is not one of them: a scalar symbol, or an
# element indexed by constants.
function _provably_distinct_leaf(v)
    SymbolicUtils.issym(v) && return !SymbolicUtils.is_array_shape(SymbolicUtils.shape(v))
    SymbolicUtils.iscall(v) && SymbolicUtils.operation(v) === getindex || return false
    return all(i -> i isa Integer || SymbolicUtils.isconst(i), @view SymbolicUtils.arguments(v)[2:end])
end

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
