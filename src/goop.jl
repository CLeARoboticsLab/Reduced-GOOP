Base.@kwdef struct ParametricGOOP{T1, T2, T3, T4, T5}
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
	primal_dims = only(BlockArrays.blocksizes(x))
	parameter_dims = only(BlockArrays.blocksizes(θ))
	equality_dims = map(equality_constraints) do f
		isnothing(f) ? 0 : length(f(x, θ))
	end
	inequality_dims = map(inequality_constraints) do g
		isnothing(g) ? 0 : length(g(x, θ))
	end
	shared_equality_dims =
		isnothing(shared_equality_constraint) ? 0 : length(shared_equality_constraint(x, θ))
	shared_inequality_dims =
		isnothing(shared_inequality_constraint) ? 0 : length(shared_inequality_constraint(x, θ))

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

mutable struct QuasiLagrangianTerm{T <: Symbolics.Num}
	expr::Union{AbstractVector{T}, T}
	duals::Union{AbstractVector{T}, Nothing}
	deriv_order::Int
end

const _complete_kkt_debug_snapshot = Ref{Any}(nothing)
const _reduced_kkt_debug_snapshot = Ref{Any}(nothing)

function Symbolics.gradient(f::QuasiLagrangianTerm, x::AbstractVector{<:Symbolics.Num})
	next_order = f.deriv_order + 1
	if f.deriv_order > 2
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
		ψ_block = ψ[(offset+1):(offset+group_length)]
		for term in group
			push!(
				terms,
				QuasiLagrangianTerm(term.expr, ψ_block, term.deriv_order),
			)
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

"Construct the Reduced KKT system corresponding to a ParametricGOOP."
function generate_slacked_reduced_kkt_system(
	goop::ParametricGOOP;
	backend = SymbolicTracingUtils.SymbolicsBackend(),
	drop_higher_order_terms = false,
)
	# Symbolic variables for all primals, parameters, and duals for shared constraints.
	x =
		SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
		to_blockvector(goop.primal_dims)
	θ = SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims)) |>
		to_blockvector(goop.parameter_dims)
	ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))

	η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

	λₛ = SymbolicTracingUtils.make_variables(backend, :λₛ, goop.shared_equality_dims)
	γₛ = SymbolicTracingUtils.make_variables(backend, :γₛ, goop.shared_inequality_dims)
	σₛ = SymbolicTracingUtils.make_variables(backend, :σₛ, goop.shared_inequality_dims)

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
	Φ = symbolic_type[] # 10/25: store duals for complementarity slackness
	Φₛ = symbolic_type[] # store duals for complementarity slackness for shared constraints

	# Keep track of all inequality constraint duals (γ) that we create.
	Γ = symbolic_type[]
	Γ_by_player = [symbolic_type[] for _ in 1:goop.num_players] # 10/25: store duals for complementarity slackness
	Γ_cs_shared_by_player = [symbolic_type[] for _ in 1:goop.num_players] # store duals for complementarity slackness for shared constraints

	# Keep track of all lower level policy constraint duals (ψ) that we create.
	Ψ = symbolic_type[]

	# Recursive function to construct a player's KKT conditions.
	function construct_player_kkt_conditions(
		preferences,
		is_prioritized_constraint;
		player,
	)
		num_levels = length(goop.preferences[player]) # Kⁱ
		@assert length(preferences) == length(is_prioritized_constraint)
		level = 1 + num_levels - length(preferences)

		f = isnothing(goop.equality_constraints[player]) ? nothing :
			goop.equality_constraints[player](x, θ)
		g = isnothing(goop.inequality_constraints[player]) ? nothing :
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
				preference_slack = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("s_$(player)_$(level)"),
					length(h), # get the right dimension
				)
				push!(s, preference_slack...)

				σₚ = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("σₚ_$(player)_$(level)"),
					length(h),
				)
				push!(Σ, σₚ...)

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

				# L =
				# 	sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
				# 	(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				# 	(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)

				L =
					sum(preference_slack .^ 2) - γₚ' * (h .+ preference_slack) -
					(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
					(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)

				vars = vcat(x[Block(player)], preference_slack)
				∇L, π_terms = if drop_higher_order_terms
					lagrangian_terms = QuasiLagrangianTerm[]
					_push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack .^ 2))
					_push_quasi_lagrangian_term!(
						lagrangian_terms,
						vcat(h .+ preference_slack, isnothing(P) ? symbolic_type[] : P),
						γₚ,
					)
					_push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
					_push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
					_push_quasi_lagrangian_term!(lagrangian_terms, fₛ, λₛ)
					_push_quasi_lagrangian_term!(lagrangian_terms, gₛ, γₛ)
					_quasi_gradient_from_terms(lagrangian_terms, vars)
				else
					Symbolics.gradient(L, vars), QuasiLagrangianTerm[]
				end

				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L .+ η * vars
							f
							h .+ preference_slack .- σₚ
							σₚ .* γₚ .- ϵ
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

				L = h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
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
					Symbolics.gradient(L, x[Block(player)]), QuasiLagrangianTerm[]
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
		push!(Ψ, ψ...)

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
			preference_slack = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("s_$(player)_$(level)"),
				length(h),
			)
			push!(s, preference_slack...)

			σₚ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("σₚ_$(player)_$(level)"),
				length(h),
			)
			push!(Σ, σₚ...)

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

			# Form reduced Lagrangian at this stage.
			blocked_Γ_cs = g === nothing ? nothing : make_blocks(Γ_by_player[player], goop.inequality_dims[player])
			blocked_Γ_cs_shared = gₛ === nothing ? nothing : make_blocks(Γ_cs_shared_by_player[player], goop.shared_inequality_dims)
			# L =
			# 	sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
			# 	ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
			# 	(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) -
			# 	(isnothing(g) ? 0 : ϕ' * (repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)])) -
			# 	(isnothing(gₛ) ? 0 : ϕₛ' * (repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)]))

			L =
				sum(preference_slack .^ 2) - γₚ' * (h .+ (preference_slack)) -
				ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) -
				(isnothing(g) ? 0 : ϕ' * (repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)])) -
				(isnothing(gₛ) ? 0 : ϕₛ' * (repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)]))

			vars = vcat(x[Block(player)], preference_slack)
			∇L, π_terms = if drop_higher_order_terms
				lagrangian_terms = QuasiLagrangianTerm[]
				_push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack .^ 2))
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					vcat(h .+ preference_slack, isnothing(P) ? symbolic_type[] : P),
					γₚ,
				)
				_append_quasi_policy_terms!(lagrangian_terms, π_term_groups, ψ)
				_push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
				_push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
				_push_quasi_lagrangian_term!(lagrangian_terms, fₛ, λ̃ₛ)
				_push_quasi_lagrangian_term!(lagrangian_terms, gₛ, γ̃ₛ)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					isnothing(g) ? nothing :
					repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)],
					ϕ,
				)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					isnothing(gₛ) ? nothing :
					repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)],
					ϕₛ,
				)
				_quasi_gradient_from_terms(lagrangian_terms, vars)
			else
				Symbolics.gradient(L, vars), QuasiLagrangianTerm[]
			end

			F̃ = [
				∇L .+ η * vars
				h .+ preference_slack .- σₚ
				σₚ .* γₚ .- ϵ
				# preference_slack .- σₚₛ
				# σₚₛ .* μₛ .- ϵ
				(isnothing(g) ? nothing : σ .* γ .- ϵ)
				(isnothing(gₛ) ? nothing : σₛ .* γ̃ₛ .- ϵ) # Note: same slacks (not duals) for all levels
				F
			]

			return (; F = F̃, π = vcat(∇L, π), π_term_groups = vcat([π_terms], π_term_groups))
		else
			@assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
			# Current priority is a cost.
			# TODO: dual variables for preference inequalities
			blocked_Γ_cs = g === nothing ? nothing : make_blocks(Γ_by_player[player], goop.inequality_dims[player])
			blocked_Γ_cs_shared = gₛ === nothing ? nothing : make_blocks(Γ_cs_shared_by_player[player], goop.shared_inequality_dims)
			L =
				h - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) -
				(isnothing(g) ? 0 : ϕ' * (repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)])) -
				(isnothing(gₛ) ? 0 : ϕₛ' * (repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)]))

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
					repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)],
					ϕ,
				)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					isnothing(gₛ) ? nothing :
					repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)],
					ϕₛ,
				)
				_quasi_gradient_from_terms(lagrangian_terms, x[Block(player)])
			else
				Symbolics.gradient(L, x[Block(player)]), QuasiLagrangianTerm[]
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

			return (; F = F̃, π = vcat(∇L, π), π_term_groups = vcat([π_terms], π_term_groups))
		end
	end

	# Recursively generate the rest of the KKT conditions for each player.
	F_π_pair = mapreduce(vcat, 1:(goop.num_players)) do player
		construct_player_kkt_conditions(
			goop.preferences[player],
			goop.is_prioritized_constraint[player];
			player,
		)
	end


	# Flatten the F and π vectors for all players.
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
		filter!(!isnothing,
			vcat(
				filter!(!iszero, flattened_F),
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

"Construct the Reduced Quasi-KKT system corresponding to a ParametricGOOP."
function generate_slacked_quasi_kkt_system(
	goop::ParametricGOOP;
	backend = SymbolicTracingUtils.SymbolicsBackend(),
)
	generate_slacked_reduced_kkt_system(
		goop;
		backend,
		drop_higher_order_terms = true,
	)
end

"Construct the Complete KKT system corresponding to a ParametricGOOP."
function generate_slacked_complete_kkt_system(
	goop::ParametricGOOP;
	backend = SymbolicTracingUtils.SymbolicsBackend(),
)
	# Symbolic variables for all primals, parameters, and duals for shared constraints.
	x =
		SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
		to_blockvector(goop.primal_dims)
	θ = SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims)) |>
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

		f = isnothing(goop.equality_constraints[player]) ? nothing :
			goop.equality_constraints[player](x, θ)
		g = isnothing(goop.inequality_constraints[player]) ? nothing :
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

				∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))

				F = Vector{symbolic_type}(
					filter!(
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
					),
				)
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
				L = h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
					(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)
				∇L = Symbolics.gradient(L, x[Block(player)])
				F = Vector{symbolic_type}(
					filter!(
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
					),
				)
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
			L = sum(preference_slack .^ 2) - γₚ' * (h .+ preference_slack) - λ' * F - γ' * G
			∇L = Symbolics.gradient(L, vcat(z, preference_slack))

			F̃ = Vector{symbolic_type}(
				filter!(
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
				),
			)
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

				return (; F = [F̃; h .+ preference_slack .- σₚ; ϵ - γₚ' * (h .+ preference_slack); G .- σ_g; ϵ - γ' * G], G = G̃, z = Vector{symbolic_type}([z; preference_slack; λ; γ; γₚ; σ_g]))
			end
			return (; F = F̃, G = G̃, z = Vector{symbolic_type}([z; preference_slack; λ; γ; γₚ]))
		else
			@assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
			# Current priority is a cost.
			L = h - λ' * F - γ' * G
			∇L = Symbolics.gradient(L, z)
			F̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L
						# G .- σ
						# σ .* γ .- ϵ
						F
					],
				),
			)
			G̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
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
				return (; F = [F̃; G .- σ_g; ϵ - γ' * G], G = G̃, z = Vector{symbolic_type}([z; λ; γ; σ_g]))
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
	F = Vector{symbolic_type}(
		filter!(!isnothing,
			vcat(
				filter!(!iszero, flattened_F),
			),
		),
	)

	# Pack all variables together.
	z = Vector{symbolic_type}(
		vcat(x, s, Σ, Λ, Γ),
	)
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
		inequality_constraint_dual_dims,
	)
end

"""[PATH Solver] Build the reduced KKT system as a `ParametricMCP`.
	- Shared constraints are not modeled separately; they are embedded in each player's constraints.
	- No interior-point slack variables ('σ') are introduced for inequality constraints.
	- The relaxation parameter `ϵ` is stored in `θ[end]`.
	- Relaxed complementarity is represented by the inequality `γ .* g - ϵ ≤ 0`.
	- Stores inequality duals in `y` instead of `z`
	- Preference constraints are stored explicitly by player and level, then flattened locally when needed.
"""

function generate_mcp_reduced_kkt_system(
	goop::ParametricGOOP;
	backend = SymbolicTracingUtils.SymbolicsBackend(),
	drop_higher_order_terms = false,
)
	# Symbolic variables for all primals, parameters, and duals for shared constraints.
	x =
		SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
		to_blockvector(goop.primal_dims)
	θ =
		SymbolicTracingUtils.make_variables(backend, :θ, sum(vcat(goop.parameter_dims, [1]))) |>
		to_blockvector(vcat(goop.parameter_dims, [1]))
	ϵ = θ[sum(goop.parameter_dims)+1] # Store the relaxation parameter as the last element of θ.

	symbolic_type = eltype(x)

	flatten_level_vectors(level_vectors, levels) =
		isempty(levels) ? symbolic_type[] :
		Vector{symbolic_type}(vcat((level_vectors[level] for level in levels)...))

	# Keep track of all the preference (s) we create.
	s = symbolic_type[]

	# Keep track of all equality constraint duals (λ) that we create.
	Λ = symbolic_type[]
	Φ = symbolic_type[] # 10/25: store duals for complementarity slackness

	# Keep track of all inequality constraint duals (γ) that we create.
	Γ = symbolic_type[]
	Γ_by_player_level = [
		[symbolic_type[] for _ in 1:length(goop.preferences[player])]
		for player in 1:goop.num_players
	]
	# Store each prioritized level's own preference constraints Pₖ = hₖ + sₖ and
	# its own preference duals γₚ,ₖ. Non-prioritized levels intentionally remain
	# empty vectors, so lower-level preference complementarity can be built level by level.
	Γₚ_by_player_level = [
		[symbolic_type[] for _ in 1:length(goop.preferences[player])]
		for player in 1:goop.num_players
	]
	preference_constraints_by_player_level = [
		[symbolic_type[] for _ in 1:length(goop.preferences[player])]
		for player in 1:goop.num_players
	]
	preference_slacks_by_player_level = [
		[symbolic_type[] for _ in 1:length(goop.preferences[player])]
		for player in 1:goop.num_players
	]

	# Keep track of all lower level policy constraint duals (ψ) that we create.
	Ψ = symbolic_type[]

	# Recursive function to construct a player's KKT conditions.
	function construct_player_kkt_conditions(
		preferences,
		is_prioritized_constraint;
		player,
	)
		num_levels = length(goop.preferences[player]) # Kⁱ
		@assert length(preferences) == length(is_prioritized_constraint)
		level = 1 + num_levels - length(preferences)

		f = isnothing(goop.equality_constraints[player]) ? nothing :
			goop.equality_constraints[player](x, θ)
		g = isnothing(goop.inequality_constraints[player]) ? nothing :
			goop.inequality_constraints[player](x, θ)

		# Base case is the inner-most layer.
		if length(preferences) == 1

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
			Γ_by_player_level[player][level] = γ

			h = only(preferences)(x, θ)

			if only(is_prioritized_constraint)
				# Highest priority is a constraint.
				preference_slack = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("s_$(player)_$(level)"),
					length(h),
				)
				push!(s, preference_slack...)
				preference_slacks_by_player_level[player][level] = preference_slack

				γₚ = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("γₚ_$(player)_$(level)"),
					length(h),
				)
				push!(Γ, γₚ...)
				Γₚ_by_player_level[player][level] = γₚ

				# Form Lagrangian at this stage.
				L =
					sum(preference_slack .^ 2) - γₚ' * (h .+ preference_slack) -
					(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)

				vars = vcat(x[Block(player)], preference_slack)
				∇L, π_terms = if drop_higher_order_terms
					lagrangian_terms = QuasiLagrangianTerm[]
					_push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack .^ 2))
					_push_quasi_lagrangian_term!(lagrangian_terms, h .+ preference_slack, γₚ)
					_push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
					_push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
					_quasi_gradient_from_terms(lagrangian_terms, vars)
				else
					Symbolics.gradient(L, vars), QuasiLagrangianTerm[]
				end

				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L
							isnothing(f) ? nothing : f
						],
					),
				)

				z = Vector{symbolic_type}(
					vcat(
						x[Block(player)],
						preference_slack,
						(isnothing(f) ? [] : λ),
					),
				)

				current_preference_constraints = Vector{symbolic_type}(h .+ preference_slack)
				preference_constraints_by_player_level[player][level] = current_preference_constraints
				Γₚ_by_player_level[player][level] = γₚ

				G, y = if level == 1
					if isnothing(g)
						current_preference_constraints, Vector{symbolic_type}(γₚ)
					else
						Vector{symbolic_type}([current_preference_constraints; g]), Vector{symbolic_type}([γₚ; γ])
					end
				else
					nothing, nothing
				end
				return (; F, z, G, y, π = ∇L, π_term_groups = [π_terms])
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
					Symbolics.gradient(L, x[Block(player)]), QuasiLagrangianTerm[]
				end

				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L
							isnothing(f) ? nothing : f
						],
					),
				)

				z = Vector{symbolic_type}(
					vcat(
						x[Block(player)],
						(isnothing(f) ? [] : λ),
					),
				)

				G, y = if level == 1
					if isnothing(g)
						symbolic_type[], symbolic_type[]
					else
						Vector{symbolic_type}(g), Vector{symbolic_type}(γ)
					end
				else
					nothing, nothing
				end
				return (; F, z, G, y, π = ∇L, π_term_groups = [π_terms])
			end
		end

		# Handle higher levels via tail recursion.
		(; F, z, G, y, π, π_term_groups) = construct_player_kkt_conditions(
			preferences[2:end],
			is_prioritized_constraint[2:end];
			player,
		)
		h = first(preferences)(x, θ)

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
		Γ_by_player_level[player][level] = γ

		ψ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("ψ_$(player)_$(level)"),
			length(π),
		)
		push!(Ψ, ψ...)

		lower_levels = (level+1):num_levels
		lower_prioritized_levels = [
			lower_level
			for lower_level in lower_levels
			if goop.is_prioritized_constraint[player][lower_level]
		]
		has_lower_preference_constraints = !isempty(lower_prioritized_levels)
		lower_preference_constraints =
			has_lower_preference_constraints ?
			flatten_level_vectors(preference_constraints_by_player_level[player], lower_prioritized_levels) :
			symbolic_type[]
		lower_preference_slacks =
			has_lower_preference_constraints ?
			flatten_level_vectors(preference_slacks_by_player_level[player], lower_prioritized_levels) :
			symbolic_type[]

		# Duals (γₚₗ) for lower level preference constraints (if exist) 
		γₚₗ = !has_lower_preference_constraints ? symbolic_type[] :
			  SymbolicTracingUtils.make_variables(
			backend,
			Symbol("γₚₗ_$(player)_$(level)"),
			length(lower_preference_constraints),
		)
		push!(Γ, γₚₗ...)

		# Duals for complementarity slackness for lower levels (w.r.t. g ≥ 0)
		ϕ = isnothing(g) ? symbolic_type[] :
			SymbolicTracingUtils.make_variables(
			backend,
			Symbol("ϕ_$(player)_$(level)"),
			length(lower_levels), # ℓ = 1 to Kⁱ - k, use sum(complementarity) ≤ ϵ
		)
		push!(Φ, ϕ...)

		# Duals for complementarity slackness for lower levels (w.r.t. lower level preference constraints)
		ϕₚ = !has_lower_preference_constraints ? symbolic_type[] :
			 SymbolicTracingUtils.make_variables(
			backend,
			Symbol("ϕₚ_$(player)_$(level)"),
			length(lower_prioritized_levels),
		)
		push!(Φ, ϕₚ...)

		# Lower level γ, γₚ duals and lower level complmentarity constraints
		lower_level_Γ =
			isnothing(g) ? nothing :
			flatten_level_vectors(Γ_by_player_level[player], lower_levels)

		lower_level_complementarity =
			isnothing(g) ? nothing :
			Vector{symbolic_type}([
				ϵ - sum(g .* Γ_by_player_level[player][lower_level])
				for lower_level in lower_levels
			])

		lower_level_Γₚ =
			!has_lower_preference_constraints ? nothing :
			flatten_level_vectors(Γₚ_by_player_level[player], lower_prioritized_levels)

		lower_level_preference_complementarity =
			!has_lower_preference_constraints ? nothing :
			Vector{symbolic_type}([
				ϵ - sum(
					preference_constraints_by_player_level[player][lower_level] .*
					Γₚ_by_player_level[player][lower_level],
				)
				for lower_level in lower_prioritized_levels
			])


		if first(is_prioritized_constraint)
			# Highest priority is a constraint.
			preference_slack = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("s_$(player)_$(level)"),
				length(h),
			)
			push!(s, preference_slack...)
			preference_slacks_by_player_level[player][level] = preference_slack

			# Duals for current level preference constraints
			γₚ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₚ_$(player)_$(level)"),
				length(h),
			)
			push!(Γ, γₚ...)

			current_preference_constraints = Vector{symbolic_type}(h .+ preference_slack)
			preference_constraints_by_player_level[player][level] = Vector{symbolic_type}(
				vcat(current_preference_constraints, lower_preference_constraints),
			)
			Γₚ_by_player_level[player][level] = Vector{symbolic_type}(
				vcat(γₚ, γₚₗ),
			)

			# Form reduced Lagrangian at this stage.
			L =
				sum(preference_slack .^ 2) - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				γₚ' * current_preference_constraints -
				(!has_lower_preference_constraints ? 0 : γₚₗ' * lower_preference_constraints) -
				(isnothing(lower_level_complementarity) ? 0 : ϕ' * lower_level_complementarity) -
				(isnothing(lower_level_preference_complementarity) ? 0 : ϕₚ' * lower_level_preference_complementarity)

			vars = vcat(x[Block(player)], preference_slack, lower_preference_slacks)

			# TODO: Implement the above for QuasiGOOP
			∇L, π_terms = if drop_higher_order_terms
				lagrangian_terms = QuasiLagrangianTerm[]
				_push_quasi_lagrangian_term!(lagrangian_terms, sum(preference_slack .^ 2))
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					current_preference_constraints,
					γₚ,
				)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					has_lower_preference_constraints ? lower_preference_constraints : nothing,
					γₚₗ,
				)
				_append_quasi_policy_terms!(lagrangian_terms, π_term_groups, ψ)
				_push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
				_push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
				_push_quasi_lagrangian_term!(lagrangian_terms, fₛ, λ̃ₛ)
				_push_quasi_lagrangian_term!(lagrangian_terms, gₛ, γ̃ₛ)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					lower_level_complementarity,
					ϕ,
				)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					lower_level_preference_complementarity,
					ϕₚ,
				)
				_quasi_gradient_from_terms(lagrangian_terms, vars)
			else
				Symbolics.gradient(L, vars), QuasiLagrangianTerm[]
			end

			F̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L
						F
						isnothing(f) ? nothing : f # 0507 hard code 
					],
				),
			)
			z̃ = Vector{symbolic_type}(
				vcat(
					x[Block(player)],
					preference_slack,
					lower_preference_slacks,
					ψ,
					isnothing(f) ? [] : λ,
					isnothing(f) ? [] : z[(end-length(f)+1):end], # 0507 lower level λₖ
				),
			)
			G̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						lower_level_complementarity
						lower_level_preference_complementarity
						G
					],
				),
			)
			ỹ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						isnothing(lower_level_complementarity) ? nothing : ϕ
						isnothing(lower_level_preference_complementarity) ? nothing : ϕₚ
						y
					],
				),
			)
			level == 1 && begin
				G̃ = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							G̃
							isnothing(g) ? nothing : g
							current_preference_constraints
							!has_lower_preference_constraints ? nothing : lower_preference_constraints
							isnothing(lower_level_Γ) ? nothing : zeros(length(lower_level_Γ))
							isnothing(lower_level_Γₚ) ? nothing : zeros(length(lower_level_Γₚ))
						],
					),
				)
				ỹ = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							ỹ # contains ϕ and ϕₚ only
							isnothing(g) ? nothing : γ
							γₚ
							!has_lower_preference_constraints ? nothing : γₚₗ
							isnothing(lower_level_Γ) ? nothing : lower_level_Γ
							isnothing(lower_level_Γₚ) ? nothing : lower_level_Γₚ
						],
					),
				)
			end
			return (; F = F̃, z = z̃, G = G̃, y = ỹ, π = vcat(∇L, π), π_term_groups = vcat([π_terms], π_term_groups))
		else
			@assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
			# Current priority is a cost. 		

			L =
				h - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				(!has_lower_preference_constraints ? 0 : γₚₗ' * lower_preference_constraints) -
				(isnothing(lower_level_complementarity) ? 0 : ϕ' * lower_level_complementarity) -
				(isnothing(lower_level_preference_complementarity) ? 0 : ϕₚ' * lower_level_preference_complementarity)

			vars = vcat(x[Block(player)], lower_preference_slacks)


			# TODO: Implement the above for QuasiGOOP
			∇L, π_terms = if drop_higher_order_terms
				lagrangian_terms = QuasiLagrangianTerm[]
				_push_quasi_lagrangian_term!(lagrangian_terms, h)
				_append_quasi_policy_terms!(lagrangian_terms, π_term_groups, ψ)
				_push_quasi_lagrangian_term!(lagrangian_terms, f, λ)
				_push_quasi_lagrangian_term!(lagrangian_terms, g, γ)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					has_lower_preference_constraints ? lower_preference_constraints : nothing,
					γₚₗ,
				)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					lower_level_complementarity,
					ϕ,
				)
				_push_quasi_lagrangian_term!(
					lagrangian_terms,
					lower_level_preference_complementarity,
					ϕₚ,
				)
				_quasi_gradient_from_terms(lagrangian_terms, x[Block(player)])
			else
				Symbolics.gradient(L, vars), QuasiLagrangianTerm[]
			end
			F̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L
						F
						isnothing(f) ? nothing : f # 0507 hard code 
					],
				),
			)
			z̃ = Vector{symbolic_type}(
				vcat(
					x[Block(player)],
					lower_preference_slacks,
					ψ,
					isnothing(f) ? [] : λ,
					isnothing(f) ? [] : z[(end-length(f)+1):end], # 0507 lower level λₖ
				),
			)
			G̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						lower_level_complementarity
						lower_level_preference_complementarity
						G
					],
				),
			)
			ỹ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						isnothing(lower_level_complementarity) ? nothing : ϕ
						isnothing(lower_level_preference_complementarity) ? nothing : ϕₚ
						y
					],
				),
			)
			level == 1 && begin
				G̃ = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							G̃
							isnothing(g) ? nothing : g
							!has_lower_preference_constraints ? nothing : lower_preference_constraints
							isnothing(lower_level_Γ) ? nothing : zeros(length(lower_level_Γ))
							isnothing(lower_level_Γₚ) ? nothing : zeros(length(lower_level_Γₚ))
						],
					),
				)
				ỹ = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							ỹ # contains only ϕ and ϕₚ
							isnothing(g) ? nothing : γ
							!has_lower_preference_constraints ? nothing : γₚₗ
							isnothing(lower_level_Γ) ? nothing : lower_level_Γ
							isnothing(lower_level_Γₚ) ? nothing : lower_level_Γₚ
						],
					),
				)
			end
			return (; F = F̃, z = z̃, G = G̃, y = ỹ, π = vcat(∇L, π), π_term_groups = vcat([π_terms], π_term_groups))
		end
	end

	# Recursively generate the rest of the KKT conditions for each player.
	"""
	F_mcp = [∇L₁; ... ; ∇Lₖ; f] = F which needs 0 rows to be appended to match the size of z_mcp
	---------------------------------------------------------------------------------------------------------------
	G_mcp = [ϵ - sum(complementarity),g, h + s,... h + s, 0]
	y_mcp = [Φ; Γ; Γₚ] where (Φ, Γ, Γₚ) are inequality duals bounded by (0, ∞)
		"""
	stacked_kkt = let
		kkt_per_player = map(1:(goop.num_players)) do player
			(; F, z, G, y) = construct_player_kkt_conditions(
				goop.preferences[player],
				goop.is_prioritized_constraint[player];
				player,
			)
			# z contains primals, equality and stationarity duals. Should be larger than F = 0.
			# y contains all inequality duals (∈ [0, ∞)) in alignment with G ≥ 0.
			# Main.@infiltrate
			@assert length(G) == length(y) "Mismatch between number of inequality constraints and inequality duals for player $(player): $(length(G)) vs $(length(y))"
			num_zero_rows = length(z) - length(F)
			@assert num_zero_rows >= 0 "F has more rows than z for player $(player): $(length(F)) vs $(length(z))"
			F = Vector{symbolic_type}(vcat(F, zeros(num_zero_rows)))
			# Main.@infiltrate
			# Construct MCP by combining F and G
			F_mcp = Vector{symbolic_type}(vcat(F, G))
			z_mcp = Vector{symbolic_type}(vcat(z, y))
			z̅_mcp = vcat(fill(Inf, length(F)), fill(Inf, length(G)))
			z̲_mcp = vcat(fill(-Inf, length(F)), fill(0.0, length(G))) # G ≥ 0 constraints	

			(; F_mcp, z_mcp, z̅_mcp, z̲_mcp)
		end

		F_mcp = Vector{symbolic_type}(vcat((pair.F_mcp for pair in kkt_per_player)...))
		z = Vector{symbolic_type}(vcat((pair.z_mcp for pair in kkt_per_player)...))
		z̅ = vcat((pair.z̅_mcp for pair in kkt_per_player)...)
		z̲ = vcat((pair.z̲_mcp for pair in kkt_per_player)...)

		# Permute so all primal rows come first.
		z_lengths = length.(getfield.(kkt_per_player, :z_mcp))
		offsets = cumsum(vcat(0, z_lengths[1:(end-1)]))
		primal_idx = mapreduce(vcat, 1:goop.num_players) do player
			(1+offsets[player]):(goop.primal_dims[player]+offsets[player])
		end
		# TODO: check that the bounds are properly assigned to primal and dual variables. 
		(;
			F_mcp = vcat(F_mcp[primal_idx], F_mcp[Not(primal_idx)]),
			z = vcat(z[primal_idx], z[Not(primal_idx)]),
			z̅ = vcat(z̅[primal_idx], z̅[Not(primal_idx)]),
			z̲ = vcat(z̲[primal_idx], z̲[Not(primal_idx)]),
		)
	end

	# Collect MCP components from the stacked KKT conditions.
	F_mcp = stacked_kkt.F_mcp
	z = stacked_kkt.z
	z̅ = stacked_kkt.z̅
	z̲ = stacked_kkt.z̲

	# Build and return ParametricMCP.
	θ = Vector{symbolic_type}(θ)
	@assert length(F_mcp) == length(z) "Reduced MCP is not square: length(F_mcp)=$(length(F_mcp)), length(z)=$(length(z))"
	@assert length(z̲) == length(z) "Reduced MCP lower-bound length mismatch: length(z̲)=$(length(z̲)), length(z)=$(length(z))"
	@assert length(z̅) == length(z) "Reduced MCP upper-bound length mismatch: length(z̅)=$(length(z̅)), length(z)=$(length(z))"

	ParametricMCP(F_mcp, z, θ, z̲, z̅; compute_sensitivities = false)
end

"""[PATH Solver] Build the complete KKT system as a `ParametricMCP`.
	- Shared constraints are not modeled separately; they are embedded in each player's constraints.
	- No interior-point slack variables ('σ') are introduced for inequality constraints.
	- The relaxation parameter `ϵ` is stored in `θ[end]`.
	- Relaxed complementarity is represented by the inequality `γ .* g - ϵ ≤ 0`.
"""

function generate_mcp_complete_kkt_system(
	goop::ParametricGOOP;
	backend = SymbolicTracingUtils.SymbolicsBackend(),
)
	# Symbolic variables for all primals, parameters, and duals.
	x =
		SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
		to_blockvector(goop.primal_dims)
	θ =
		SymbolicTracingUtils.make_variables(backend, :θ, sum(vcat(goop.parameter_dims, [1]))) |>
		to_blockvector(vcat(goop.parameter_dims, [1]))
	ϵ = θ[sum(goop.parameter_dims)+1] # Store the relaxation parameter as the last element of θ.

	symbolic_type = eltype(x)

	# Keep track of all the preference (s) we create.
	s = symbolic_type[]

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

		f = isnothing(goop.equality_constraints[player]) ? nothing :
			goop.equality_constraints[player](x, θ)
		g = isnothing(goop.inequality_constraints[player]) ? nothing :
			goop.inequality_constraints[player](x, θ)

		# Base case is the inner-most layer.
		if length(preferences) == 1

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

			h = only(preferences)(x, θ)

			if only(is_prioritized_constraint)
				# Highest priority is a constraint.
				preference_slack = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("s_$(player)_$(level)"),
					length(h),
				)
				push!(s, preference_slack...)

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
				# L =
				# 	sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
				# 	(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
				L =
					sum(preference_slack .^ 2) - γₚ' * (h .+ preference_slack) -
					(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)

				∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))

				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L
							isnothing(f) ? nothing : f
						],
					),
				)
				G = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							h .+ preference_slack
							γₚ
							ϵ - γₚ' * (h .+ preference_slack)
							# preference_slack
							# μₛ
							# ϵ - preference_slack .* μₛ
							isnothing(g) ? nothing : g
							isnothing(g) ? nothing : γ
							isnothing(g) ? nothing : ϵ - γ' * g
						],
					),
				)
				z = Vector{symbolic_type}(
					vcat(
						x[Block(player)],
						preference_slack,
						(isnothing(f) ? [] : λ),
						γₚ,
						(isnothing(g) ? [] : γ),
					),
				)
				level == 1 && return (;
					F,
					G = Vector{symbolic_type}(
						isnothing(g) ? h .+ preference_slack : [h .+ preference_slack; g],
					),
					z,
				)
				return (; F, G, z)
			else
				@assert length(h) == 1 "Expected a single preference function at the base level, but got $(length(h))"
				# Highest priority is a cost. 
				L = h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
				∇L = Symbolics.gradient(L, x[Block(player)])
				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L
							isnothing(f) ? nothing : f
						],
					),
				)
				G = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							isnothing(g) ? nothing : g
							isnothing(g) ? nothing : γ
							isnothing(g) ? nothing : ϵ - γ' * g
						],
					),
				)
				z = Vector{symbolic_type}(
					vcat(
						x[Block(player)],
						(isnothing(f) ? [] : λ),
						(isnothing(g) ? [] : γ),
					),
				)
				level == 1 && return (;
					F,
					G = isnothing(g) ? symbolic_type[] : Vector{symbolic_type}(g),
					z,
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

		if first(is_prioritized_constraint)
			# Highest priority is a constraint.
			preference_slack = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("s_$(player)_$(level)"),
				length(h),
			)
			push!(s, preference_slack...)

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
			L = sum(preference_slack .^ 2) - γₚ' * (h .+ preference_slack) - λ' * F - γ' * G
			∇L = Symbolics.gradient(L, vcat(z, preference_slack))

			F̃ = Vector{symbolic_type}([
				∇L
				F
			])
			G̃ = Vector{symbolic_type}([
				h .+ preference_slack # h(x) + s ≥ 0
				γₚ
				ϵ - γₚ' * (h .+ preference_slack)
				# preference_slack 		# s ≥ 0
				# μₛ
				# ϵ - preference_slack .* μₛ
				G
				γ
				ϵ - γ' * G
			])
			level == 1 && return (;
				F = F̃,
				G = [h .+ preference_slack; G],
				z = [z; preference_slack; λ; γₚ; γ],
			)
			return (; F = F̃, G = G̃, z = [z; preference_slack; λ; γₚ; γ])
		else
			@assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
			# Current priority is a cost.
			L = h - λ' * F - γ' * G
			∇L = Symbolics.gradient(L, z)
			F̃ = Vector{symbolic_type}([
				∇L
				F
			])
			G̃ = Vector{symbolic_type}([
				γ
				G
				ϵ - γ' * G
			])
			level == 1 && return (;
				F = F̃,
				G = G,
				z = [z; λ; γ],
			)
			return (; F = F̃, G = G̃, z = [z; λ; γ])
		end
	end

	# Recursively generate and stack player-wise KKT conditions.
	stacked_kkt = let
		kkt_per_player = map(1:goop.num_players) do player
			(; F, G, z) = construct_player_kkt_conditions(
				goop.preferences[player],
				goop.is_prioritized_constraint[player];
				player,
			)
			F_mcp = Vector{symbolic_type}(vcat(F, G))
			z̅ = vcat(fill(Inf, length(F)), fill(Inf, length(G)))
			z̲ = vcat(fill(-Inf, length(F)), fill(0.0, length(G)))

			(; F_mcp, z, z̅, z̲)
		end

		F_mcp = Vector{symbolic_type}(vcat((pair.F_mcp for pair in kkt_per_player)...))
		z = Vector{symbolic_type}(vcat((pair.z for pair in kkt_per_player)...))
		z̅ = vcat((pair.z̅ for pair in kkt_per_player)...)
		z̲ = vcat((pair.z̲ for pair in kkt_per_player)...)

		# Permute so all primal rows come first.
		z_lengths = length.(getfield.(kkt_per_player, :z))
		offsets = cumsum(vcat(0, z_lengths[1:(end-1)]))
		primal_idx = mapreduce(vcat, 1:goop.num_players) do player
			(1+offsets[player]):(goop.primal_dims[player]+offsets[player])
		end

		(;
			F_mcp = vcat(F_mcp[primal_idx], F_mcp[Not(primal_idx)]),
			z = vcat(z[primal_idx], z[Not(primal_idx)]),
			z̅ = vcat(z̅[primal_idx], z̅[Not(primal_idx)]),
			z̲ = vcat(z̲[primal_idx], z̲[Not(primal_idx)]),
		)
	end

	# Collect MCP components from the stacked KKT conditions.
	F_mcp = stacked_kkt.F_mcp
	z = stacked_kkt.z
	z̅ = stacked_kkt.z̅
	z̲ = stacked_kkt.z̲

	# Build and return ParametricMCP.
	θ = Vector{symbolic_type}(θ)
	ParametricMCP(F_mcp, z, θ, z̲, z̅; compute_sensitivities = false)
end

# Helper functions
make_blocks(vec, b) = (@assert length(vec) % b == 0; BlockArray(vec, fill(b, length(vec) ÷ b)))

function unflatten_trajectory(z, state_dimension, control_dimension)
	Z = reshape(z, state_dimension + control_dimension, :)
	X = @view Z[1:state_dimension, :]
	U = @view Z[(state_dimension+1):end, :]
	xs = eachcol(X) .|> collect
	us = eachcol(U) .|> collect
	(; xs, us)
end
