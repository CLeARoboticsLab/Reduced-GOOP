Base.@kwdef struct ParametricGOOP{T1, T2, T3, T4, T5}
	"Vector of callable preference functions for each player, called via h(x, θ)."
	preferences::Vector{Vector{T1}}

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
	parameter_dims::Int
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
	primal_dims = [length(x)] #TODO: BlockArrays.blocksizes(x)
	parameter_dims = length(θ)
	equality_dims = map(equality_constraints) do f
		isnothing(f) ? 0 : length(f(x, θ))
	end
	inequality_dims = map(inequality_constraints) do g
		isnothing(g) ? 0 : length(g(x, θ))
	end
	shared_equality_dims =
		isnothing(shared_equality_constraint) ? 0 : shared_equality_constraint(x, θ)
	shared_inequality_dims =
		isnothing(shared_inequality_constraint) ? 0 : shared_inequality_constraint(x, θ)

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

"Construct the KKT system corresponding to a ParametricGOOP."
function generate_slacked_kkt_system(
	goop::ParametricGOOP;
	backend = SymbolicTracingUtils.SymbolicsBackend(),
)
	# NOTE FOR LATER: 
	# 0807: Do we need to account for previous preference slacks when we take gradient of Lagrangian function at upper levels?
	# 		Different relaxations for different preferences is already handled
	# 		x \argmin_{x̃, s} x̃ st ...

	# Symbolic variables for all primals, parameters, and duals for shared constraints.
	x =
		SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
		to_blockvector(goop.primal_dims)
	θ = SymbolicTracingUtils.make_variables(backend, :θ, goop.parameter_dims)
	ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))

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
	s = []
	Σ = []

	# Keep track of all equality constraint duals (λ) that we create.
	Λ = []

	# Keep track of all inequality constraint duals (γ) that we create.
	Γ = []

	# Keep track of all lower level policy constraint duals (ψ) that we create.
	Ψ = []

	# Recursive function to construct a player's KKT conditions.
	function construct_player_kkt_conditions(
		preferences,
		is_prioritized_constraint;
		player,
	)
		@assert length(preferences) == length(is_prioritized_constraint)
		level = 1 + length(goop.preferences[player]) - length(preferences)

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

		# Shared constraints are not supported at this point.
		if !isnothing(goop.shared_equality_constraint) || !isnothing(goop.shared_inequality_constraint)
			error("Shared constraints are not supported at this level.")
		end
		# # Shared constraints exist at every level. https://github.com/CLeARoboticsLab/Quasi-GOOP/issues/6
		# Option (1): Share the multipliers only at all players' innermost levels, but let successive outer levels have their own separate multipliers for all players.


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
				@assert false
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

				γₚ = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("γₚ_$(player)_$(level)"),
					length(h),
				)
				push!(Γ, γₚ...)

				L =
					sum(preference_slack) - γₚ' * (h .+ preference_slack) -
					(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
					(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)

				∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))
				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L
							h .+ preference_slack .- σₚ
							f
							(isnothing(g) ? g : g .- σ)
							(isnothing(g) ? nothing : σ .* γ .- ϵ)
							σₚ .* γₚ .- ϵ
						],
					),
				)

				return (; F, π = ∇L)
			else
				# Highest priority is a cost. 
				L = h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
					(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)
				∇L = Symbolics.gradient(L, x[Block(player)])
				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L
							f
							(isnothing(g) ? nothing : g .- σ)
							(isnothing(g) ? nothing : σ .* γ .- ϵ)
						],
					),
				)

				return (; F, π = ∇L)
			end
		end

		# Handle higher levels via tail recursion.
		(; F, π) = construct_player_kkt_conditions(
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

		if first(is_prioritized_constraint)
			@assert false
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

			γₚ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₚ_$(player)_$(level)"),
				length(h),
			)
			push!(Γ, γₚ...)

			λ̃ₛ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("λ̃ₛ_$(player)_$(level)"),
				goop.shared_equality_dims,
			)
			push!(Λ, λ̃ₛ...)

			γ̃ₛ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γ̃ₛ_$(player)_$(level)"),
				goop.shared_inequality_dims,
			)
			push!(Γ, γ̃ₛ...)

			# Form partial Lagrangian at this stage.
			L = sum(preference_slack) - γₚ' * (h .+ preference_slack) -
				- ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
			- (isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ)

			∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))
			F̃ = [
				∇L
				h .+ preference_slack .- σₚ
				σₚ .* γₚ .- ϵ
				(isnothing(g) ? nothing : σ .* γ .- ϵ)
				(isnothing(gₛ) ? nothing : σₛ .* γ̃ₛ .- ϵ) # Note: same slacks (not duals) for all levels
				F
			]

			return (; F = F̃, π = ∇L)
		else
			# Current priority is a cost.
			# ip_slack = SymbolicTracingUtils.make_variables(
			# 	backend,
			# 	Symbol("σ_$(player)_$(level)"),
			# 	goop.inequality_dims[player],
			# )
			# push!(Σ, ip_slack...)

			L = h - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ)
			∇L = Symbolics.gradient(L, x[Block(player)])
			F̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L
						#(isnothing(g) ? nothing : g - σ) # TODO: Isn't this repeated at other levels?
						(isnothing(g) ? nothing : σ .* γ .- ϵ)
						(isnothing(gₛ) ? nothing : σₛ .* γ̃ₛ .- ϵ)
						F
					],
				),
			)

			return (; F = F̃, π = vcat(∇L, π))
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

	# Filter out zeros
	F = Vector{symbolic_type}(
		filter!(!iszero && !isnothing,
			vcat(
				F_π_pair.F,
				fₛ,
				(isnothing(gₛ) ? nothing : gₛ .- σₛ),
				(isnothing(gₛ) ? nothing : σₛ .* γₛ .- ϵ),
			),
		),
	)

	# Pack all variables together.
	z = Vector{symbolic_type}(
		vcat(x, s, Σ, Λ, Γ, Ψ, λₛ, γₛ, σₛ),
	)
	
	idx = blockedrange(length.([x, s, Σ, Λ, Γ, Ψ, λₛ, γₛ, σₛ]))
	preference_slack_dims = idx[Block(2)]
	interior_point_slack_dims = vcat(idx[Block(3)], idx[Block(9)]) # Σ, σₛ
	inequality_constraint_dual_dims = vcat(idx[Block(5)], idx[Block(8)]) # Γ, γₛ

	GOOPKKTSystem(
		F,
		z,
		θ,
		ϵ,
		preference_slack_dims,
		interior_point_slack_dims,
		inequality_constraint_dual_dims,
	)

	# (; F, z, θ)
end
