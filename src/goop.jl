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
	primal_dims = [length(x)] #BlockArrays.blocksizes(x)
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

	# Symbolic variables for all primals, parameters, and duals for shared constraints.
	x =
		SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims)) |>
		to_blockvector(goop.primal_dims)
	θ = SymbolicTracingUtils.make_variables(backend, :θ, goop.parameter_dims)
	λ̃ = SymbolicTracingUtils.make_variables(backend, :λ̃, goop.shared_equality_dims)
	γ̃ = SymbolicTracingUtils.make_variables(backend, :γ̃, goop.shared_inequality_dims)
	symbolic_type = eltype(x)

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

		# TODO: Shared constraints do not belong to recursion here
		# fₛ =
		# 	isnothing(goop.shared_equality_constraint) ? nothing :
		# 	goop.shared_equality_constraint(x, θ)
		# gₛ =
		# 	isnothing(goop.shared_inequality_constraint) ? nothing :
		# 	goop.shared_inequality_constraint(x, θ)

		# Base case is the inner-most layer.
		if length(preferences) == 1
			h = only(preferences)(x, θ)

			if only(is_prioritized_constraint)
				@assert false
				# Highest priority is a constraint.
				preference_slack = only(
					SymbolicTracingUtils.make_variables(
						backend,
						Symbol("s_$(player)_$(level)"),
						1, 
					),
				)
				push!(s, preference_slack)

				ip_slack = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("σ_$(player)_$(level)"),
					1 + goop.shared_inequality_dims,
				)
				push!(σ, ip_slack...)

				dual = only(
					SymbolicTracingUtils.make_variables(
						backend,
						Symbol("μ_$(player)_$(level)"),
						1,
					),
				)
				push!(μ, dual)

				L =
					preference_slack - dual * (h - preference_slack) -
					(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
				∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))
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

				return (; F, π = ∇L)
			else
				# Highest priority is a cost. 

				L = h - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
				∇L = Symbolics.gradient(L, x[Block(player)])
				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L
							f
							(isnothing(g) ? nothing : g - ip_slack)
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
			preference_slack = only(
				SymbolicTracingUtils.make_variables(
					backend,
					Symbol("s_$(player)_$(level)"),
					1,
				),
			)
			push!(s, preference_slack)

			ip_slack = only(
				SymbolicTracingUtils.make_variables(
					backend,
					Symbol("σ_$(player)_$(level)"),
					1,
				),
			)
			push!(σ, ip_slack)

			dual = only(
				SymbolicTracingUtils.make_variables(
					backend,
					Symbol("μ_$(player)_$(level)"),
					1,
				),
			)
			push!(μ, dual)

			# Form partial Lagrangian at this stage.
			L̃ = preference_slack - dual * (h - preference_slack)

			# Calculate derivative of lower level policy, treating lower multipliers
			# as implicit functions of top level variables.
			# TODO: Check this! Probably made a mistake.
			primals = vcat(x[Block(player)], preference_slack)
			# ∇π_player_primals = Symbolics.jacobian(π, primals)
			# ∇π_other_primals = Symbolics.jacobian(π, x[Block.(Not(player))])
			# ∇π_lower_duals = Symbolics.jacobian(π, vcat(λ̃, γ̃, γ, ψ))
			# ∇lower_duals_player_primals = -[∇π_other_primals

			∇L = Symbolics.gradient(L̃ - ψ' * π, primals) #- (∇π_primals' + \nabla∇π_lower_duals') * ψ̃
			F̃ = [
				∇L
				h - preference_slack - ip_slack
				F
			]

			return (; F = F̃, π = ∇L)
		else
			# Highest priority is a cost.
			L = h - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
			∇L = Symbolics.gradient(L, x[Block(player)])
			F̃ = [
				∇L
				F
			]

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
		filter(!iszero, F_π_pair.F),
	)

	# Pack all variables together.
	# z = vcat(x, s, Σ, Λ, Γ, Ψ, λ̃, γ̃,)

	z = Vector{symbolic_type}(
		vcat(x, Λ, Ψ),
	)
	idx = blockedrange(length.([x, s, Σ, Λ, Γ, Ψ, λ̃, γ̃,]))
	preference_slack_dims = idx[Block(2)]
	interior_point_slack_dims = idx[Block(3)]
	inequality_constraint_dual_dims = vcat(idx[Block(5)], idx[Block(7)])

	GOOPKKTSystem(
		F,
		z,
		θ,
		preference_slack_dims,
		interior_point_slack_dims,
		inequality_constraint_dual_dims,
	)

    # (; F, z, θ)
end
