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

"Construct the Reduced KKT system corresponding to a ParametricGOOP."
function generate_slacked_reduced_kkt_system(
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
	Γ_cs = symbolic_type[] # 10/25: store duals for complementarity slackness
	Γ_cs_shared = symbolic_type[] # store duals for complementarity slackness for shared constraints

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
		push!(Γ_cs, γ...) # 10/25

		γ̃ₛ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("γ̃ₛ_$(player)_$(level)"),
			goop.shared_inequality_dims,
		)
		push!(Γ, γ̃ₛ...)
		push!(Γ_cs_shared, γ̃ₛ...) # 10/25

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

				∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))

				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L .+ η * vcat(x[Block(player)], preference_slack)
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

				return (; F, π = ∇L)
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
							∇L .+ η * x[Block(player)]
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
		# push!(Γ_cs_shared, γ̃ₛ...) # 10/25

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
			
			# Form partial Lagrangian at this stage.
			blocked_Γ_cs = g === nothing ? nothing : make_blocks(Γ_cs, goop.inequality_dims[player])
			blocked_Γ_cs_shared = gₛ === nothing ? nothing : make_blocks(Γ_cs_shared, goop.shared_inequality_dims)
			# L =
			# 	sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
			# 	ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
			# 	(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) -
			# 	(isnothing(g) ? 0 : ϕ' * (repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)])) -
			# 	(isnothing(gₛ) ? 0 : ϕₛ' * (repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)]))

			L =
				sum(preference_slack.^2) - γₚ' * (h .+ preference_slack) -
				ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) -
				(isnothing(g) ? 0 : ϕ' * (repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)])) -
				(isnothing(gₛ) ? 0 : ϕₛ' * (repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)]))

			∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))

			F̃ = [
				∇L .+ η * vcat(x[Block(player)], preference_slack)
				h .+ preference_slack .- σₚ
				σₚ .* γₚ .- ϵ
				# preference_slack .- σₚₛ
				# σₚₛ .* μₛ .- ϵ
				(isnothing(g) ? nothing : σ .* γ .- ϵ)
				(isnothing(gₛ) ? nothing : σₛ .* γ̃ₛ .- ϵ) # Note: same slacks (not duals) for all levels
				F
			]

			return (; F = F̃, π = ∇L)
		else
			@assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
			# Current priority is a cost.
			# TODO: dual variables for preference inequalities
			blocked_Γ_cs = g === nothing ? nothing : make_blocks(Γ_cs, goop.inequality_dims[player])
			blocked_Γ_cs_shared = gₛ === nothing ? nothing : make_blocks(Γ_cs_shared, goop.shared_inequality_dims)
			L =
				h - ψ' * π - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
				(isnothing(fₛ) ? 0 : λ̃ₛ' * fₛ) - (isnothing(gₛ) ? 0 : γ̃ₛ' * gₛ) -
				(isnothing(g) ? 0 : ϕ' * (repeat(g, num_levels - level) .* blocked_Γ_cs[Block(level+1):Block(num_levels)])) -
				(isnothing(gₛ) ? 0 : ϕₛ' * (repeat(gₛ, num_levels - level) .* blocked_Γ_cs_shared[Block(level+1):Block(num_levels)]))

			∇L = Symbolics.gradient(L, x[Block(player)])
			F̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L .+ η * x[Block(player)]
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

			σ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("σ_$(player)_$(level)"),
				goop.inequality_dims[player],
			)
			push!(Σ, σ...)

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

				σₚ = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("σₚ_$(player)_$(level)"),
					length(h),
				)
				push!(Σ, σₚ...)

				σₚₛ = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("σₚₛ_$(player)_$(level)"),
					length(h),
				)
				push!(Σ, σₚₛ...)

				γₚ = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("γₚ_$(player)_$(level)"),
					length(h),
				)
				push!(Γ, γₚ...)

				μₛ = SymbolicTracingUtils.make_variables(
					backend,
					Symbol("μₛ_$(player)_$(level)"),
					length(h),
				)
				push!(Γ, μₛ...)

				L =
					sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
					(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g) -
					(isnothing(fₛ) ? 0 : λₛ' * fₛ) - (isnothing(gₛ) ? 0 : γₛ' * gₛ)

				∇L = Symbolics.gradient(L, vcat(x[Block(player)], preference_slack))

				F = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							∇L #.+ η * vcat(x[Block(player)], preference_slack)
							f
							fₛ
							h .+ preference_slack .- σₚ
							σₚ .* γₚ .- ϵ
							preference_slack .- σₚₛ
							σₚₛ .* μₛ .- ϵ
							isnothing(g) ? nothing : g .- σ
							isnothing(g) ? nothing : σ .* γ .- ϵ
							isnothing(gₛ) ? nothing : gₛ .- σₛ
							isnothing(gₛ) ? nothing : σₛ .* γₛ .- ϵ
						],
					),
				)
				G = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							h .+ preference_slack
							preference_slack
							γₚ
							μₛ
							isnothing(g) ? nothing : γ
							isnothing(gₛ) ? nothing : γₛ
							isnothing(g) ? nothing : g
							isnothing(gₛ) ? nothing : gₛ
						],
					),
				)
				z = Vector{symbolic_type}(
					vcat(
						x[Block(player)],
						preference_slack,
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
							fₛ
							isnothing(g) ? nothing : g .- σ
							isnothing(g) ? nothing : σ .* γ .- ϵ
							isnothing(gₛ) ? nothing : gₛ .- σₛ
							isnothing(gₛ) ? nothing : σₛ .* γₛ .- ϵ
						],
					),
				)
				G = Vector{symbolic_type}(
					filter!(
						!isnothing,
						[
							isnothing(g) ? nothing : γ
							isnothing(gₛ) ? nothing : γₛ
							isnothing(g) ? nothing : g
							isnothing(gₛ) ? nothing : gₛ
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

		σ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("σ_$(player)_$(level)"),
			length(G),
		)
		push!(Σ, σ...)

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

			σₚₛ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("σₚₛ_$(player)_$(level)"),
				length(h),
			)
			push!(Σ, σₚₛ...)

			γₚ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₚ_$(player)_$(level)"),
				length(h),
			)
			push!(Γ, γₚ...)

			μₛ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("μₛ_$(player)_$(level)"),
				length(h),
			)
			push!(Γ, μₛ...)

			# Form partial Lagrangian at this stage.
			L = sum(preference_slack) - γₚ' * (h .+ preference_slack) - μₛ' * preference_slack -
				λ' * F - γ' * G

			∇L = Symbolics.gradient(L, vcat(z, preference_slack))

			F̃ = [
				∇L #.+ η * vcat(z, preference_slack)
				h .+ preference_slack .- σₚ
				σₚ .* γₚ .- ϵ
				preference_slack .- σₚₛ
				σₚₛ .* μₛ .- ϵ
				(isnothing(g) ? nothing : G .- σ)
				(isnothing(gₛ) ? nothing : σ .* γ .- ϵ)
				F
			]
			G̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						h .+ preference_slack
						preference_slack
						γₚ
						μₛ
						isnothing(g) ? nothing : γ
						G
					],
				),
			)
			return (; F = F̃, G = G̃, z = [z; preference_slack; λ; γ])
		else
			@assert length(h) == 1 "Expected a single preference function at the level $(level), but got $(length(h))"
			# Current priority is a cost.
			L = h - λ' * F - γ' * G
			∇L = Symbolics.gradient(L, z)
			F̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L .+ η * z
						isnothing(g) ? nothing : G .- σ
						isnothing(g) ? nothing : σ .* γ .- ϵ
						F
					],
				),
			)
			G̃ = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						isnothing(g) ? nothing : γ
						G
					],
				),
			)
			return (; F = F̃, G = G̃, z = [z; λ; γ])
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

	# 

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

# Helper functions
make_blocks(vec, b) = (@assert length(vec) % b == 0; BlockArray(vec, fill(b, length(vec) ÷ b)))
