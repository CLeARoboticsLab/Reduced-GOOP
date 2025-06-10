mutable struct Lagrangian_term{T <: Symbolics.Num}
	expr::Union{Vector{T}, T}
	duals::Union{Vector{T}, Nothing}
	deriv_order::Int
end

function Symbolics.gradient(f::Lagrangian_term, x::AbstractVector{<:Symbolics.Num})
	if f.deriv_order > 2 # 1: drop terms, if 2 or higher: keep terms
		return Lagrangian_term(zero.(x), f.duals, f.deriv_order + 1)
	end

	f.deriv_order += 1
	# Return new object after taking gradient
	expr = isnothing(f.duals) ? f.expr : -f.expr' * f.duals
	Lagrangian_term(Symbolics.gradient(expr, x), f.duals, f.deriv_order)
end

mutable struct Equalities_ii{T <: Lagrangian_term}
	stationarity::Vector{Union{Vector{T}, Nothing}}
	equalities::Vector{Vector{T}}
end

function Equalities_ii{T}(num_levels::Int) where {T <: Lagrangian_term}
	Equalities_ii{T}(
		[nothing for _ in 1:(num_levels+1)],
		[Vector{T}() for _ in 1:(num_levels+1)],
	)
end

function Equalities_ii(num_levels::Int, num_players::Int)
	[Equalities_ii{Lagrangian_term}(num_levels) for _ in 1:num_players]
end

struct ordered_preferences
	"Vector of callable preference functions"
	preferences::Vector{Vector{Function}}

	"Vector of boolean values indicating if the preference is a constraint"
	is_prioritized_constraint::Vector{Vector{Bool}}
end

"""
Function that constructs a `ParametricQuasiGOOP` object from callable functins of objectives, constraints, and preferences.
"""
function ParametricQuasiGOOP(;
	objectives,
	equality_constraints,
	inequality_constraints,
	preferences::ordered_preferences,
	shared_equality_constraints,
	shared_inequality_constraints,
	primal_dimensions,
	parameter_dimensions,
	equality_dimensions,
	inequality_dimensions,
)

	(;
		K_symbolic,
		z_symbolic,
		θ_symbolic,
		lower_bounds,
		upper_bounds,
		dims,
	) = QuasiGOOP_to_PDSyst(;
		objectives,
		equality_constraints,
		inequality_constraints,
		preferences,
		shared_equality_constraints,
		shared_inequality_constraints,
		primal_dimensions,
		parameter_dimensions,
		equality_dimensions,
		inequality_dimensions,
	)

	PrimalDualSys(
		K_symbolic,
		z_symbolic,
		θ_symbolic,
		lower_bounds,
		upper_bounds,
		dims,
	)
	# Main.@infiltrate
end

"Helper function to create components of the Primal-Dual system of equations from QuasiGOOP."
function QuasiGOOP_to_PDSyst(;
	objectives::Vector{<:Function},
	equality_constraints::Vector{<:Function},
	inequality_constraints::Vector{<:Function},
	preferences::ordered_preferences,
	shared_equality_constraints,
	shared_inequality_constraints,
	primal_dimensions::Vector{Int},
	parameter_dimensions::Vector{Int},
	equality_dimensions::Vector{Int},
	inequality_dimensions::Vector{Int},
)
	;
	println("Make a PrimalDualSys object from callable functions + implement approximations")
	backend = SymbolicTracingUtils.SymbolicsBackend()

	# Problem data
	ordered_priority_levels = eachindex(preferences.preferences[1]) # assume all players have the same number of preferences
	num_players = length(preferences.preferences)
	num_levels = length(ordered_priority_levels)

	primal_dimension_ii = 0
	equality_dimension_ii = copy(equality_dimensions)
	inequality_dimension_ii = copy(inequality_dimensions)

	# Account for annealing parameter, σ → 0
	augmented_parameter_dimension = sum(parameter_dimensions) + 1

	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	# Preallocate arrays of symbolic functions for each player
	private_inner_equality_constraints = Vector{Symbolics.Num}[]
	Lagrangian_terms = Lagrangian_term[]
	F_ii = Equalities_ii(num_levels, num_players)

	# Store dimensions
	private_primals = [[dim] for dim in primal_dimensions]

	# Store (callable) slacks
	private_slacks = Function[]

	start_idx = 1

	function set_up_level(priority_level, player_idx)
		# Do the following for preferences, not for topmost level objective
		# Step 1. Reformulation for prioritized constraints
		prioritized_constraints_ii =
			preferences.preferences[player_idx][priority_level] # fᵢ(x,θ) ≥ 0
		preference_slack_dimension_ii =
			length(prioritized_constraints_ii(dummy_primals, dummy_parameters))
		primal_dimension_ii += preference_slack_dimension_ii
		append!(private_primals[player_idx], preference_slack_dimension_ii)
		inequality_dimension_ii[player_idx] += 2preference_slack_dimension_ii # account for sᵢ ≥ 0

		# Step 2. Reformulate inequality constraints as equality constraints (via additional slacks)
		equality_dimension_ii[player_idx] += inequality_dimension_ii[player_idx]
		barrier_slacks_dimension_ii = copy(inequality_dimension_ii[player_idx])
		primal_dimension_ii += barrier_slacks_dimension_ii
		append!(private_primals[player_idx], barrier_slacks_dimension_ii)
		@assert sum(private_primals[player_idx]) == primal_dimension_ii

		#Main.@infiltrate

		# Step 3. Define symbolic variables for primals
		total_dimension =
			primal_dimension_ii +
			equality_dimension_ii[player_idx]

		z̃ = Symbolics.scalarize(
			only(Symbolics.@variables(z̃[start_idx:(total_dimension+start_idx-1)])),
		)
		z = BlockArray(
			z̃,
			[
				primal_dimension_ii,
				equality_dimension_ii[player_idx],
			],
		)
		θ̃ = SymbolicTracingUtils.make_variables(backend, :θ̃, augmented_parameter_dimension)
		θ = BlockArray(
			θ̃,
			vcat(
				parameter_dimensions,
				[1],
			),
		)
		σ = θ[end] # annealing parameter

		x = BlockArray(z[Block(1)], private_primals[player_idx])
		λ = z[Block(2)] # dual variables for equality constraints
		# Main.@infiltrate

		# Step 4. Define symbolic expression for objective and (equality) constraints
		# Step 4a. Define symbolic expression for objective and (equality) constraints
		preference_slacks_ii = blocks(x)[end-1] # These slacks are introduced before reformulation slacks; second-last block
		barrier_slacks_ii = blocks(x)[end] # last block
		objective_ii = sum(preference_slacks_ii) # or sum(preference_slacks_ii)
		println("objective_$(priority_level) for player $(player_idx) = ", objective_ii)
		println("barrier slacks at $(priority_level) for player $(player_idx) = ", barrier_slacks_ii)
		barrier_objective_ii = σ * sum(log.(barrier_slacks_ii))

		# Replace inequality constriants: fg(x,sₚ) = sₚ + g(x) ≥ 0, fg(x,sₚ) - sᵦ = 0
		auxillary_constraints = prioritized_constraints_ii(x, θ) .+ preference_slacks_ii

		if priority_level == first(ordered_priority_levels)
			push!(
				Lagrangian_terms,
				Lagrangian_term(
					equality_constraints[player_idx](x, θ), nothing, 0), # cₑ(z) = 0
			)
			push!(
				Lagrangian_terms,
				Lagrangian_term(
					vcat(
						auxillary_constraints, # h(z) + s̄ ≥ 0
						preference_slacks_ii, # s̄ ≥ 0
						inequality_constraints[player_idx](x, θ), # g(z) ≥ 0
					) .- barrier_slacks_ii, nothing, 0), # cᵢ(z, s̄) - s = 0
			)
		else
			# equalities from previous levels become equality constraints 
			for kk in 1:length(F_ii[player_idx].equalities[priority_level-1])
				let
					expr = F_ii[player_idx].equalities[priority_level-1][kk].expr
					push!(
						Lagrangian_terms,
						Lagrangian_term(expr, nothing, 0),
					)
				end
			end
			#subsequent levels also have additional equalities from preference function
			push!(
				Lagrangian_terms,
				Lagrangian_term(
					vcat(
						auxillary_constraints,
						preference_slacks_ii,
					) .- barrier_slacks_ii, nothing, 0), #fg
			)
		end
		append!(F_ii[player_idx].equalities[priority_level], Lagrangian_terms)
		empty!(Lagrangian_terms)
		inequality_dimension_ii[player_idx] = 0 # reset inequality dimension

		x_temp = let
			Symbolics.scalarize(
				only(Symbolics.@variables(z̃[1:total_dimension])),
			)
		end
		sum_slacks = Symbolics.build_function(
			sum(preference_slacks_ii),
			x_temp,
			θ,
			expression = Val{false},
		)
		push!(private_slacks, sum_slacks)


		# append!(F_ii[player_idx].equalities[priority_level], Lagrangian_terms)
		# empty!(Lagrangian_terms)

		# Step 5. Define symbolic expression for Lagrangian and stationarity conditions/constraints
		if priority_level == first(ordered_priority_levels)
			dims = [length(e.expr) for e in F_ii[player_idx].equalities[priority_level]] # level 1: [28, 64]
		else
			dims = let
				ns = [length(e.expr) for e in F_ii[player_idx].stationarity[priority_level-1]]
				@assert all(item -> item == ns[1], ns)
				ds = [length(e.expr) for e in F_ii[player_idx].equalities[priority_level]]
				vcat(ns[1], ds) # level 2:[110, 28, 64, 14], level 3: [223, 28, 64, 14, 56], level 4: [321, 28, 64, 14, 56]
			end
		end

		λ = BlockArray(λ, dims)

		# Build Lagrangian function using F_ii
		push!(
			Lagrangian_terms,
			Lagrangian_term(
				objective_ii - barrier_objective_ii, nothing, 0),
		)

		if priority_level > 1
			#Main.@infiltrate
			prev_st = F_ii[player_idx].stationarity[priority_level-1]
			@assert all([length(term.expr) for term in prev_st] .== length(λ[Block(1)]))
			for element in prev_st
				expr = element.expr
				deriv_order = element.deriv_order
				push!(
					Lagrangian_terms,
					Lagrangian_term(expr, λ[Block(1)], deriv_order), # λ[Block(1)] reserved for stationarity
				)
			end
		end
		eqs = F_ii[player_idx].equalities[priority_level]
		for (kk, eq) in enumerate(eqs)
			dual_idx = priority_level > 1 ? 1 + kk : kk
			deriv_order = priority_level > 1 ? eq.deriv_order : 0
			expr = eq.expr
			@assert length(expr) == length(λ[Block(dual_idx)])
			push!(
				Lagrangian_terms,
				Lagrangian_term(expr, λ[Block(dual_idx)], deriv_order),
			)
		end

		prune_zeros = false
		stationarity = build_and_filter_stationarity!(x, Lagrangian_terms; prune_zeros) # Lagrangian_terms updated in_place
		# # use perturbed KKT interpretation (from NW textbook) This didn't really work but let's come back to it later
		stationarity = let
			m = length(barrier_slacks_ii)
			for term in Lagrangian_terms
				# Main.@infiltrate
				@views term.expr[(end-m+1):end] .*= barrier_slacks_ii
			end
			stationarity[(end-m+1):end] .*= barrier_slacks_ii
			Symbolics.expand.(stationarity)
		end
		#Main.@infiltrate
		F_ii[player_idx].stationarity[priority_level] = copy(Lagrangian_terms)
		empty!(Lagrangian_terms)
		#Main.@infiltrate

		if priority_level < 4 # hardcoded for now since after level 3, quasiGOOP is significantly different
			check_stationarity!(
				priority_level,
				ordered_priority_levels,
				private_inner_equality_constraints, # `private_inner_equality_constraints` is updated in-place,
				equality_constraints,
				inequality_constraints,
				F_ii[player_idx],
				player_idx,
				x, θ,
				auxillary_constraints,
				preference_slacks_ii,
				barrier_slacks_ii,
				objective_ii,
				barrier_objective_ii,
				λ,
				stationarity;
				prune_zeros,
			)
		end

		#Main.@infiltrate

		# Update primal_dimension_ii with induced primal dimension
		induced_primal_dimension_ii = length(λ)
		primal_dimension_ii += induced_primal_dimension_ii
		append!(private_primals[player_idx], induced_primal_dimension_ii)

		# Update equality_dimension_ii
		stationarity_dimension = length(stationarity)
		prev_equality_dimension_ii = sum(length(e.expr) for e in F_ii[player_idx].equalities[priority_level])
		equality_dimension_ii[player_idx] = stationarity_dimension + prev_equality_dimension_ii
	end

	# Build KKT system for each priority level for each player's own problem
	for player in 1:num_players
		primal_dimension_ii = sum(private_primals[player])
		for priority_level in ordered_priority_levels
			set_up_level(priority_level, player)
		end
		start_idx += primal_dimension_ii
	end

	#Main.@infiltrate

	# Build the topmost level of the KKT system 
	#TODO: Check final_equality and final_stationarity
	# 1. Build equalitiy constraints for the topmost level 
	for player_idx in 1:num_players
		# Equality constraints from previous level
		for kk in 1:length(F_ii[player_idx].equalities[num_levels])
			let
				expr = F_ii[player_idx].equalities[num_levels][kk].expr
				push!(
					Lagrangian_terms,
					Lagrangian_term(expr, nothing, 0),
				)
			end
		end
		F_ii[player_idx].equalities[num_levels+1] = copy(Lagrangian_terms)
		empty!(Lagrangian_terms)
	end
	final_equality = [
		reduce(vcat, [t.expr for t in F_ii[player].equalities[num_levels+1]]) for player in 1:num_players
	]
	# 2. Build stationarity (of the previous level) for the topmost level
	final_stationarity = [
		mapreduce(
			t -> t.expr,
			.+,
			F_ii[player].stationarity[num_levels];
			init = zero.(F_ii[player].stationarity[num_levels][1].expr))
		for player in 1:num_players]
	#Main.@infiltrate
	@assert all(Symbolics.iszero, vcat(final_stationarity[1], final_equality[1]) .- private_inner_equality_constraints[1])
	@assert all(Symbolics.iszero, vcat(final_stationarity[2], final_equality[2]) .- private_inner_equality_constraints[2])

	# # Use perturbed KKT interpretation (from NW textbook) This didn't really work but let's come back to it later
	# for player in 1:num_players
	# 	private_inner_equality_constraints[player] = vcat(final_stationarity[player], final_equality[player])
	# end

	# Main.@infiltrate

	# 3. Shared inequality constraints
	shared_inequality_dimension = length(shared_inequality_constraints(dummy_primals, dummy_parameters))
	shared_equality_dimension = length(shared_equality_constraints(dummy_primals, dummy_parameters))

	# Build shared equality constraints from shared inequality constraints
	primal_dimension_ii = map(sum, private_primals)
	total_dimension =
		sum(primal_dimension_ii) +
		sum(equality_dimension_ii) +
		shared_inequality_dimension +
		shared_equality_dimension

	# Build symbolic variables for this MCP
	z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
	z = BlockArray(
		z̃,
		[
			sum(primal_dimension_ii),
			sum(equality_dimension_ii),
			shared_equality_dimension,
			shared_inequality_dimension,
		],
	)
	θ̃ = Symbolics.scalarize(
		only(Symbolics.@variables(θ̃[1:augmented_parameter_dimension])),
	)
	θ = BlockArray(θ̃, vcat(parameter_dimensions, [1]))

	x = BlockArray(z[Block(1)], primal_dimension_ii)
	λ = BlockArray(z[Block(2)], equality_dimension_ii)
	λₛ = z[Block(3)]
	μₛ = z[Block(4)]

	# Main.@infiltrate
	trajectory_primals = [
		i > 1 ? z[(1:private_primals[i][1]) .+ sum(primal_dimension_ii[1:(i-1)])] :
		z[1:private_primals[i][1]] for i in 1:num_players
	]
	trajectory_x = BlockArray(
		vcat(trajectory_primals...),
		[private_primals[i][1] for i in 1:num_players],
	)
	fs = map(f -> f(trajectory_x, θ), objectives)
	gs = private_inner_equality_constraints # contains MPCC (nested) constraints
	g̃ = shared_equality_constraints(trajectory_x, θ)
	h̃ = shared_inequality_constraints(trajectory_x, θ)

	# Build Lagrangian for all players.
	Ls = map(zip(1:num_players, fs, gs)) do (i, f, g)
		f - λ[Block(i)]' * g - λₛ' * g̃ - μₛ' * h̃
	end

	# Build F = [∇ₓLs, gs, g̃,].
	∇ₓLs = map(zip(Ls, blocks(x))) do (L, xᵢ) # TODO: drop terms at topmost level too? 
		# ∇ₓL = Symbolics.gradient(L, xᵢ)
		# mask = .!Symbolics.iszero.(∇ₓL)
		# ∇ₓL[mask] # filter out zero rows
		Symbolics.gradient(L, xᵢ)
	end

	symbolic_type = eltype(x)
	K = Vector{symbolic_type}(
		filter!(
			!isnothing,
			[
				reduce(vcat, ∇ₓLs)
				reduce(vcat, gs)
				g̃
				h̃
			],
		),
	)

	# Set lower and upper bounds for z. 
	z̲ = [
		fill(-Inf, sum(primal_dimension_ii)) #sum(primal_dimension_ii)
		fill(-Inf, sum(equality_dimension_ii))
		fill(-Inf, shared_equality_dimension)
		fill(0, shared_inequality_dimension)
	]
	z̅ = [
		fill(Inf, sum(primal_dimension_ii)) #sum(length(∇ₓL) for ∇ₓL in ∇ₓLs)
		fill(Inf, sum(equality_dimension_ii))
		fill(Inf, shared_equality_dimension)
		fill(Inf, shared_inequality_dimension)
	]

	dims = (;
		x = only(blocksizes(x)),
		θ = only(blocksizes(θ)),
		λ = only(blocksizes(λ)),
		λₛ = only(blocksizes(λₛ)),
		μₛ = only(blocksizes(μₛ)),
		private_primals,
	)

	(;
		K_symbolic = collect(K),
		z_symbolic = collect(z),
		θ_symbolic = collect(θ),
		lower_bounds = z̲,
		upper_bounds = z̅,
		dims,
	)
end

"Helper function to filter zeros out of the stationarity vector."
function build_and_filter_stationarity!(x, Lagrangian_terms; prune_zeros = true)
	n = length(x)
	stationarity = zero.(x)
	mask = falses(n) # mask[j] = true iff row j was ever non‑zero

	for (i, term) in enumerate(Lagrangian_terms)
		# Main.@infiltrate
		new_term = Symbolics.gradient(term, collect(x))
		Lagrangian_terms[i] = new_term
		stationarity .+= new_term.expr
		println("....new_term.deriv_order = ", new_term.deriv_order)
		if new_term.deriv_order > 2
			all(Symbolics.iszero.(new_term.expr))
		end
		# record which positions are non‑zero in this term
		mask .|= .!Symbolics.iszero.(new_term.expr) # mask[j] = mask[j] || (!Symbolics.iszero(expr[j])), elementwise OR and assign
	end

	# drop all rows that never lit up
	if prune_zeros
		if any(.!mask)
			stationarity = stationarity[mask]
			for t in Lagrangian_terms
				t.expr = t.expr[mask]
			end
		end
	end

	stationarity
end

"""
Compute and verify stationarity at the given `priority_level (<4)` for `player_idx`.  
- Appends the appropriate equality (and slack) constraints into `private_inner_equality_constraints`  
- Computes the gradient of the Lagrangian and compares it to `stationarity`  
"""
# function check_stationarity!(
# 	priority_level::Int,
# 	ordered_priority_levels::AbstractVector{<:Int},
# 	private_inner_equality_constraints::Vector{Vector{Symbolics.Num}},
# 	equality_constraints,
# 	inequality_constraints::Vector{Function},
# 	F_ii,
# 	player_idx::Int,
# 	x,
# 	θ,
# 	auxillary_constraints::Vector{Symbolics.Num},
# 	preference_slacks_ii::Vector{Symbolics.Num},
# 	barrier_slacks_ii::Vector{Symbolics.Num},
# 	objective_ii::Symbolics.Num,
# 	barrier_objective_ii::Symbolics.Num,
# 	λ,
# 	stationarity;
# 	prune_zeros = true,
# )::Bool

# 	if priority_level == first(ordered_priority_levels)
# 		# — base level: include all equality + slacks —
# 		push!(
# 			private_inner_equality_constraints,
# 			equality_constraints[player_idx](x, θ),
# 		)
# 		append!(
# 			private_inner_equality_constraints[player_idx],
# 			vcat(
# 				auxillary_constraints,
# 				preference_slacks_ii,
# 				inequality_constraints[player_idx](x, θ),
# 			) .- barrier_slacks_ii,
# 		)

# 		g_ii = private_inner_equality_constraints[player_idx]
# 		L = objective_ii - barrier_objective_ii - λ' * g_ii
# 		computed = Symbolics.gradient(L, x)

# 		computed = let
# 			m = length(barrier_slacks_ii)
# 			computed[(end-m+1):end] .*= barrier_slacks_ii
# 			Symbolics.expand.(computed)
# 		end

# 		match = all(isequal.(stationarity, computed))
# 		println("Check stationarity at level $priority_level = $match")

# 		Main.@infiltrate

# 		# prepend for the next level
# 		pushfirst!(private_inner_equality_constraints[player_idx], stationarity...)

# 	elseif priority_level > 1
# 		# — higher levels: only the slack equalities —
# 		append!(
# 			private_inner_equality_constraints[player_idx],
# 			vcat(auxillary_constraints, preference_slacks_ii) .- barrier_slacks_ii,
# 		)

# 		g_ii = private_inner_equality_constraints[player_idx]
# 		L = objective_ii - barrier_objective_ii - λ' * g_ii
# 		computed = Symbolics.gradient(L, x)

# 		computed = let
# 			m = length(barrier_slacks_ii)
# 			computed[(end-m+1):end] .*= barrier_slacks_ii
# 			Symbolics.expand.(computed)
# 		end

# 		match = all(isequal.(stationarity, computed))
# 		println("Check stationarity at level $priority_level = $match")

# 		# mask out always-zero entries
# 		if prune_zeros
# 			mask = .! Symbolics.iszero.(computed)
# 			computed = computed[mask]
# 		end

# 		if length(stationarity) == length(computed)
# 			match = all(isequal.(stationarity, computed))
# 		else
# 			println(
# 				"Warning: cannot compare stationarity (length=$(length(stationarity))) ",
# 				"with computed (length=$(length(computed))). ",
# 				"Setting match = false.",
# 			)
# 			match = false
# 		end

# 		# remove previous stationarity constraints
# 		num_prev = length(F_ii.stationarity[priority_level-1][1].expr)
# 		deleteat!(private_inner_equality_constraints[player_idx], 1:num_prev)

# 		# prepend the new stationarity
# 		pushfirst!(private_inner_equality_constraints[player_idx], stationarity...)

# 	else
# 		error("Invalid priority_level: $priority_level")
# 	end

# 	return match
# end

function check_stationarity!(
    priority_level::Int,
    ordered_priority_levels::AbstractVector{<:Int},
    private_inner_equality_constraints::Vector{Vector{Symbolics.Num}},
    equality_constraints,
    inequality_constraints::Vector{Function},
    F_ii,
    player_idx::Int,
    x,
    θ,
    auxillary_constraints::Vector{Symbolics.Num},
    preference_slacks_ii::Vector{Symbolics.Num},
    barrier_slacks_ii::Vector{Symbolics.Num},
    objective_ii::Symbolics.Num,
    barrier_objective_ii::Symbolics.Num,
    λ,
    stationarity;
    prune_zeros = true,
)::Bool
    # Determine if at base or higher level
    is_base = (priority_level == first(ordered_priority_levels))
    is_higher = (priority_level > 1)

    if is_base
        # Base level: include all equality + slacks
        push!(
            private_inner_equality_constraints,
            equality_constraints[player_idx](x, θ)
        )
        append!(
            private_inner_equality_constraints[player_idx],
            vcat(
                auxillary_constraints,
                preference_slacks_ii,
                inequality_constraints[player_idx](x, θ)
            ) .- barrier_slacks_ii
        )
    elseif is_higher
        # Higher levels: only the slack equalities
        append!(
            private_inner_equality_constraints[player_idx],
            vcat(auxillary_constraints, preference_slacks_ii) .- barrier_slacks_ii
        )
    else
        error("Invalid priority_level: $priority_level")
    end

    # Common computation of gradient and checking
    g_ii = private_inner_equality_constraints[player_idx]
    L = objective_ii - barrier_objective_ii - λ' * g_ii
    computed = Symbolics.gradient(L, x)

    # Scale and expand barrier terms
    computed = let m = length(barrier_slacks_ii)
        computed[(end-m+1):end] .*= barrier_slacks_ii
        Symbolics.expand.(computed)
    end

    # Initial match for both levels
    match = all(isequal.(stationarity, computed))
    println("Check stationarity at level $priority_level = $match")

    if is_higher && prune_zeros
        # Prune zeros for higher levels
        mask = .!Symbolics.iszero.(computed)
        computed = computed[mask]
    end

    if is_higher
        # Remove previous stationarity constraints
        num_prev = length(F_ii.stationarity[priority_level-1][1].expr)
        deleteat!(private_inner_equality_constraints[player_idx], 1:num_prev)
    end

    # Prepend stationarity for next level only
    pushfirst!(private_inner_equality_constraints[player_idx], stationarity...)

	# Write stationarity elements to a text file
    open("stationarity_level$(priority_level).txt", "w") do io
        println(io, "Stationarity at level $priority_level:")
        for elem in stationarity
            println(io, elem)
        end
    end

    return match
end


"Solve GOOP." 
function solve(
	mcp::PrimalDualSys,
	θ,
	σ, κ, max_iterations, tolerance;
	solver_type = InteriorPoint(),
	kwargs...,
)
	# Main.@infiltrate
	relaxations = σ * κ .^ (0:max_iterations)
	warmstart_sol = nothing
	kkt_error = Inf
	x, y, s = 0, 0, 0
	status = nothing
	ii = 1
	while kkt_error > tolerance && ii < max_iterations
		θ_augmented = vcat(θ, relaxations[ii]) # relaxation parameter σ for inner inequalities
		(; x, y, s, kkt_error, status) = solve(solver_type, mcp, θ_augmented, warmstart_sol; kwargs...)
		status === :solved ? warmstart_sol = (; x, y, s) : warmstart_sol = nothing
		println("Iteration $ii: relaxation[ii] = $(relaxations[ii]), kkt_error = $kkt_error, status = $status")
		ii += 1
	end

	# Unpack primals per-player for ease of access later.
	private_primals = mcp.dims.private_primals
	num_players = length(private_primals)
	primals = [
		ii > 1 ? x[(1:private_primals[ii][1]) .+ sum(sum(private_primals[jj]) for jj ∈ 1:(ii-1))] :
		x[1:private_primals[ii][1]] for ii in 1:num_players
	]

	(; primals, variables = (; x, y, s), kkt_error, status)
end

# struct ParametricQuasiGOOP{T1, T2, T3, T4, T5, T6, T7, T8}
#     "Objective functions for all players"
#     objectives::T1
#     "Equality constraints for all players"
#     private_inner_equality_constraints::T2
#     "Inequality constraints for all players"
#     private_inner_inequality_constraints::T3
#     "Shared equality constraint"
#     shared_equality_constraints::T4
#     "Shared inequality constraint"
#     shared_inequality_constraints::T5

#     "Dimension of parameter vector"
#     parameter_dimensions::T6
#     "Dimension of primal variables for all players"
#     primal_dimensions::T7
#     "Dimension of equality constraints for all players"
#     equality_dimensions::T7
#     "Dimension of inequality constraints for all players"
#     inequality_dimensions::T7
#     "Dimension of shared equality constraint"
#     shared_equality_dimension::T8
#     "Dimension of shared inequality constraint"
#     shared_inequality_dimension::T8

#     "Corresponding Primal Dual System Representation."
#     pd_system::PrimalDualSys 
# end
