mutable struct Lagrangian_term{T <: Symbolics.Num}
	expr::Union{Vector{T}, T}
	duals::Union{Vector{T}, Nothing}
	deriv_order::Int
end

function Symbolics.gradient(f::Lagrangian_term, x::AbstractVector{<:Symbolics.Num})
	f.deriv_order > 1 && Lagrangian_term(zero.(x), f.duals, f.deriv_order + 1)

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

struct PrimalDualSysEqn{T1, T2}
	"A callable function that computes F!(val, x, λ; θ, μ) in-place for 'val'"
	F!::T1
	"A callable function that computes ∇F!(val, x, λ; θ, μ) in-place for 'val'"
	∇F!::T2
	#TODO dimensions
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
		objectives_symbolic,
		equality_symbolic,
		inequality_symbolic,
		preferences_symbolic,
		shared_equality_symbolic,
		shared_inequality_symbolic,
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

	PrimalDualSysEqn(;
		objectives_symbolic,
		equality_symbolic,
		inequality_symbolic,
		preferences_symbolic,
		shared_equality_symbolic,
		shared_inequality_symbolic,
	)
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
	println("Make a PrimalDualSysEqn object from callable functions + implement approximations")
	backend = SymbolicTracingUtils.SymbolicsBackend()

	# Problem data
	# Append objectives to the end of preferences for each player_idx
	for (i, obj) in enumerate(objectives)
		push!(preferences.preferences[i], obj)
	end
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
		if priority_level < last(ordered_priority_levels)
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
		end

		Main.@infiltrate

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
		μ = θ[end] # annealing parameter

		x = BlockArray(z[Block(1)], private_primals[player_idx])
		λ = z[Block(2)] # dual variables for equality constraints
		Main.@infiltrate

		# Step 4. Define symbolic expression for objective and (equality) constraints
		if priority_level < last(ordered_priority_levels)
			# Step 4a. Define symbolic expression for objective and (equality) constraints
			preference_slacks_ii = blocks(x)[end-1] # These slacks are introduced before reformulation slacks; second-last block
			barrier_slacks_ii = blocks(x)[end] # last block
			objective_ii = sum(preference_slacks_ii) # or sum(preference_slacks_ii)
			println("objective_$(priority_level) for player $(player_idx) = ", objective_ii)
			barrier_objective_ii = μ * sum(log.(barrier_slacks_ii))

			# Replace inequality constriants: fg(x,sₚ) = sₚ + g(x) ≥ 0, fg(x,sₚ) - sᵦ = 0
			auxillary_constraints = prioritized_constraints_ii(x, θ) .+ preference_slacks_ii

			if priority_level == first(ordered_priority_levels)
				push!(
					Lagrangian_terms,
					Lagrangian_term(
						equality_constraints[player_idx](x, θ), nothing, 0), #f
				)
				push!(
					Lagrangian_terms,
					Lagrangian_term(
						vcat(
							auxillary_constraints,
							preference_slacks_ii,
							inequality_constraints[player_idx](x, θ)) - barrier_slacks_ii, nothing, 0), #fg
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
							preference_slacks_ii) - barrier_slacks_ii, nothing, 0), #fg
				)
			end
			# append!(F_ii[player_idx].equalities[priority_level], Lagrangian_terms)
			# empty!(Lagrangian_terms)
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

		else #topmost level objective
			Main.@infiltrate
			objective_ii = preferences.preferences[player_idx][priority_level](x, θ)
			println("objective_$(priority_level) for player $(player_idx) = ", objective_ii)
			for kk in 1:length(F_ii[player_idx].equalities[priority_level-1])
				let
					expr = F_ii[player_idx].equalities[priority_level-1][kk].expr
					push!(
						Lagrangian_terms,
						Lagrangian_term(expr, nothing, 0),
					)
				end
			end
			barrier_objective_ii = 0
		end
		append!(F_ii[player_idx].equalities[priority_level], Lagrangian_terms)
		empty!(Lagrangian_terms)

		Main.@infiltrate
		# Step 5. Define symbolic expression for Lagrangian and stationarity conditions/constraints
		if priority_level == first(ordered_priority_levels)
			dims = [length(e.expr) for e in F_ii[player_idx].equalities[priority_level]]
		else
			dims = let
				ns = [length(e.expr) for e in F_ii[player_idx].stationarity[priority_level-1]]
				ds = [length(e.expr) for e in F_ii[player_idx].equalities[priority_level]]
				vcat(ns[1], ds) # level 2:[110, 28, 64, 14], level 3: [223, 28, 64, 14, 56]
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
			prev_st = F_ii[player_idx].stationarity[priority_level-1]
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
			push!(
				Lagrangian_terms,
				Lagrangian_term(expr, λ[Block(dual_idx)], deriv_order),
			)
		end

		# Main.@infiltrate
		stationarity = build_and_filter_stationarity!(x, Lagrangian_terms)
		F_ii[player_idx].stationarity[priority_level] = copy(Lagrangian_terms)
		empty!(Lagrangian_terms)
		if priority_level < last(ordered_priority_levels)
			check_stationarity!(
				priority_level,
				ordered_priority_levels,
				private_inner_equality_constraints, # `private_inner_equality_constraints` has been updated in-place,
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
				stationarity,
			)
		end

		# Update primal_dimension_ii with induced primal dimension
		induced_primal_dimension_ii = length(λ)
		primal_dimension_ii += induced_primal_dimension_ii
		append!(private_primals[player_idx], induced_primal_dimension_ii)

		# Update equality_dimension_ii
		Main.@infiltrate
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

end

"Helper function to filter zeros out of the stationarity vector."
function build_and_filter_stationarity!(x, Lagrangian_terms)
	n = length(x)
	stationarity = zero.(x)
	mask = falses(n) # mask[j] = true iff row j was ever non‑zero


	for (i, term) in enumerate(Lagrangian_terms)
		new_term = Symbolics.gradient(term, collect(x))
		Lagrangian_terms[i] = new_term
		stationarity .+= new_term.expr
		println("new_term.deriv_order = ", new_term.deriv_order)
		new_term.deriv_order > 2 && @assert all(Symbolics.iszero.(new_term.expr))
		# record which positions are non‑zero in this term
		mask .|= .!Symbolics.iszero.(new_term.expr) # mask[j] = mask[j] || (!Symbolics.iszero(expr[j])), elementwise OR and assign
	end

	# drop all rows that never lit up
	if any(.!mask)
		stationarity = stationarity[mask]
		for t in Lagrangian_terms
			t.expr = t.expr[mask]
		end
	end

	stationarity
end

"""
Compute and verify stationarity at the given `priority_level` for `player_idx`.  
- Appends the appropriate equality (and slack) constraints into `private_inner_equality_constraints`  
- Computes the gradient of the Lagrangian and compares it to `stationarity`  
- Returns `match::Bool`
"""
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
	stationarity,
)::Bool

	if priority_level == first(ordered_priority_levels)
		# — base level: include all equality + slacks —
		push!(
			private_inner_equality_constraints,
			equality_constraints[player_idx](x, θ),
		)
		append!(
			private_inner_equality_constraints[player_idx],
			vcat(
				auxillary_constraints,
				preference_slacks_ii,
				inequality_constraints[player_idx](x, θ),
			) .- barrier_slacks_ii,
		)

		g_ii = private_inner_equality_constraints[player_idx]
		L = objective_ii - barrier_objective_ii - λ' * g_ii
		computed = Symbolics.gradient(L, x)

		match = all(isequal.(stationarity, computed))
		println("Check stationarity at level $priority_level = $match")

		# prepend for the next level
		pushfirst!(private_inner_equality_constraints[player_idx], stationarity...)

	elseif priority_level > 1
		# — higher levels: only the slack equalities —
		append!(
			private_inner_equality_constraints[player_idx],
			vcat(auxillary_constraints, preference_slacks_ii) .- barrier_slacks_ii,
		)

		g_ii = private_inner_equality_constraints[player_idx]
		L = objective_ii - barrier_objective_ii - λ' * g_ii
		computed = Symbolics.gradient(L, x)

		# mask out always-zero entries
		mask = .! Symbolics.iszero.(computed)
		computed_masked = computed[mask]
		if length(stationarity) == length(computed_masked)
			match = all(isequal.(stationarity, computed_masked))
		else
			println(
				"Warning: cannot compare stationarity (length=$(length(stationarity))) ",
				"with computed_masked (length=$(length(computed_masked))). ",
				"Setting match = false.",
			)
			match = false
		end
		println("Check stationarity at level $priority_level = $match")

		# remove previous stationarity constraints
		num_prev = length(F_ii.stationarity[priority_level-1][1].expr)
		deleteat!(private_inner_equality_constraints[player_idx], 1:num_prev)

		# prepend the new stationarity
		pushfirst!(private_inner_equality_constraints[player_idx], stationarity...)

	else
		error("Invalid priority_level: $priority_level")
	end

	return match
end

"Helper function to create a 'PrimalDualSysEqn' object from symbolic functions."
function PrimalDualSysEqn(;
	objectives::Vector{<:Symbolics.Num},
	equality_constraints::Vector{<:Symbolics.Num},
	inequality_constraints::Vector{<:Symbolics.Num},
	preferences::ordered_preferences,
	shared_equality_constraints,
	shared_inequality_constraints,
	primal_dimensions::Vector{Int},
	parameter_dimensions::Vector{Int},
	equality_dimensions::Vector{Int},
	inequality_dimensions::Vector{Int},
)
	println("Make a PrimalDualSysEqn object from symbolic functions")
	#TODO

	F! = nothing
	∇F! = nothing
	PrimaldualSysEqn(F!, ∇F!)
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
#     pd_system::PrimalDualSysEqn 
# end

# "Helper function to create a 'PrimalDualSysEqn' object from callable functions."
# function PrimalDualSysEqn(;
#     objectives::Vector{<:Function},
#     equality_constraints::Vector{<:Function},
#     inequality_constraints::Vector{<:Function},
#     preferences::ordered_preferences,
#     shared_equality_constraints::Vector{<:Function},
#     shared_inequality_constraints::Vector{<:Function},
#     primal_dimensions::Vector{Int},
#     parameter_dimensions::Vector{Int},
#     equality_dimensions::Vector{Int},
#     inequality_dimensions::Vector{Int},
#     )
#     println("Make a PrimalDualSysEqn object from callable functions")
#     #TODO

# end
