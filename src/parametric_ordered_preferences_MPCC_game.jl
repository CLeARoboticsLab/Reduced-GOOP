# Struct for a Lagrangian term
mutable struct Lagrangian_term{T <: Symbolics.Num}
	expr::Union{Vector{T}, T}
	duals::Union{Vector{T}, Nothing}
	deriv_order::Int
end

# Overload of Symbolics.gradient for Lagrangian_term
function Symbolics.gradient(f::Lagrangian_term, x::AbstractVector{<:Symbolics.Num})
	if f.deriv_order > 1 # 1: drop terms for quasiGOOP, # ≥2: keep terms for GOOP
		Main.@infiltrate
		return Lagrangian_term(zero.(x), f.duals, f.deriv_order + 1)
	end

	f.deriv_order += 1
	expr = isnothing(f.duals) ? f.expr : -f.expr' * f.duals
	return Lagrangian_term(Symbolics.gradient(expr, x), f.duals, f.deriv_order)
end

# Struct for constraints at multiple levels
mutable struct Constraints_ii{T <: Lagrangian_term}
	stationarity::Vector{Union{Nothing, Vector{T}}}
	equalities::Vector{Vector{T}}
	inequalities::Vector{Vector{T}}

	function Constraints_ii{T}(num_levels::Int) where {T <: Lagrangian_term}
		new{T}(
			fill(nothing, num_levels + 1),
			[Vector{T}() for _ in 1:(num_levels+1)],
			[Vector{T}() for _ in 1:(num_levels+1)],
		)
	end
end

# Factory for multiple players
Constraints_ii(num_levels::Int, num_players::Int) =
	[Constraints_ii{Lagrangian_term}(num_levels) for _ in 1:num_players]

struct ParametricOrderedPreferencesMPCCGame{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12}
	"Objective functions for all players"
	objectives::T1
	"Equality constraints for all players"
	private_inner_equality_constraints::T2
	"Inequality constraints for all players"
	private_inner_inequality_constraints::T3
	"Shared equality constraint"
	shared_equality_constraints::T4
	"Shared inequality constraint"
	shared_inequality_constraints::T5

	"Dimension of parameter vector"
	parameter_dimensions::T6
	"Dimension of primal variables for all players"
	primal_dimensions::T7
	"Dimension of equality constraints for all players"
	equality_dimensions::T7
	"Dimension of inequality constraints for all players"
	inequality_dimensions::T7
	"Dimension of shared equality constraint"
	shared_equality_dimension::T8
	"Dimension of shared inequality constraint"
	shared_inequality_dimension::T8

	"Corresponding ParametricMCP."
	parametric_mcp::Union{PrimalDualMCP, ParametricMCP}

	"Exact complementarity constraints for convergence evaluation."
	exact_complementarity_constraints::T9

	"Trajectory primals."
	trajectory_idx::T10

	"Sum of slacks for each level"
	private_slacks::T11

	"Slack indices for each level"
	slack_indices::T12
end

function ParametricOrderedPreferencesMPCCGame(;
	objectives,
	equality_constraints,
	inequality_constraints,
	prioritized_preferences,
	is_prioritized_constraint,
	shared_equality_constraints,
	shared_inequality_constraints,
	primal_dimensions,
	parameter_dimensions,
	equality_dimensions,
	inequality_dimensions,
	relaxation_mode = :standard,
	solver = "PATH",
)
	# Problem data
	ordered_priority_levels = eachindex(prioritized_preferences[1]) #TODO: eachindex(prioritized_preferences)
	num_players = length(objectives)
	num_levels = length(ordered_priority_levels)

	dual_dimension = 0
	inner_complementarity_constraints = Vector{Symbolics.Num}() # Symbolics.Num[]

	primal_dimension_ii = 0
	inequality_dimension_ii = inequality_dimensions
	equality_dimension_ii = equality_dimensions

	# one extra parameter for relaxation
	augmented_parameter_dimension = sum(parameter_dimensions) + 1

	dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions) #Block(1), Block(2),... original primals for each player
	dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

	# Initialize symbolic expression for player's private constraints.
	private_inner_equality_constraints = Vector{Symbolics.Num}[]
	private_inner_inequality_constraints = Vector{Symbolics.Num}[]

	K_ii = Symbolics.Num[] # KKT system for each player
	G_ii = Symbolics.Num[] # G(x,y;θ) = 0 for each player
	H_ii = Symbolics.Num[] # H(x,y;θ) ≥ 0 for each player

	∇K_ii = nothing # Jacobian

	# Store dimensions
	private_primals = [[dim] for dim in primal_dimensions]

	# Store (callable) slacks
	private_slacks = Function[]

	preference_slacks_dim_ii = 0
	prev_s̃ = Symbolics.Num[]
	prev_λ = Symbolics.Num[]


	start_idx = 1

	# Main.@infiltrate

	function set_up_level(priority_level, player_idx)
		########## 1a. PRIORITY & DIMENSION SETUP ##########
		is_pc = is_prioritized_constraint[player_idx][priority_level]

		if is_pc
			pc_fun = prioritized_preferences[player_idx][priority_level]
			preference_slacks_dim_ii = length(pc_fun(dummy_primals, dummy_parameters))
			# primal_dimension_ii += slack_dim
			append!(private_primals[player_idx], preference_slacks_dim_ii)
			# inequality_dimension_ii[player_idx] += preference_slacks_dim_ii
		end

		########## 1b. IP SLACKS DIMENSION SETUP ##########
		ip_slacks_dim_ii = priority_level > 1 ? preference_slacks_dim_ii : inequality_dimension_ii[player_idx] + preference_slacks_dim_ii
		λ_dim_ii = ip_slacks_dim_ii
		μ_dim_ii = equality_dimension_ii[player_idx]

		########## 2. SYMBOLIC VARIABLES ##########
		dims = [
			preference_slacks_dim_ii,
			ip_slacks_dim_ii,
			λ_dim_ii,
			μ_dim_ii,
		] # use NamedTuple dims = (; s̃, s, λ, μ)

		if priority_level == 1
			pushfirst!(dims, primal_dimension_ii)
		end

		# ψ appears only when priority_level > 1
		if priority_level > 1
			ψ_dim_ii, _ = size(∇K_ii)
			push!(dims, ψ_dim_ii)
		end

		Main.@infiltrate


		total_dim = sum(dims)

		if priority_level == 1
			Main.@infiltrate
			z̃_sym = Symbolics.@variables z̃[start_idx:(start_idx+total_dim-1)] # TODO: for priority_level > 1
			z̃ = Symbolics.scalarize(only(z̃_sym))
			z = BlockArray(z̃, dims)

			x = z[Block(1)]
			s = z[Block(2)]
			s̃ = z[Block(3)]
			λ = z[Block(4)]
			μ = z[Block(5)]
		else
			Main.@infiltrate
			x̃_sym = Symbolics.@variables z̃[1:primal_dimensions[player_idx]]
			x = Symbolics.scalarize(only(x̃_sym))

			z̃_sym = Symbolics.@variables z̃[start_idx:(start_idx+total_dim-1)] # TODO: for priority_level > 1
			z̃ = Symbolics.scalarize(only(z̃_sym))
			z = BlockArray(z̃, dims)

			s = z[Block(1)]
			s̃ = z[Block(2)]
			λ = z[Block(3)]
			μ = z[Block(4)]
			ψ = z[Block(5)]
		end
		append!(prev_s̃, s̃)
		append!(prev_λ, λ)

		θ̃ = Symbolics.scalarize(only(Symbolics.@variables(θ̃[1:augmented_parameter_dimension])))
		θ = BlockArray(θ̃, vcat(parameter_dimensions, [1]))

		Main.@infiltrate

		if priority_level == first(ordered_priority_levels) || isempty(private_inner_equality_constraints)
			push!(private_inner_equality_constraints, equality_constraints[player_idx](x, θ))
			push!(private_inner_inequality_constraints, inequality_constraints[player_idx](x, θ))
		end

		########## 3. PRIORITIZED OBJECTIVE OR CONSTRAINTS ##########
		if is_pc
			objective_ii = sum(s)
			println("Objective at level $priority_level for player $player_idx = ", objective_ii)

			aux_constraints = prioritized_preferences[player_idx][priority_level](x, θ) .+ s
			append!(private_inner_inequality_constraints[player_idx], aux_constraints) # Note: where is slacks ≥ 0?

			# Store slack function
			x_temp = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:(total_dim+start_idx+1)])))
			push!(private_slacks, Symbolics.build_function(sum(s), x_temp, θ, expression = Val{false}))
		else
			objective_ii = prioritized_preferences[player_idx][priority_level](x, θ)[1]
		end

		Main.@infiltrate

		########## 4. INITIAL CONSTRAINTS ##########
		ϵ = θ[augmented_parameter_dimension]
		if priority_level == 1 || isempty(G_ii)
			append!(G_ii, private_inner_equality_constraints[player_idx])
			append!(H_ii, private_inner_inequality_constraints[player_idx])
		end
		Main.@infiltrate

		########## 5. ORIGINAL GOOP: Lagrangian & Stationarity ##########
		h_ii = isempty(H_ii) ? Symbolics.Num[] : H_ii
		g_ii = isempty(G_ii) ? Symbolics.Num[] : G_ii
		L = objective_ii - λ' * h_ii - μ' * g_ii
		stationarity = Symbolics.expand.(Symbolics.gradient(L, [x; s]))

		if priority_level > 1
			@assert !isnothing(ψ)
			# add "policy dual" variables to the ∇ₓL
			stationarity -= ∇K_ii' * ψ # TODO: need -∇π'λₖₘᵢ
		end
		append!(G_ii, stationarity)

		Main.@infiltrate

		########## Update dual dimension and primal dimension ##########
		# dual_dimension = inequality_dimension_ii[player_idx] + equality_dimension_ii[player_idx]
		# primal_dimension_ii += dual_dimension
		# append!(private_primals[player_idx], dual_dimension) #[30,4,68,4,151]

		########## 10. FINAL COMPLEMENTARITY (LAST LEVEL CHECK) ##########
		# is_last_level = priority_level == last(ordered_priority_levels) || isnothing(is_prioritized_constraint[player_idx][priority_level+1])
		# relaxed_comp = sum(complementarity) .+ θ[augmented_parameter_dimension]

		# append!(private_inner_inequality_constraints[player_idx], relaxed_comp)
		# inequality_dimension_ii[player_idx] += length(dual_nonnegativity) + 1

		# Form the KKT system

		if priority_level > 1
			Main.@infiltrate
			# TODO
		end
		K_ii = [G_ii; H_ii .- s̃; s̃ .* λ .- ϵ]

		∇K_ii = Symbolics.jacobian(K_ii, [x; s])




		Main.@infiltrate

		start_idx = start_idx + total_dim

		# For subsequent levels
		# 1. Express s̃ in terms of inequality
		# s̃ₙ = /
		# 2. s .* λ - ϵ = 0 -> express λ in terms of s and ϵ


		# if is_last_level
		# start_idx += primal_dimension_ii
		# end

		# equality_dimension_ii[player_idx] += length(stationarity) # stationarity here to use quasi GOOP
		# complementarity = λ .* h_ii
		# append!(inner_complementarity_constraints, complementarity)

	end

	# Build KKT system for each priority level for each player's own problem
	for player in 1:num_players
		primal_dimension_ii = sum(private_primals[player])
		for priority_level in ordered_priority_levels
			if !isnothing(is_prioritized_constraint[player][priority_level])
				set_up_level(priority_level, player)
			end
		end
	end

	Main.@infiltrate

	# Use quasi GOOP to build the parametric game
	# use_quasi = true
	# if use_quasi
	# 	for player in 1:num_players
	# 		# Equality constraints: group and sum
	# 		eq_sum = group_and_sum_exprs_by_length(F_ii[player].equalities[end])

	# 		# Inequality constraints: flatten
	# 		ineq_flat = vcat((term.expr for term in F_ii[player].inequalities[end])...)

	# 		# Sanity checks
	# 		@assert all(isequal.(eq_sum, private_inner_equality_constraints[player]))
	# 		@assert all(isequal.(ineq_flat, private_inner_inequality_constraints[player]))
	# 	end
	# end

	# Inner slack indices
	offsets = cumsum([0; map(sum, private_primals[1:(end-1)])])
	all_ranges = mapreduce(vcat, enumerate(private_primals)) do (i, v)
		compute_slack_ranges(v, offsets[i])
	end
	slack_indices = mapreduce(r -> collect(r), vcat, all_ranges)

	# Set up for parametric game
	primal_dimensions = map(x->sum(x), private_primals)

	@assert num_players ==
			length(private_inner_equality_constraints) ==
			length(private_inner_inequality_constraints) ==
			length(primal_dimensions) ==
			length(equality_dimensions) ==
			length(inequality_dimensions)

	shared_equality_dimension =
		length(shared_equality_constraints(dummy_primals, dummy_parameters))
	shared_inequality_dimension =
		length(shared_inequality_constraints(dummy_primals, dummy_parameters))
	total_dimension =
		sum(primal_dimensions) +
		sum(equality_dimensions) +
		sum(inequality_dimensions) +
		shared_equality_dimension +
		shared_inequality_dimension

	# Define symbolic variables for this MCP.
	z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
	z = BlockArray(
		z̃,
		[
			sum(primal_dimensions),
			sum(equality_dimensions),
			sum(inequality_dimensions),
			shared_equality_dimension,
			shared_inequality_dimension,
		],
	)
	θ̃ = Symbolics.scalarize(
		only(Symbolics.@variables(θ̃[1:augmented_parameter_dimension])),
	)
	θ = BlockArray(θ̃, vcat(parameter_dimensions, [1]))

	x = BlockArray(z[Block(1)], primal_dimensions)
	μ = BlockArray(z[Block(2)], equality_dimensions)
	λ = BlockArray(z[Block(3)], inequality_dimensions)
	μₛ = z[Block(4)]
	λₛ = z[Block(5)]

	# Build symbolic expressions for objectives and (shared) constraints
	# Take original primals out of x
	trajectory_primals = [
		i > 1 ? z[(1:private_primals[i][1]) .+ sum(primal_dimensions[1:(i-1)])] :
		z[1:private_primals[i][1]] for i in 1:num_players
	]

	trajectory_x = BlockArray(
		vcat(trajectory_primals...),
		[private_primals[i][1] for i in 1:num_players],
	)
	# Block(1): z[1], ..., z[30]
	# Block(2): z[290],...,z[319]
	# Block(3): z[674], ..., z[703]

	fs = map(f->f(trajectory_x, θ), objectives)
	gs = private_inner_equality_constraints # contains MPCC (nested) constraints
	hs = private_inner_inequality_constraints # here too
	g̃ = shared_equality_constraints(trajectory_x, θ)
	h̃ = shared_inequality_constraints(trajectory_x, θ)

	if length(g̃) == 1 && g̃[1] == 0 && eltype(g̃) == Int
		g̃ = Symbolics.Num[0]
	end
	if length(h̃) == 1 && h̃[1] == 0 && eltype(h̃) == Int
		h̃ = Symbolics.Num[0]
	end

	# Build Lagrangian for all players.
	Ls = map(zip(1:num_players, fs, gs, hs)) do (i, f, g, h)
		f - μ[Block(i)]' * g - λ[Block(i)]' * h - μₛ' * g̃ - λₛ' * h̃
	end

	# Build F = [∇ₓLs, gs, hs, g̃, h̃].
	∇ₓLs = map(zip(Ls, blocks(x))) do (L, xᵢ)
		Symbolics.gradient(L, xᵢ)
	end

	# # Build Lagrangian for quasi GOOP
	# priority_level = length(ordered_priority_levels) + 1
	# quasi_∇ₓLs = map(1:num_players) do i
	# 	empty!(Lagrangian_terms)
	# 	push!(Lagrangian_terms, Lagrangian_term(fs[i], nothing, 0))
	# 	add_lagrangian_terms!(Lagrangian_terms, F_ii[i].equalities[priority_level], μ[Block(i)], priority_level, :equalities)
	# 	add_lagrangian_terms!(Lagrangian_terms, F_ii[i].inequalities[priority_level], λ[Block(i)], priority_level, :inequalities)
	# 	push!(Lagrangian_terms, Lagrangian_term(g̃, μₛ, 0))
	# 	push!(Lagrangian_terms, Lagrangian_term(h̃, λₛ, 0))

	# 	Symbolics.expand.(build_and_filter_stationarity!(x[Block(i)], Lagrangian_terms; prune_zeros = false))
	# end

	# TODO: compare Lagrangian function for quasi GOOP with the original GOOP


	F_symbolic = [reduce(vcat, ∇ₓLs), reduce(vcat, gs), reduce(vcat, hs), g̃, h̃]

	# Set lower and upper bounds for z.
	z̲ = [
		fill(-Inf, sum(primal_dimensions)) #length(F_symbolic[1])
		fill(-Inf, sum(equality_dimensions))
		fill(0, sum(inequality_dimensions))
		fill(-Inf, shared_equality_dimension)
		fill(0, shared_inequality_dimension)
	]
	z̅ = [
		fill(Inf, sum(primal_dimensions))
		fill(Inf, sum(equality_dimensions))
		fill(Inf, sum(inequality_dimensions))
		fill(Inf, shared_equality_dimension)
		fill(Inf, shared_inequality_dimension)
	]
	# Main.@infiltrate
	@info "MiCP total dimension: $(length(z̃))"
	@info "sum(iszero.(vcat(F_symbolic...))): $(sum(iszero.(vcat(F_symbolic...))))"

	# Build parametric (or PrimalDual) MCP.
	if solver == "PATH"
		parametric_mcp = ParametricMCP(
			reduce(vcat, F_symbolic),
			z̃,
			θ̃,
			z̲,
			z̅;
			compute_sensitivities = true,
		)
	else
		parametric_mcp = PrimalDualMCP(
			reduce(vcat, F_symbolic),
			z̃,
			θ̃,
			z̲,
			z̅;
			compute_sensitivities = true,
		)
	end

	# Standard relaxation mode ONLY
	if relaxation_mode === :standard
		println("Standard Relaxation Mode: ")
	else
		error("Unknown relaxation mode: $relaxation_mode")
	end

	# Main.@infiltrate

	# Form exact complementarity constraints to evaluate convergence
	exact_complementarity_constraints = Symbolics.build_function(
		inner_complementarity_constraints,
		z̃,
		θ̃,
		expression = Val{false},
	)[1]

	trajectory_idx = [
		i > 1 ? [(1:private_primals[i][1]) .+ primal_dimensions[i-1]] :
		[1:private_primals[i][1]] for i in 1:num_players
	]

	# Return ParametricOrderedPreferencesMPCCGame
	ParametricOrderedPreferencesMPCCGame(
		objectives,
		private_inner_equality_constraints,
		private_inner_inequality_constraints,
		shared_equality_constraints,
		shared_inequality_constraints,
		parameter_dimensions,
		primal_dimensions,
		equality_dimensions,
		inequality_dimensions,
		shared_equality_dimension,
		shared_inequality_dimension,
		parametric_mcp,
		exact_complementarity_constraints,
		trajectory_idx,
		private_slacks,
		slack_indices,
	)
end

function total_dim(problem::ParametricOrderedPreferencesMPCCGame)
	sum(problem.primal_dimensions) +
	sum(problem.equality_dimensions) +
	sum(problem.inequality_dimensions) +
	problem.shared_equality_dimension +
	problem.shared_inequality_dimension
end

"Solve a constrained parametric game."
function solve(
	problem::ParametricOrderedPreferencesMPCCGame,
	parameter_value;
	PATH_tolerance = 1e-6,
	initial_guess = nothing,
	verbose = false,
	return_primals = true,
)
	if problem.parametric_mcp isa ParametricMCP
		z0 = if !isnothing(initial_guess)
			initial_guess
		else
			zeros(total_dim(problem))
		end

		# Main.@infiltrate

		z, status, info = ParametricMCPs.solve(
			problem.parametric_mcp,
			parameter_value;
			initial_guess = z0,
			verbose = verbose,
			cumulative_iteration_limit = 1000000,
			proximal_perturbation = 1e-2,
			major_iteration_limit = 10000,
			minor_iteration_limit = 15000,
			convergence_tolerance = PATH_tolerance, #1e-1
			nms_initial_reference_factor = 50000,
			nms_maximum_watchdogs = 8000,
			nms_memory_size = 16000,
			nms_mstep_frequency = 5000,
			lemke_start_type = "advanced",
			lemke_rank_deficiency_iterations = 50,
			restart_limit = 120,
			gradient_step_limit = 120,
			use_basics = true,
			use_start = true,
		)

		solved = string(status) == "MCP_Solved"
	else
		# TODO PHM: Improve this
		lower_bounds = [
			fill(-Inf, sum(problem.primal_dimensions))
			fill(-Inf, sum(problem.equality_dimensions))
			fill(0, sum(problem.inequality_dimensions))
			fill(-Inf, problem.shared_equality_dimension)
			fill(0, problem.shared_inequality_dimension)
		]

		unconstrained_indices = findall(isinf, lower_bounds)
		constrained_indices = findall(!isinf, lower_bounds)

		# initial_guess[problem.slack_indices] .= 1.0
		# Main.@infiltrate

		x₀ = initial_guess[unconstrained_indices]
		# The dual variables must be positive
		y₀ = clamp.(initial_guess[constrained_indices], 1, Inf)
		s₀ = ones(length(y₀))

		(; status, x, y, s, kkt_error, ϵ, outer_iters) =
			MixedComplementarityProblems.solve(
				InteriorPoint(),
				problem.parametric_mcp,
				parameter_value;
				x₀,
				y₀,
				s₀,
				verbose = verbose,
				tol = 1e-4, # 1e-4
				max_inner_iters = 200, # 20
				max_outer_iters = 50, # 50
				tightening_rate = 0.05, # 0.1
				loosening_rate = 0.25, # 0.5
				min_stepsize = 5e-5, # 1e-4
			)
		z = vcat(x, y, s)

		# TODO PHM: Check what info ParametricMCPs was giving us to see if we are missing something useful now 
		info = (; kkt_error, ϵ, outer_iters)

		solved = string(status) == "solved"
	end

	# Compute slacks for each level
	slacks = vcat(
		map(private_slacks -> private_slacks(z, parameter_value), problem.private_slacks),
	)

	if return_primals
		primals = blocks(
			BlockArray(z[1:sum(problem.primal_dimensions)], problem.primal_dimensions),
		)
		return (; primals, variables = z, slacks, solved, info)
	else
		return (; variables = z, slacks, solved, info)
	end
end

function solve_relaxed_pop_game(
	problem::ParametricOrderedPreferencesMPCCGame,
	initial_guess::Union{Nothing, Vector{Float64}},
	parameters;
	ϵ = 1.0,
	κ = 0.1,
	max_iterations = 10,
	tolerance = 1e-7,
	verbose = false,
)
	solutions = []
	residuals = []
	relaxations = []

	# Main.@infiltrate

	exact_complementarity_constraints = problem.exact_complementarity_constraints

	if isnothing(initial_guess)
		initial_guess = zeros(total_dim(problem))
	else
		# warmstart duals as zeros
		initial_guess = vcat(initial_guess, zeros(total_dim(problem) - length(initial_guess)))
	end

	# warmstart initial_guess slacks
	initial_guess[problem.slack_indices] .= 1.0

	complementarity_residual = 1.0
	converged_tolerance = 1e-6
	PATH_tolerance = 5e-3 #2e-2

	relaxations = ϵ * κ .^ (0:max_iterations) # [1.0, 0.1, 0.01, ... 1e-10]
	ii = 1
	while complementarity_residual > tolerance && ii ≤ max_iterations + 1
		# If the relaxed problem is infeasible, terminate. Otherwise solve the relaxed problem for xᵏ⁺¹
		ϵ = relaxations[ii]
		augmented_parameters = vcat(parameters, ϵ)

		solution =
			solve(problem, augmented_parameters; PATH_tolerance, initial_guess, verbose)

		if !solution.solved
			printstyled(
				"Could not solve relaxed problem at relaxation factor $(ii).\n";
				color = :red,
			)
			ii += 1
			continue
		end

		# Check complementarity residual. Augmented_parameters should now have zero relaxation(ϵ)
		complementarity_violations =
			exact_complementarity_constraints(solution.variables, vcat(parameters, 0.0))
		complementarity_residual = findmax(-complementarity_violations)[1]
		println("complementarity_residual at iteration $(ii): ", complementarity_residual)
		push!(residuals, complementarity_residual)
		push!(relaxations, ϵ)
		push!(solutions, solution)

		if complementarity_residual < tolerance
			printstyled(
				"Found a solution with complementarity residual less than tol=$(tolerance).\n";
				color = :blue,
			)
			break
		end

		# Stop if iteration does not improve after 7 iterations
		if ii > 7 && norm(initial_guess - solution.variables) < converged_tolerance
			verbose && printstyled("Converged at iteration $(ii).\n"; color = :green)
			push!(solutions, solution)
			break
		end

		# Update initial_guess
		# initial_guess = zeros(total_dim(problem))
		# if complementarity_residual < 2.0 * tolerance
		# 	for i in 1:length(problem.objectives)
		# 		initial_guess[first(problem.trajectory_idx[i])] = solution.variables[first(problem.trajectory_idx[i])]
		# 	end
		# end
		initial_guess = solution.variables

		# Begin next iteration
		ii += 1
	end

	(; relaxation = relaxations, solution = solutions, residual = residuals)
end

"""
Helper functions
"""
function add_lagrangian_terms!(
	Lagrangian_terms::Vector{Lagrangian_term},
	constraints::Vector{<:Any},  # eqs or ineqs, each with `.expr` and `.deriv_order`
	duals::Vector{<:Any},        # mu or lambda
	priority_level::Int,
	constraint_type::Symbol,      # :equalities or :inequalities
)
	if priority_level == 1 || constraint_type == :inequalities
		constraint_dims = [length(e.expr) for e in constraints]
	elseif constraint_type == :equalities
		constraint_dims = unique([length(e.expr) for e in constraints]) # Level 2: [28, 46, 46, 46, 46] -> [28, 46]
	end

	duals = BlockArray(duals, constraint_dims)

	prev_len = nothing
	dual_idx = 0

	for constraint in constraints
		expr = constraint.expr
		deriv_order = constraint.deriv_order

		if prev_len !== length(expr)
			dual_idx += 1
			prev_len = length(expr)
		end

		@assert length(expr) == length(duals[Block(dual_idx)])
		push!(
			Lagrangian_terms,
			Lagrangian_term(expr, duals[Block(dual_idx)], deriv_order),
		)
	end
end


"Helper function to filter zeros out of the stationarity vector."
function build_and_filter_stationarity!(x, Lagrangian_terms; prune_zeros = true)
	n = length(x)
	stationarity = zero.(x)
	mask = falses(n) # mask[j] = true iff row j was ever non‑zero

	for (i, term) in enumerate(Lagrangian_terms)
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
		# stationarity = filter(!iszero, collect(stationarity)) # filter out all zero entries
	end

	stationarity
end

function group_and_sum_exprs_by_length(terms::Vector{Lagrangian_term})
	grouped = Dict{Int, Vector{Symbolics.Num}}()

	for term in terms
		len = length(term.expr)
		if haskey(grouped, len)
			grouped[len] .+= term.expr
		else
			grouped[len] = copy(term.expr)
		end
	end

	# Expand each grouped sum and vertically concatenate into a single vector
	return vcat([Symbolics.expand.(grouped[k]) for k in sort(collect(keys(grouped)))]...)
end

function compute_slack_ranges(v, offset::Int = 0)
	cs = cumsum(v)
	n_range = (length(cs) - 1) ÷ 2

	[(offset+cs[2*i-1]+1):(offset+cs[2*i]) for i in 1:n_range]
end

function save_data_to_file(filename::String, data)
	open("$filename.txt", "w") do io
		for elem in data
			println(io, elem)
		end
	end
end
