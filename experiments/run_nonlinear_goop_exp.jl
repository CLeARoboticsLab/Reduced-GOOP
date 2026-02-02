using QuasiGOOP
using Random
using BlockArrays
using NonlinearSolve
using Symbolics
using SymbolicTracingUtils

include("refactored_old_goop.jl")

# Simple PSD generator
function rand_psd(n, r)
	R = randn(r, n)
	return R' * R
end

function run_goop_experiments(; num_iters::Int = 100, seed::Int = 0)
	#### Trilevel Problem Definition ####
	n = 4
	n_digits = 4
	backend = SymbolicTracingUtils.SymbolicsBackend()

	player = 1

	# g_eq(x, θ) = [sin(x[1]) + x[2]^2 - 1.0;
	#               x[4] + cos(x[3])]

	g_eq(x, θ) = [x[1]^3 + x[2]^4 + x[3]^4 + x[4]^4 - 1.0; cos(x[3])]

	warmstart_x = [0.0; 0.0; 1.5708; 0.0]

	is_prioritized_constraint    = [[false, false, false, false]]
	equality_constraints         = [g_eq]
	inequality_constraints       = [nothing] # only equality constraints
	shared_equality_constraint   = nothing
	shared_inequality_constraint = nothing

	Random.seed!(seed)

	solved_new_goop = 0

	for iteration in 1:num_iters
		@info "........................STARTING NEW GOOP (iteration $(iteration))........................"

		# --- New GOOP: random data for this iteration ---
		Q₁ = rand_psd(n, 1)
		c₁ = rand(n)
		Q₂ = rand_psd(n, 1)
		c₂ = rand(n)
		Q₃ = rand_psd(n, 1)
		c₃ = rand(n)
        Q₄ = rand_psd(n, 1)
        c₄ = rand(n)

		# J₁(x, θ) = 0.5 * x[1:n]' * Q₁ * x[1:n] + c₁' * x[1:n] + x[1]^3     # non-quadratic
		# J₂(x, θ) = 0.5 * x[1:n]' * Q₂ * x[1:n] + c₂' * x[1:n] + sin(x[2])  # non-quadratic
		# J₃(x, θ) = 0.5 * x[1:n]' * Q₃ * x[1:n] + c₃' * x[1:n] + log(x[3])^2

		J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n] + x[1]^3 # non quadratic objective 
		J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n] + sin(x[2]) # non quadratic objective
		J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n] + x[3]^4
        J₄(x, θ) = 0.5x[1:n]'*Q₄*x[1:n] + c₄'*x[1:n] + x[4]^4


		# single player x, parameters
		x_block    = BlockArray(zeros(n), [n])
		θ_block   = BlockArray([0.0], [1])
		goop_prefs = [[J₁, J₂, J₃, J₄]]  # single player

		GOOP_trial = QuasiGOOP.ParametricGOOP(
			x_block,
			θ_block;
			preferences                  = goop_prefs,
			is_prioritized_constraint    = is_prioritized_constraint,
			equality_constraints         = equality_constraints,
			inequality_constraints       = inequality_constraints,
			shared_equality_constraint   = shared_equality_constraint,
			shared_inequality_constraint = shared_inequality_constraint,
		)

		NG_kkt_system = QuasiGOOP.generate_slacked_kkt_system(GOOP_trial)

		# Solve reduced GOOP
		status_new_goop, z_sol_new_goop, x_sol, s_sol, σ_sol, γ_sol,
		kkt_error, ϵ_val, outer_iters, total_iters = QuasiGOOP.solve(
			QuasiGOOP.InteriorPoint(),
			NG_kkt_system,
			θ_block;
			tol = 1e-5,
			η₀ = 0.0,   # no regularization
			min_stepsize = 1e-5,
			max_outer_iters = 100,
			z₀ = copy(warmstart_x),
			verbose = false,
		)

		if status_new_goop == :solved
			solved_new_goop += 1
		end

		println("[$iteration][New G] status = $(status_new_goop)")
		println("[$iteration][New G] Primal solution: $(round.(z_sol_new_goop[1:n], digits = n_digits))")
		println("[$iteration][New G] Dual solution ($(length(z_sol_new_goop) - n) variables): ",
			"$(round.(z_sol_new_goop[Not(1:n)], digits = n_digits))")
		println("[$iteration][New G] Objective: $(round(J₁(z_sol_new_goop[1:n], 0), digits = n_digits))")
		println("[$iteration][New G] number of equations: $(NG_kkt_system.kkt_dimension)")

		if status_new_goop != :solved
			@info "Skipping OLD GOOP (iteration $(iteration)) because NEW GOOP did not solve."
			continue
		end

		@info "........................STARTING OLD GOOP (iteration $(iteration))........................"

		# --- Build OLD GOOP KKT system (symbolic) ---
		# Symbolic variables
		x_sym = SymbolicTracingUtils.make_variables(backend, :x, n)
		θ_sym = SymbolicTracingUtils.make_variables(backend, :θ, n)
		ϵ_sym = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
		η_sym = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

		symbolic_type = eltype(x_sym)

		# Containers for duals and slacks
		Λ = symbolic_type[]
		Γ = symbolic_type[]
		Σ = symbolic_type[]
		s  = symbolic_type[]  # preference slacks (unused in this specific setup)

		total_levels = length(goop_prefs[player])

		# Build KKT for levels 2..K
		F_lower, z_lower = construct_kkt_old_goop!(
			backend,
			goop_prefs[player][2:end],
			is_prioritized_constraint[player][2:end],
			player,
			total_levels,
			x_sym,
			θ_sym,
			ϵ_sym,
			equality_constraints,
			inequality_constraints,
			Λ,
			Γ,
			Σ,
			s,
		)

		# Topmost level
		λ_top = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("λ_$(player)_1"),
			length(F_lower),
		)
		append!(Λ, λ_top)

		L_top = first(goop_prefs[player][1](z_lower, θ_sym)) - λ_top' * F_lower
		∇L_top = Symbolics.gradient(L_top, z_lower)

		F_full = Vector{symbolic_type}([∇L_top; F_lower])
		z_full = Vector{symbolic_type}(vcat(x_sym, s, Λ, Σ, Γ))

		# Indices for GOOP KKT system
		idx = blockedrange(length.([x_sym, s, Λ, Σ, Γ]))
		primal_dims = idx[Block(1)]
		preference_slack_dims = idx[Block(2)]
		interior_point_slack_dims = idx[Block(4)]
		inequality_constraint_dual_dims = idx[Block(5)]

		OG_kkt_system = QuasiGOOP.BuildGOOPKKTSystem(
			F_full,
			z_full,
			θ_sym,
			ϵ_sym,
			η_sym,
			primal_dims,
			preference_slack_dims,
			interior_point_slack_dims,
			inequality_constraint_dual_dims,
		)

		### Check if there exist OG duals given NG primal solutions via NonlinearSolve

		# Re-introduce symbolic x and λ₃ for substitution (same pattern as original code)
		x_sub = SymbolicTracingUtils.make_variables(backend, :x, n)
		# λ₃_sub = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_3"), 2)
        λ₄_sub = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_4"), 2)
		ϵ_sub = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
		η_sub = only(SymbolicTracingUtils.make_variables(backend, :η, 1))

		F_symbolic = OG_kkt_system.F_symbolic
		z_symbolic = OG_kkt_system.z_symbolic
		symbolic_type_sub = eltype(x_sub)

		# Substitute in primal NG solution for x and λ₃, and set ϵ, η = 0
		F_symbolic_after_sub = Vector{symbolic_type_sub}(
			Symbolics.substitute(
				F_symbolic,
				Dict(
					x_sub[1] => z_sol_new_goop[1],
					x_sub[2] => z_sol_new_goop[2],
					x_sub[3] => z_sol_new_goop[3],
					x_sub[4] => z_sol_new_goop[4],
					# λ₃ terms
					# λ₃_sub[1] => z_sol_new_goop[9],
					# λ₃_sub[2] => z_sol_new_goop[10],
                    # λ₄ terms
                    λ₄_sub[1] => z_sol_new_goop[11],
                    λ₄_sub[2] => z_sol_new_goop[12],
					ϵ_sub => 0,
					η_sub => 0,
				),
			),
		)

		# Compile numeric function F(u)
		F_eval, F_eval_ip! = Symbolics.build_function(
			F_symbolic_after_sub,
			z_symbolic[Not(1:(n+2))];  # +2 for λ₃
			expression = Val(false),
		)

		# Nonlinear residual (return vector)
		test_f!(u, p) = F_eval(u)

		# Initial guess for remaining variables
		z0 = zeros(length(z_symbolic) - n - 2)

		# Construct and solve the problem
		prob =  NonlinearLeastSquaresProblem(test_f!, z0)

		sol = NonlinearSolve.solve(prob)
		@assert length(sol.u) == length(z_symbolic) - n - 2

		@info "[$iteration] sol.retcode = $(sol.retcode)"
		@info "[$iteration] OG Duals from nonlinear solve: $(sol.u), length: $(length(sol.u))"

		# Compare λ₂, ψ₂
		# @info "[$iteration] λ₂ from NG: $(z_sol_new_goop[7:8]), λ₂ from OG (via nonlinearsolve): $(sol.u[5:6])"
		# @info "[$iteration] ψ₂ from NG: $(z_sol_new_goop[11:14]), ψ₂ from OG (via nonlinearsolve): $(sol.u[1:4])"

		# Full residual check
		F_eval_full, _ = Symbolics.build_function(
			F_symbolic,
			z_symbolic;
			expression = Val(false),
		)

		@info "[$iteration] maximum(abs.(F_eval(sol.u))): $(maximum(abs.(F_eval(sol.u))))"
		@info "[$iteration] maximum(abs.(F_eval_full(vcat(z_sol_new_goop[1:n], z_sol_new_goop[9:10], sol.u)))): " *
			#   "$(maximum(abs.(F_eval_full(vcat(z_sol_new_goop[1:n], z_sol_new_goop[9:10], sol.u)))))"
              "$(maximum(abs.(F_eval_full(vcat(z_sol_new_goop[1:n], z_sol_new_goop[11:12], sol.u)))))"


		@assert maximum(abs.(F_eval(sol.u))) < 1e-5 "iteration $iteration: NG solved but OG KKT conditions not satisfied!"
	end

	println("New GOOP solved $solved_new_goop / $num_iters iterations.")
	return solved_new_goop
end

run_goop_experiments()
