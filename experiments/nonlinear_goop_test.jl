using QuasiGOOP

include("old_goop.jl")

#### Trilevel Problem Definition ####
n = 4
n_digits = 4
backend = SymbolicTracingUtils.SymbolicsBackend()

player = 1

g_eq(x, θ) = [x[1]^3 + x[2]^4 + x[3]^4 + x[4]^4 - 1.0; cos(x[3])]

warmstart_x = [0.0; 0.0; 1.5708; 0.0]

is_prioritized_constraint = [[false, false, false]]
equality_constraints = [g_eq]
inequality_constraints = [nothing] # Use only equality constraints here, make sure goop.jl is modified
shared_equality_constraint = nothing
shared_inequality_constraint = nothing

Random.seed!()

for iteration in 1:1
	@info "........................STARTING NEW GOOP (iteration $(iteration))........................"
	Q₁ = rand_psd(n, 1)
	c₁ = rand(n)
	Q₂ = rand_psd(n, 1)
	c₂ = rand(n)
	Q₃ = rand_psd(n, 1)
	c₃ = rand(n)
	J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n] + x[1]^3 # non quadratic objective 
	J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n] + sin(x[2]) # non quadratic objective
	J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n] + x[3]^4

	global x = BlockArray(zeros(n), [n]) # single player
	parameters = BlockArray([0.0], [1])
	global goop_preferences = [[J₁, J₂, J₃]] # single player

	GOOP_trial1 = QuasiGOOP.ParametricGOOP(
		x,
		parameters;
		preferences = goop_preferences,
		is_prioritized_constraint,
		equality_constraints,
		inequality_constraints,
		shared_equality_constraint,
		shared_inequality_constraint,
	)

	NG_kkt_system = QuasiGOOP.generate_slacked_kkt_system(GOOP_trial1)

	# Solve reduced goop
	status_new_goop, z_sol_new_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = QuasiGOOP.solve(
		QuasiGOOP.InteriorPoint(),
		NG_kkt_system,
		parameters;
		tol = 1e-5,
		η₀ = 0.0, # no regularization
		min_stepsize = 1e-5,
		max_outer_iters = 100,
		z₀ = copy(warmstart_x),
		verbose = false,
	)

	println("[$iteration][New G] status = $(status_new_goop)")
	println("[$iteration][New G] Primal solution: $(round.(z_sol_new_goop[1:n], digits = n_digits))")
	println("[$iteration][New G] Dual solution ($(length(z_sol_new_goop) - n) variables): $(round.(z_sol_new_goop[Not(1:n)], digits = n_digits))")
	println("[$iteration][New G] Objective: $(round(J₁(z_sol_new_goop[1:n], 0), digits = n_digits))")
	println("[$iteration][New G] number of equations: $(NG_kkt_system.kkt_dimension)")

	@info "........................STARTING OLD GOOP (iteration $(iteration))........................"

	global x = SymbolicTracingUtils.make_variables(backend, :x, n)
	global θ = SymbolicTracingUtils.make_variables(backend, :θ, n)
	global ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
	global η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))
	global symbolic_type = eltype(x)

	# Keep track of all equality constraint duals (λ) that we create.
	global Λ = symbolic_type[]

	# Keep track of all inequality constraint duals (γ) that we create.
	global Γ = symbolic_type[]

	# Keep track of all preference slack (s) and interior point slack variables (σ) that we create.
	global s = symbolic_type[]
	global Σ = symbolic_type[]
	(; F, z) = construct_kkt_old_goop(goop_preferences[player][2:end], is_prioritized_constraint[player][2:end], player)

	# Topmost level (final)
	λ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("λ_$(player)_1"),
		length(F),
	)
	push!(Λ, λ...)

	L = first(goop_preferences[player][1](z, θ)) - λ'*F
	∇L = Symbolics.gradient(L, z)
	F = Vector{symbolic_type}(
		[
			∇L;
			F
		],
	)
	z = Vector{symbolic_type}(
		vcat(x, s, Λ, Σ, Γ),
	)

	# Set up common memory
	variable_dimension = length(z)
	kkt_dimension = length(F)
	idx = blockedrange(length.([x, s, Λ, Σ, Γ]))
	primal_dims = idx[Block(1)]
	preference_slack_dims = idx[Block(2)] # s
	interior_point_slack_dims = idx[Block(4)] # Σ
	inequality_constraint_dual_dims = idx[Block(5)] # Γ

	OG_kkt_system = QuasiGOOP.BuildGOOPKKTSystem(
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

	### Check if there exists OG duals given NG primal solutions via NonlinearSolve
	x = SymbolicTracingUtils.make_variables(backend, :x, n)
	λ₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_3"), 2) # length(λ₃) = 2
	ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
	η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))
	F_symbolic = OG_kkt_system.F_symbolic
	z_symbolic = OG_kkt_system.z_symbolic
	symbolic_type = eltype(x)

	F_symbolic_after_sub = Vector{symbolic_type}(
		Symbolics.substitute(
			F_symbolic,
			Dict([
				x[1] => z_sol_new_goop[1],
				x[2] => z_sol_new_goop[2],
				x[3] => z_sol_new_goop[3],
				x[4] => z_sol_new_goop[4],
				# Add λ₃
				λ₃[1] => z_sol_new_goop[9],
				λ₃[2] => z_sol_new_goop[10],
				ϵ => 0, η => 0]
			),
		),
	)

	# Compile the numeric function (returns F given z)
	F_eval, F_eval_ip! = Symbolics.build_function(
		F_symbolic_after_sub,
		z_symbolic[Not(1:(n+2))]; # +2 for λ₃
		expression = Val(false),
	)

	function test_f!(u, p)
		# F_eval_ip!(du, u) # This keeps getting segfaults
		# return nothing
		return F_eval(u)
	end

	# Initial guess and parameters
	z_val = zeros(length(z_symbolic) - n - 2)

	# Construct the problem (3 positional args max) and solve it
	prob = isnothing(inequality_constraints[1]) ? NonlinearProblem(test_f!, z_val) : NonlinearLeastSquaresProblem(test_f!, z_val)
	sol = NonlinearSolve.solve(prob)
	@assert length(sol.u) == length(z_symbolic) - n - 2

	@info "[$iteration] sol.retcode(sol) is $(sol.retcode)"
	@info "[$iteration] OG Duals from nonlinear solve: $(sol.u), length: $(length(sol.u))"

	# Check λ₃, λ₂, ψ₂
	# @info "λ₃ from NG: $(z_sol_new_goop[9:10]), λ₃ from OG (via nonlinearsolve): $(sol.u[1:2])"
	# @info "λ₂ from NG: $(z_sol_new_goop[7:8]), λ₂ from OG (via nonlinearsolve): $(sol.u[7:8])"      # λ_1_2[5]
	# @info "ψ₂ from NG: $(z_sol_new_goop[11:14]), ψ₂ from OG (via nonlinearsolve): $(sol.u[3:6])" # λ_1_2[1:4]

	@info "[$iteration] λ₂ from NG: $(z_sol_new_goop[7:8]), λ₂ from OG (via nonlinearsolve): $(sol.u[5:6])"      # λ_1_2[5]
	@info "[$iteration] ψ₂ from NG: $(z_sol_new_goop[11:14]), ψ₂ from OG (via nonlinearsolve): $(sol.u[1:4])" # λ_1_2[1:4]


	# Sanity check
	F_eval_full, _ = Symbolics.build_function(
		F_symbolic,
		z_symbolic;
		expression = Val(false),
	)

	# Check sol.u is indeed the dual OG solution
	@info "[$iteration] maximum(abs.(F_eval(sol.u))): $(maximum(abs.(F_eval(sol.u))))"
	@info "[$iteration] maximum(abs.(F_eval_full(vcat(z_sol_new_goop[1:n], sol.u))): $(maximum(abs.(F_eval_full(vcat(z_sol_new_goop[1:n], z_sol_new_goop[9:10], sol.u)))))"

	# Final check
	if status_new_goop == :solved
		@assert maximum(abs.(F_eval(sol.u))) < 1e-5 "iteration $iteration: NG solved but OG KKT conditions not satisfied!"
	end


	# compute l1 difference between two solutions
	# @show sol.u .- z_sol_new_goop[Not(1:n)]
	# @show norm(sol.u .- z_sol_new_goop[Not(1:n)], Inf)
end
