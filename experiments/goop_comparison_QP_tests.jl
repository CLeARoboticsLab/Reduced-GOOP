using QuasiGOOP
using QuasiGOOP: SymbolicTracingUtils
using Random
using LinearAlgebra
using BlockArrays
using ParametricMCPs
using Symbolics
using ProgressMeter

include("old_goop.jl")

const NUM_TESTS = 100
const MATCH_TOL = 1e-2
const N = 4
const M = 2
const PLAYER = 1

backend = SymbolicTracingUtils.SymbolicsBackend()
parameters = BlockArray([0.0], [1])

Q₁ = zeros(N, N)
c₁ = zeros(N)
Q₂ = zeros(N, N)
c₂ = zeros(N)
Q₃ = zeros(N, N)
c₃ = zeros(N)
Aₑ = zeros(M, N)
bₑ = zeros(M)
Aᵢ = [1 0 0 0; 0 1 0 0]
bᵢ = [0.5, 0.5]

function rand_psd(n::Int, r::Int)
	R = randn(r, n)
	return R' * R
end

J₁(x, θ) = 0.5 * x[1:N]' * Q₁ * x[1:N] + c₁' * x[1:N]
J₂(x, θ) = 0.5 * x[1:N]' * Q₂ * x[1:N] + c₂' * x[1:N]
J₃(x, θ) = 0.5 * x[1:N]' * Q₃ * x[1:N] + c₃' * x[1:N]
g_eq(x, θ) = Aₑ * x[1:N] .- bₑ
g_ineq(x, θ) = Aᵢ * x[1:N] .- bᵢ

function solutions_match(primal_new, primal_old; primal_tol = 1e-2)
	primal_diff = primal_new .- primal_old
	max_primal_diff = maximum(abs, primal_diff)
	return max_primal_diff < primal_tol
end

function run_trials(num_tests::Int)
	mismatch_count = 0
	solved_count = 0
	infeasible_count = 0
	error_count = 0
	older_failed_trials = Int[]
	new_failed_trials = Int[]
	mismatch_trials = Int[]
	new_mismatch_complementarity = Int[]
	col_space_inclusion = Int[]

	@showprogress for trial in 1:num_tests
		Random.seed!(trial)
		global Q₁ = rand_psd(N, 1)
		global c₁ = rand(N)
		global Q₂ = rand_psd(N, 1)
		global c₂ = rand(N)
		global Q₃ = rand_psd(N, 1)
		global c₃ = rand(N)
		global Aₑ = rand(M, N)
		global bₑ = rand(M)
		global Aᵢ = [1 0 0 0; 0 1 0 0]
		global bᵢ = [0.5, 0.5]

		# Check column space inclusion tests (1,2)
		colspace_issubset(Aᵢ', hcat(Aₑ', Q₃)) && push!(col_space_inclusion, trial)
		A = [
			Q₃ Aᵢ';
			zeros(M, N + M)
		]
		B = [ 
			Q₃ Aᵢ';
			Aᵢ zeros(M, M)
		]
		if !colspace_issubset(A, B)
			error("Column space inclusion check failed: NG ⊈ OG")
		end

		try
			x_block = BlockArray(zeros(N), [N])

			global goop_preferences = [[J₁, J₂, J₃]]
			global is_prioritized_constraint = [[false, false, false]]
			global equality_constraints = [g_eq]
			global inequality_constraints = [g_ineq]

			goop_model = QuasiGOOP.ParametricGOOP(
				x_block,
				parameters;
				preferences = goop_preferences,
				is_prioritized_constraint,
				equality_constraints,
				inequality_constraints,
				shared_equality_constraint = nothing,
				shared_inequality_constraint = nothing,
			)

			kkt_system = QuasiGOOP.generate_slacked_kkt_system(goop_model)

			status_new, z_sol_new, _, _, _, _, _, _, _, _ = QuasiGOOP.solve(
				QuasiGOOP.InteriorPoint(),
				kkt_system,
				parameters;
				tol = 1e-5,
				η₀ = 0.0,
				min_stepsize = 1e-4,
				max_outer_iters = 50,
				z₀ = nothing,
				verbose = false,
			)
			if "$(status_new)" != "solved"
				push!(new_failed_trials, trial)
			else
				# new goop solved
				z_primal = z_sol_new[1:N]
				γ₂ = z_sol_new[13:14]
				γ₃ = z_sol_new[15:16]
				gineq_vals = g_ineq(z_primal, [0.0])
				mismatch_complementarity = !all(isapprox.(gineq_vals .* γ₂, 0.0, atol = MATCH_TOL)) # check complementarity
				if mismatch_complementarity
					push!(new_mismatch_complementarity, trial)
				end
			end
			global x = SymbolicTracingUtils.make_variables(backend, :x, N)
			global θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1))
			global symbolic_type = eltype(x)

			(; F, G, z) = construct_kkt_older_goop(
				goop_preferences[PLAYER][2:end],
				is_prioritized_constraint[PLAYER][2:end],
				PLAYER,
			)

			λ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("λ_$(PLAYER)_1"),
				length(F),
			)
			γ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γ_$(PLAYER)_1"),
				length(G),
			)

			L = first(goop_preferences[PLAYER][1](z, θ)) - λ' * F - γ' * G
			∇L = Symbolics.gradient(L, z)
			F_full = Vector{symbolic_type}([∇L; F])

			z_lower = vcat(fill(-Inf, length(F_full)), fill(0.0, length(G)))
			z_upper = vcat(fill(Inf, length(F_full)), fill(Inf, length(G)))

			parametric_mcp = ParametricMCPs.ParametricMCP(
				[F_full; G],
				[z; λ; γ],
				[θ],
				z_lower,
				z_upper;
				compute_sensitivities = false,
			)

			z_sol_older, status_older, _ = ParametricMCPs.solve(
				parametric_mcp,
				[1e-4];
				initial_guess = zeros(length([z; λ; γ])),
				verbose = false,
				cumulative_iteration_limit = 100000,
				proximal_perturbation = 1e-2,
				use_basics = true,
				use_start = true,
			)

			if "$(status_older)" == "MCP_Solved"
				solved_count += 1
				primal_new = z_sol_new[1:N]
				primal_old = z_sol_older[1:N]
				if !solutions_match(primal_new, primal_old)
					mismatch_count += 1
					push!(mismatch_trials, trial)
				elseif status_new == :failed
					pop!(new_failed_trials)
				end
			else
				infeasible_count += 1
				push!(older_failed_trials, trial)
			end
		catch
			error_count += 1
		end
	end

	println("Ran $(num_tests) randomized tests.")
	println("Older G solved cases: $(solved_count)")
	println("Older G infeasible or unsolved cases: $(infeasible_count), trials: $(older_failed_trials)")
	println("New G mismatches when Older G solved: $(mismatch_count), trials: $(mismatch_trials)")
	if !isempty(new_failed_trials)
		println("Trials where New G failed: $(setdiff(new_failed_trials, older_failed_trials))")
	end
	println("Trials where New G mismatch complementarity: $(length(new_mismatch_complementarity)), trials: $(setdiff(new_mismatch_complementarity, new_failed_trials))")
	println("Trials where column space inclusion held: $(length(col_space_inclusion)), trials: $(col_space_inclusion)")
end



run_trials(NUM_TESTS)
