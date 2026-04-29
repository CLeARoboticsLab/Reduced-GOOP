using BlockArrays: Block, BlockArray
using LinearAlgebra: dot, norm
using QuasiGOOP

function run_single_player_path_constraint_test(; n = 3, verbose = false)
	x_template = BlockArray(zeros(n), [n])
	θ_template = BlockArray([0.0], [1])

	zero_objective(x, θ) = 0.0 * sum(x[Block(1)])
	nonnegative_constraint(x, θ) = x[Block(1)]

	problem = QuasiGOOP.ParametricGOOP(
		x_template,
		θ_template;
		preferences = [[zero_objective]],
		is_prioritized_constraint = [[false]],
		equality_constraints = [nothing],
		inequality_constraints = [nonnegative_constraint],
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

	mcp = QuasiGOOP.generate_mcp_complete_kkt_system(problem)

	options = QuasiGOOP.PATHOptions(;
		convergence_tolerance = 1e-8,
		ϵ₀ = 0.0,
		cumulative_iteration_limit = 100000,
		proximal_perturbation = 1e-2,
		major_iteration_limit = 1000,
		minor_iteration_limit = 1000,
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
		verbose,
	)

	θ_value = [0.0]
	z₀ = [ones(n); zeros(n)]
	output = QuasiGOOP.solve(QuasiGOOP.PATHSolver(), mcp, θ_value; z₀, options)

	x = output.z[1:n]
	γ = output.z[(n+1):(2n)]
	@assert all(x .>= -1e-8)
	@assert all(γ .>= -1e-8)
	@assert abs(dot(x, γ)) <= 1e-8
	@assert norm(γ) <= 1e-8

	println("status = ", output.status)
	println("x = ", x)
	println("γ = ", γ)
	println("residual = ", output.info.residual)

	return (; output, x, γ, mcp)
end

run_single_player_path_constraint_test()
