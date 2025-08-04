using Test: @testset, @test

using QuasiGOOP
using ParametricMCPs: ParametricMCPs
using LinearAlgebra: I, norm
using Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes

@testset "Trilevel Equality-Constrained Quadratic Program" begin
	""" Upper-level problem:
	min_{x} (1/2)x' Q₁ x + c₁'x
	subject to:
		x ∈ argmin_{x} (1/2)x' Q₂ x + c₂' x
						  subject to: x ∈ argmin_{x} (1/2)x' Q₃ x + c₃' x
														A₃x = b₃
	"""
	n = 4 # x dimension
	m = 2 # equality dimension

	# Problem data
	Q₁ = I(n)
	c₁ = [1.0, 0.0, -1.0, 2.0]
	Q₂ = 2I(n) # [0 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 1]
	c₂ = [-1.0, 2.0, 0.0, 1.0]
	Q₃ = 3I(n) #[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]
	c₃ = [0.5, -0.5, 1.0, 0.0]
	A₃ = [1 0 1 1; 0 1 1 0]
	b₃ = [1.0, 2.0]

	##### ORIGINAL GOOP VERSION ######

	f(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]

	g(x, θ) = [
		Q₂ * x[1:n] .+ c₂ - A₃'*x[(n+1):(n+m)] - Q₃'*x[(n+m+1):(n+m+n)]; # Q₂x + c₂ - A₃'μ₂₁ - Q₃'μ₂₂ = 0
		A₃*x[(n+m+1):(n+m+n)]; # A₃μ₂₂ = 0
		Q₃ * x[1:n] .+ c₃ - A₃'*x[(n+m+n+1):(n+m+n+m)]; # Q₃x + c₃ - A₃'μ₃ = 0
		A₃*x[1:n] .- b₃ # A₃x - b₃ = 0
	]
	h(x, θ) = []

	dummy_primals = zeros(n + m + n + m)
	dummy_parameters = [0.0]

	problem = QuasiGOOP.ParametricOptimizationProblem(;
		objective = f,
		equality_constraint = g,
		inequality_constraint = h,
		parameter_dimension = 1,
		primal_dimension = length(dummy_primals),
		equality_dimension = length(g(dummy_primals, dummy_parameters)),
		inequality_dimension = 0,
	)

	(; primals, variables, status, info) = QuasiGOOP.solve(problem, [0])
    orig_primals = primals[1:n]
    orig_objective = f(orig_primals, 0)

	##### NEW VERSION ######
	dummy_primals = zeros(n+2n+m+n+m+m)
	dummy_parameters = [0.0]

	J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]
	J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n]
	J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n]
	new_g(x, θ) = A₃*x[1:n] .- b₃

	x = zeros(n) #zeros(n+2n+m+n+m+m)
	θ = 0.0


	GOOP_trial1 = QuasiGOOP.ParametricGOOP(
		x,
		θ;
		preferences = [[J₁, J₂, J₃]],
		is_prioritized_constraint = [[false, false, false]],
		equality_constraints = [new_g],
		inequality_constraints = [nothing],
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

	(; F, z, θ) = QuasiGOOP.generate_slacked_kkt_system(GOOP_trial1)
	F = vcat(F, zeros(length(z) - length(F))) # TODO: (DH) cross-check with David
	z̲ = fill(-Inf, length(z))
	z̅ = fill(Inf, length(z))
	parameter_value = zeros(length(θ))

	parametric_mcp = ParametricMCPs.ParametricMCP(F, z, θ, z̲, z̅; compute_sensitivities = false)
	z_sol, status, info = ParametricMCPs.solve(
		parametric_mcp,
		parameter_value;
		initial_guess = zeros(length(z)),
		verbose = false,
		cumulative_iteration_limit = 100000,
		proximal_perturbation = 1e-2,
		use_basics = true,
		use_start = true,
	)
    new_primals = z_sol[1:n]
    new_objective = f(new_primals, 0)

	@test isapprox(orig_primals, new_primals, atol = 1e-6)
	@test isapprox(orig_objective, new_objective, atol = 1e-6)
end
