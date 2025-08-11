using Test: @testset, @test

using QuasiGOOP
using ParametricMCPs: ParametricMCPs
using LinearAlgebra: I, norm
using Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes

@testset "Trilevel Quadratic Program" begin
	""" Upper-level problem:
	min_{x} (1/2)x' Q₁ x + c₁'x
	subject to:
		x ∈ argmin_{x} (1/2)x' Q₂ x + c₂' x
						  subject to: x ∈ argmin_{x} (1/2)x' Q₃ x + c₃' x
														A₃x - b₃ = 0, G₃x - h₃ ≥ 0
	"""
	n = 4 # x dimension
	m = 2 # equality dimension

	# Problem data
	Q₁ = I(n)
	c₁ = [1.0, 0.0, -1.0, 2.0]
	Q₂ = 2I(n) # [0 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 1]
	c₂ = [-1.0, 2.0, 0.0, 1.0]
	Q₃ = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0] # I(n)
	c₃ = [0.5, -0.5, 1.0, 0.0]
	A₃ = [1 0 1 1; 0 1 1 0]
	b₃ = [1.0, 2.0]

	dummy_primals = zeros(n+2n+m+n+m+m)
	dummy_parameters = [0.0]

	J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]
	J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n]
	J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n]
	g_eq(x, θ) = A₃*x[1:n] .- b₃
	g_ineq(x, θ) = [x[1] - 0.5; x[2] - 0.5]

	x = BlockArray(zeros(n), [n]) # single player
	θ = 0.0

	GOOP_trial1 = QuasiGOOP.ParametricGOOP(
		x,
		θ;
		preferences = [[J₁, J₂, J₃]],
		is_prioritized_constraint = [[false, false, false]],
		equality_constraints = [g_eq],
		inequality_constraints = [g_ineq],
		shared_equality_constraint = nothing,
		shared_inequality_constraint = nothing,
	)

	GOOP_kkt_system = QuasiGOOP.generate_slacked_kkt_system(GOOP_trial1)
	parameter_value = [0, 0]
	(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
		QuasiGOOP.InteriorPoint(),
		GOOP_kkt_system,
		parameter_value;
		z₀ = nothing,
	)
	new_primals = x[1:n]
	new_objective = J₁(new_primals, 0)

	@test isapprox([0.5, 1.75, 0.25, 0.25], new_primals, atol = 1e-3)
	@test isapprox(2.4688, new_objective, atol = 1e-3)
end
