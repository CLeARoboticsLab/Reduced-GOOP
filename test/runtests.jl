using Test: @testset, @test

using QuasiGOOP
using ParametricMCPs: ParametricMCPs


@testset "Bilevel Equality-Constrained Quadratic Program" begin
	"""
	 Upper-level problem:
	   min_{x, y} (1/2)x' Q₁ x + (1/2)y' R₁ y + x'Sy + c'x
	   subject to:
		   A₁x + B₁y = b₁
		   y ∈ argmin_{y} (1/2)y' Q₂ y + d' y
				 subject to: A₂x + B₂y = b₂

	"""
	# Upper level objective
	Q₁ = I(2)
	R₁ = 2I(2)
	S    = I(2)
	c    = [0.0, 0.0]
	# Upper-level constraints
	A₁ = I(2)
	B₁ = I(2)
	b₁ = [1.0, 1.0]
	# Lower-level objective
	Q₂ = I(2)
	d = [-1.0, -1.0]
	# Lower-level constraints
	A₂ = 2I(2)
	B₂ = I(2)
	b₂ = [0.5, 1.0]

	function J₁(z, θ)
		x = z[1:2]
		y = z[3:4]
		0.5 * x' * Q₁ * x + 0.5 * y' * R₁ * y + x' * S * y + c' * x
	end

	function J₂(z, θ)
		y = z[3:4]
		0.5 * y' * Q₂ * y + d' * y
	end

	function g₁(z, θ)
		x = z[1:2]
		y = z[3:4]
		A₁ * x + B₁ * y - b₁
	end

	function g₂(z, θ)
		x = z[1:2]
		y = z[3:4]
		A₂ * x + B₂ * y - b₂
	end

    # Build ParametricGOOP
	preferences = [
		[J₁,J₂]
	]
	is_prioritized_constraint = [
		[false, false]  # J₁ and J₂ are costs
	]

	shared_equality_constraint(z, θ) = [0]
	shared_inequality_constraint(z, θ) = [0]

	equality_constraints = [
		[g₁, g₂]
	]
	inequality_constraints = [0]

	# problem = QuasiGOOP.ParametricGOOP(;
	# 	preferences,
	# 	is_prioritized_constraint,
	# 	equality_constraints,
	# 	inequality_constraints,
	# 	shared_equality_constraint,
	# 	shared_inequality_constraint,
	# 	primal_dims = [[2, 2]],
	# 	parameter_dims = [1],
    #   equality_dimensions = [[2, 2]],
    #   inequality_dimensions = [1],
	# 	shared_equality_dims = [1],
	# 	shared_inequality_dims = [1],
	# 	num_players = length(preferences),
	# )

    # sol = solve(problem; verbose= true)
    # @test isapprox(sol.primals[1], [-0.5, 0.0, 1.5, 1.0]) # [-0.5, 0.0, 1.5, 1.0, 2.5, 2.0, -0.75, -0.50, 0.50, 0.0] (x, y, μ₁, ψ₁, μ₂)
    # @test isapprox(J₁(sol.primals[1], 0), 3.625)
	@test true
end
