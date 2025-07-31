using QuasiGOOP

using LinearAlgebra: I, norm, pinv

using Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes


# Bilevel Equality-Constrained Quadratic Program (Toy Example)
# --------------------------------------------------------------
#
# Upper-level problem:
#   min_{x} (1/2)x' Q₁ x + c'x
#   subject to:
#       x ∈ argmin_{x} (1/2)x' Q₂ x + d' x
#                         subject to: A₂x = b₂
#
# --------------------------------------------------------------
# Cost and cross-term matrices
Q₁ = I(3)
c = [0.0, 0.0, 0.0]
# Lower-level problem
Q₂ = [1 0 0; 0 1 0; 0 0 0]
d = [-1.0, -1.0, -1.0]
A₂ = [1 0 1; 0 1 1] 
b₂ = [2.0, 2.0]

##### ORIGINAL GOOP VERSION ######

f(x, θ) = 0.5x[1:3]'*Q₁*x[1:3] + c'*x[1:3]

g(x, θ) = [
	Q₂ * x[1:3] .+ d - A₂'*x[4:5];
	A₂*x[1:3] .- b₂
]
h(x, θ) = []

problem = ParametricOptimizationProblem(;
	objective = f,
	equality_constraint = g,
	inequality_constraint = h,
	parameter_dimension = 1,
	primal_dimension = 5,
	equality_dimension = 5,
	inequality_dimension = 0,
)

(; primals, variables, status, info) = solve(problem, [0])
@show status
println("Primal solution: $primals")
println("Variables: $variables")
println("Objective: $(f(primals, 0))")

##### NEW VERSION ######

M = [Q₂ -A₂'; A₂ zeros(2, 2)]
∇ₓπ₂ = Q₂

quasi_f(x, θ) = 0


quasi_g(x, θ) = [
	Q₁*x[1:3] .+ c .- ∇ₓπ₂'*x[4:6] - A₂'*x[9:10]; #- [0;0;x[9]]; # ∇ₓL₁ = Q₁x + c - A₂'μ₁ - ∇ₓπ₂'ψ₁
	Q₂ * x[1:3] .+ d .- A₂'*x[7:8]; # π₂(x) = Q₂x + d - A₂'μ₂
	# x[7:8] .- [zeros(2, 3) I(2)] * (pinv(M) * [-d; b₂]); # μ₂ = [0 I] * M^-1 * ([-d; b₂])
	A₂ * x[1:3] .- b₂; # g₂ = A₂x - b₂ =  0
]

# # Symbolics Implementation to construct Bilevel QuasiKKTsystem
# upper_level_dimension = 2 # x
# lower_level_dimension = 2 # y
# upper_level_constraint_dimension = 2 # g₁, μ₁
# lower_level_constraint_dimension = 2 # g₂, μ₂	
# lower_level_transition_dimension = 2 # ψ₁
# total_dimension = upper_level_dimension + lower_level_dimension + upper_level_constraint_dimension + lower_level_transition_dimension + lower_level_constraint_dimension
# z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
# z = BlockArray(z̃, [upper_level_dimension, lower_level_dimension, upper_level_constraint_dimension, lower_level_transition_dimension, lower_level_constraint_dimension]) # x, y, μ₁, ψ₁, μ₂
# x = z[Block(1)]
# y = z[Block(2)]
# μ₁ = z[Block(3)]
# ψ₁ = z[Block(4)]
# μ₂ = z[Block(5)]

# J₁(x,y) = 0.5x'*Q₁*x + 0.5y'*R₁*y + x'*S*y + c'*x
# J₂(x,y) = 0.5y'*Q₂*y + d'*y
# g₁(x,y) = A₁ * x .+ B₁ * y .- b₁
# g₂(x,y) = A₂ * x .+ B₂ * y .- b₂

# # Build L₂
# L₂ = J₂(x,y) - μ₂' * g₂(x, y)

# # Build π₂(x,y)
# π_2 = Symbolics.gradient(L₂, y)

# # Build L₁
# L₁ = J₁(x,y) - μ₁' * g₁(x, y) - ψ₁' * π_2

# Main.@infiltrate
# equality_constraints = []
# append!(equality_constraints, Symbolics.gradient(L₁, x)) # ∇ₓL₁
# append!(equality_constraints, Symbolics.gradient(L₁, y)) # ∇yL₁
# append!(equality_constraints, g₁(x,y)) # g₁
# append!(equality_constraints, π_2) # π₂(x,y)
# #TODO μ₂

quasi_problem = ParametricOptimizationProblem(;
	objective = quasi_f,
	equality_constraint = quasi_g,
	inequality_constraint = h,
	parameter_dimension = 1,
	primal_dimension = 10, 
	equality_dimension = 8, 
	inequality_dimension = 0,
)

(; primals, variables, status, info) = solve(quasi_problem, [0])
@show status
println("QUASI Primal solution: $primals")
println("QUASI Variables: $variables")
println("QUASI Objective: $(f(primals, 0))")

