using QuasiGOOP

using LinearAlgebra: I, norm, pinv

using Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes


# Trilevel Equality-Constrained Quadratic Program (Toy Example)
# --------------------------------------------------------------
#
# Upper-level problem:
#   min_{x} (1/2)x' Q₁ x + c₁'x
#   subject to:
#       x ∈ argmin_{x} (1/2)x' Q₂ x + c₂' x
#                         subject to: x ∈ argmin_{x} (1/2)x' Q₃ x + c₃' x
#                                                       A₃x = b₃
#
# --------------------------------------------------------------
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

problem = ParametricOptimizationProblem(;
	objective = f,
	equality_constraint = g,
	inequality_constraint = h,
	parameter_dimension = 1,
	primal_dimension = length(dummy_primals),
	equality_dimension = length(g(dummy_primals, dummy_parameters)),
	inequality_dimension = 0,
)

(; primals, variables, status, info) = solve(problem, [0])
@show status
println("Primal solution: $primals")
println("Variables: $variables")
println("Objective: $(f(primals, 0))")
println("# Primals: $(length(primals))")
println("# Equality constraints: $(length(g(primals, 0)))")
println("Total variables: $(length(variables))")

##### NEW VERSION ######

M₃ = [Q₂ -A₃'; A₃ zeros(m, m)]
∇ₓπ₃ = Q₂
∇ₓπ₂ = [Q₂; Q₃]
M₂ = [Q₂ -Q₃' -A₃'; Q₃ zeros(n, n) zeros(n, m); A₃ zeros(m, n) zeros(m, m)]

new_f(x, θ) = 0
new_g = function (x, θ)
	x₁ = x[1:n]
	ψ₁ = x[(n+1):(n+2n)]
	μ₁ = x[(n+2n+1):(n+2n+m)]
	ψ₂ = x[(n+2n+m+1):(n+2n+m+n)]
	μ₂ = x[(n+2n+m+n+1):(n+2n+m+n+m)]
	μ₃ = x[(n+2n+m+n+m+1):(n+2n+m+n+m+m)]

	[
		Q₁*x₁ + c₁ - ∇ₓπ₂'*ψ₁ - A₃'*μ₁; # ∇ₓL₁ (n)
		Q₂*x₁ + c₂ - ∇ₓπ₃'*ψ₂ - A₃'*μ₂; # π₂(x) = [Q₂x + c₂ - ∇ₓπ₃(x)'ψ₂ - A₃'μ₂; Q₃x + c₃ - A₃'μ₃] (n)
		Q₃*x₁ + c₃ - A₃'*μ₃; #- [0; x[23]; 0 ; x[24]]; # (n)
		# [ψ₂; μ₂] - [zeros(n, n) I(n) zeros(n, m); zeros(m, n) zeros(m, n) I(m)] * (pinv(M₂) * [-c₂; A₃'*μ₃ - c₃; b₃]) # (n+m)
		# μ₃ - [zeros(m, n) I(m)] * (pinv(M₃) * [-c₃; b₃]); # μ₃ = [0 I] * M₃^-1 * ([-c₃; b₃]) (m)
		A₃ * x₁ - b₃; # g₂ = 0 (m)
	]
end

dummy_primals = zeros(n+2n+m+n+m+m)
dummy_parameters = [0.0]

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
	objective = new_f,
	equality_constraint = new_g,
	inequality_constraint = h,
	parameter_dimension = 1,
	primal_dimension = length(dummy_primals),
	equality_dimension = length(new_g(dummy_primals, dummy_parameters)),
	inequality_dimension = 0,
)

(; primals, variables, status, info) = solve(quasi_problem, [0])
@show status
println("v2 Primal solution: $primals")
println("v2 Variables: $variables")
println("v2 Objective: $(f(primals, 0))")
println("# Primals: $(length(primals))")