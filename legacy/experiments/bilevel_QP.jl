using QuasiGOOP

using LinearAlgebra: I, norm

# Bilevel Equality-Constrained Quadratic Program (Toy Example)
# --------------------------------------------------------------
#
# Upper-level problem:
#   min_{x, y} (1/2)x' Q₁ x + (1/2)y' R₁ y + x'Sy + c'x
#   subject to:
#       A₁x + B₁y = b₁
#       y ∈ argmin_{y} (1/2)y' Q₂ y + d' y
#                         subject to: A₂x + B₂y = b₂
#
# --------------------------------------------------------------
# Cost and cross-term matrices
Q₁ = I(2)
R₁ = 2I(2)
S    = I(2)
c    = [0.0, 0.0]
# Upper-level constraints
A₁ = I(2)
B₁ = I(2)
b₁ = [1.0, 1.0]
# Lower-level problem
Q₂ = I(2)
d = [-1.0, -1.0]
A₂ = 2I(2)
B₂ = I(2)
b₂ = [0.5, 1.0]

##### ORIGINAL GOOP VERSION ######

f(x, θ) = 0.5x[1:2]'*Q₁*x[1:2] + 0.5x[3:4]'*R₁*x[3:4] + x[1:2]'*S*x[1:2] + c'*x[1:2]
g(x, θ) = [
	A₁*x[1:2] .+ B₁*x[3:4] .- b₁;
	Q₁*x[3:4] .+ d - B₂'*x[5:6];
	A₂*x[1:2] .+ B₂*x[3:4] .- b₂
]
h(x, θ) = []

problem = ParametricOptimizationProblem(;
	objective = f,
	equality_constraint = g,
	inequality_constraint = h,
	parameter_dimension = 1,
	primal_dimension = 6,
	equality_dimension = 6,
	inequality_dimension = 0,
)

(; primals, variables, status, info) = solve(problem, [0])
@show status
println("Primal solution: $primals")
println("Variables: $variables")
println("Objective: $(f(primals, 0))")

##### QUASI VERSION ######
M = [Q₂ -B₂'; B₂ zeros(2, 2)]
∇ₓπ₂ = - [zeros(2,2) -B₂'] * (M \ [zeros(2, 2); A₂])
∇yπ₂ = Q₂ - B₂' * (-[zeros(2, 2) I(2)] * (M \ [zeros(2, 2); A₂])) / (-[I(2) zeros(2, 2)] * (M \ [zeros(2, 2); A₂]))

quasi_f(x, θ) = 0
quasi_g(x,θ) = [
	Q₁*x[1:2] .+ S*x[3:4] .+ c .- A₁'*x[5:6] .- ∇ₓπ₂'*x[7:8]; # ∇ₓL₁ = Q₁x + Sy + c - A₁'μ₁ - ∇ₓπ₂'ψ₁
	R₁ * x[3:4] .+ S'*x[1:2] .- B₁'*x[5:6] .- ∇yπ₂'*x[7:8]; # ∇yL₁ = R₁y + S'x - B₁'μ₁ - ∇yπ₂'ψ₁
	A₁ * x[1:2] .+ B₁*x[3:4] .- b₁; # g₁ = A₁x + B₁y - b₁ = 0
	Q₂ * x[3:4] .+ d .- B₂'*x[9:10]; # π₂(x,y) = Q₂y + d - B₂'μ₂
	x[9:10] .- [zeros(2, 2) I(2)] * (M \ ([-d; b₂] .- [zeros(2, 2); A₂] * x[1:2])); # μ₂ = [0 I] * M \ ([-d; b₂] - [0; A₂]x)
]

quasi_problem = ParametricOptimizationProblem(;
	objective = quasi_f,
	equality_constraint = quasi_g,
	inequality_constraint = h,
	parameter_dimension = 1,
	primal_dimension = 10,
	equality_dimension = 10,
	inequality_dimension = 0,
)

(; primals, variables, status, info) = solve(quasi_problem, [0])
@show status
println("QUASI Primal solution: $primals")
println("QUASI Variables: $variables")
println("QUASI Objective: $(f(primals, 0))")

