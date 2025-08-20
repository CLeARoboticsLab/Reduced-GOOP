using QuasiGOOP

using ParametricMCPs: ParametricMCPs

using LinearAlgebra: I, norm, pinv

using Symbolics
using SymbolicTracingUtils
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
Q₃ = -[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0] #3I(n)
c₃ = -[0.5, -0.5, 1.0, 0.0]
A₃ = [1 0 1 1; 0 1 1 0] # A₃x = b₃
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
@show status
println("Primal solution: $primals")
println("Variables: $variables")
println("Objective: $(f(primals, 0))")
println("# Primals: $(length(primals))")
println("# Equality constraints: $(length(g(primals, 0)))")
println("Total variables: $(length(variables))")

##### NEW VERSION ######
player = 1
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
		A₃ * x₁ - b₃; # g₂ = 0 (m)
	]
end

dummy_primals = zeros(n+2n+m+n+m+m)
dummy_parameters = [0.0]

J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]
J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n]
J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n]
g_eq(x, θ) = A₃*x[1:n] .- b₃
g_ineq(x, θ) = [x[1] - 0.5; x[2] - 0.5]

x = BlockArray(zeros(n), [n]) # single player
θ = BlockArray([0.0], [1])
goop_preferences = [[J₁, J₂, J₃]]
is_prioritized_constraint = [[false, false, true]]
equality_constraints = [g_eq]
inequality_constraints = [g_ineq]
shared_equality_constraint = nothing
shared_inequality_constraint = nothing

GOOP_trial1 = QuasiGOOP.ParametricGOOP(
	x,
	θ;
	preferences = goop_preferences,
	is_prioritized_constraint,
	equality_constraints,
	inequality_constraints,
	shared_equality_constraint,
	shared_inequality_constraint,
)

GOOP_kkt_system = QuasiGOOP.generate_slacked_kkt_system(GOOP_trial1)
parameter_value = θ
(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters) = QuasiGOOP.solve(
	QuasiGOOP.InteriorPoint(),
	GOOP_kkt_system,
	parameter_value;
	tol = 1e-6,
	min_stepsize = 1e-4,
	max_outer_iters = 50,
	z₀ = nothing,
	verbose = true,
)
@show status
println("v3 Primal solution: $(x)")
println("v3 Variables: $(z)")
println("v3 Objective: $(f(x[1:n], 0))")


## Cross check with original GOOP##
# θ is the relaxation factor
backend = SymbolicTracingUtils.SymbolicsBackend()
x = SymbolicTracingUtils.make_variables(backend, :x, n)
θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1))
symbolic_type = eltype(x)

# Keep track of all equality constraint duals (λ) that we create.
Λ = []

# Keep track of all inequality constraint duals (γ) that we create.
Γ = []
function construct_kkt(preferences, is_prioritized_constraint, player)
	level = 1 + length(goop_preferences[player]) - length(preferences)

	f = isnothing(equality_constraints[player]) ? nothing : equality_constraints[player](x, θ)
	g = isnothing(inequality_constraints[player]) ? nothing : inequality_constraints[player](x, θ)
	λ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("λ_$(player)_$(level)"),
		isnothing(f) ? 1 : length(f),
	)

	γ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("γ_$(player)_$(level)"),
		isnothing(g) ? 1 : length(g),
	)
	# Base level
	if length(preferences) == 1
		if only(is_prioritized_constraint) # For now, only the innermost is preference constraint.
			# Highest priority is a constraint.
			h = only(preferences)(x, θ)

			preference_slack = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("s_$(player)_$(level)"),
				length(h), # get the right dimension
			)

			γₚ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₚ_$(player)_$(level)"),
				length(h),
			)

			γₛ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₛ_$(player)_$(level)"),
				length(h),
			)

			L =
				sum(preference_slack) - γₚ' * (h .+ preference_slack) - γₛ' * preference_slack -
				(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)

			∇L = Symbolics.gradient(L, vcat(x, preference_slack))
			F = Vector{symbolic_type}([∇L; f])
			G = Vector{symbolic_type}(
				[
					g; γ; θ - γ'*g;
					γₚ; h .+ preference_slack; θ - γₚ'*(h .+ preference_slack);
					γₛ; preference_slack; θ - γₛ' * preference_slack
				],
			)
			return (; F, G, z = [x; preference_slack; λ; γ; γₚ; γₛ])
		else

			J = only(preferences)(x, θ)
			L = J - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
			∇L = Symbolics.gradient(L, x)
			F = Vector{symbolic_type}([∇L; f])
			G = Vector{symbolic_type}([g; γ; θ - γ'*g]) #ϵ - γ'*g
			return (; F, G, z = [x; λ; γ])
		end
	end

	# Recursive call for the next level.
	(; F, G, z) = construct_kkt(preferences[2:end], is_prioritized_constraint[2:end], player)

	J = first(preferences)(x, θ)

	λ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("λ_$(player)_$(level)"),
		length(F),
	)
	γ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("γ_$(player)_$(level)"),
		length(G),
	)

	L = J - λ'*F - γ'*G
	∇L = Symbolics.gradient(L, z)
	F̃ = Vector{symbolic_type}([∇L; F])
	G̃ = Vector{symbolic_type}([G; γ; θ - γ'*G])
	# return level == 1 ? (; F = F̃, G = G̃, z = z) : (; F = F̃, G = G̃, z = [z; λ; γ])
	return (; F = F̃, G = G̃, z = [z; λ; γ])
end

(; F, G, z) = construct_kkt(goop_preferences[player][2:end], is_prioritized_constraint[player][2:end], player)
λ = SymbolicTracingUtils.make_variables(
	backend,
	Symbol("λ_$(player)_1"),
	length(F),
)
γ = SymbolicTracingUtils.make_variables(
	backend,
	Symbol("γ_$(player)_1"),
	length(G),
)

# Topmost level
L = first(goop_preferences[player][1](z, θ)) - λ'*F - γ'*G
∇L = Symbolics.gradient(L, z)
F = Vector{symbolic_type}([∇L; F])
z̲ = [
	fill(-Inf, length(F))
	fill(0, length(G))
]
z̅ = [
	fill(Inf, length(F))
	fill(Inf, length(G))
]
parameter_value = [1e-5]
Main.@infiltrate
parametric_mcp = ParametricMCPs.ParametricMCP([F; G], [z; λ; γ], [θ], z̲, z̅; compute_sensitivities = false)
z_sol, status, info = ParametricMCPs.solve(
	parametric_mcp,
	parameter_value;
	initial_guess = zeros(length([z; λ; γ])),
	verbose = false,
	cumulative_iteration_limit = 100000,
	proximal_perturbation = 1e-2,
	# major_iteration_limit = 1000,
	# minor_iteration_limit = 2000,
	# nms_initial_reference_factor = 50,
	use_basics = true,
	use_start = true,
)
@show status
println("v2 Primal solution: $(z_sol[1:n])")
println("v2 Variables: $(z_sol)")
println("v2 Objective: $(f(z_sol[1:n], 0))")