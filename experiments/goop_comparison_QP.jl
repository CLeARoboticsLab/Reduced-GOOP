using QuasiGOOP

include("old_goop.jl")

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
backend = SymbolicTracingUtils.SymbolicsBackend()

# Problem data
player = 1
Q₁ = I(n)
c₁ = [1.0, 0.0, -1.0, 2.0]
Q₂ = 2I(n) # [0 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 1]
c₂ = [-1.0, 2.0, 0.0, 1.0]
Q₃ = 3I(n)# -[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]
c₃ = -[0.5, -0.5, 1.0, 0.0]
A₃ = [1 0 1 1; 1 0 0 1] # A₃x = b₃
b₃ = [1.0, 1.0]


# f(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]

J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]
J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n]
J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n]
g_eq(x, θ) = A₃*x[1:n] .- b₃
g_ineq(x, θ) = [x[1] - 0.5; x[2] - 0.5]

################# NEW GOOP #########################

x = BlockArray(zeros(n), [n]) # single player
parameters = BlockArray([0.0], [1])
goop_preferences = [[J₁, J₂, J₃]]
is_prioritized_constraint = [[false, false, false]]
equality_constraints = [g_eq]
inequality_constraints = [g_ineq] #[g_ineq] # [g_ineq]
shared_equality_constraint = nothing
shared_inequality_constraint = nothing

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

GOOP_kkt_system = QuasiGOOP.generate_slacked_kkt_system(GOOP_trial1)

Main.@infiltrate

status, z_sol_new_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = QuasiGOOP.solve(
	QuasiGOOP.InteriorPoint(),
	GOOP_kkt_system,
	parameters;
	tol = 1e-6,
	η₀ = 0.0, # no regularization
	min_stepsize = 1e-4,
	max_outer_iters = 50,
	z₀ = nothing,
	verbose = true,
)
@show status
println("[New G] Primal solution: $(z_sol_new_goop[1:n])")
println("[New G] Dual solution ($(length(z_sol_new_goop) - n) variables): $(z_sol_new_goop[Not(1:n)])")
println("[New G] Objective: $(J₁(z_sol_new_goop[1:n], 0))")

Main.@infiltrate


################# OLD GOOP #########################
x = SymbolicTracingUtils.make_variables(backend, :x, n)
θ = SymbolicTracingUtils.make_variables(backend, :θ, n)
ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))
symbolic_type = eltype(x)

# Keep track of all equality constraint duals (λ) that we create.
Λ = symbolic_type[]

# Keep track of all inequality constraint duals (γ) that we create.
Γ = symbolic_type[]

# Keep track of all preference slack (s) and interior point slack variables (σ) that we create.
s = symbolic_type[]
Σ = symbolic_type[]

# Construct F (equalities), G (inequalities) and z (variables) for the topmost level after recursion.
function construct_kkt(preferences, is_prioritized_constraint, player)
	level = 1 + length(goop_preferences[player]) - length(preferences)

	# Base level
	if length(preferences) == 1

		f = isnothing(equality_constraints[player]) ? nothing : equality_constraints[player](x, θ)
		g = isnothing(inequality_constraints[player]) ? nothing : inequality_constraints[player](x, θ)
		λ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("λ_$(player)_$(level)"),
			isnothing(f) ? 0 : length(f),
		)
		push!(Λ, λ...)

		γ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("γ_$(player)_$(level)"),
			isnothing(g) ? 0 : length(g),
		)
		push!(Γ, γ...)

		σ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("σ_$(player)_$(level)"),
			isnothing(g) ? 0 : length(g),
		)
		push!(Σ, σ...)

		if only(is_prioritized_constraint) # TODO for old GOOP
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
				filter!(
					!isnothing,
					[
						isnothing(g) ? nothing : g;
						isnothing(g) ? nothing : γ;
						isnothing(g) ? nothing : θ - γ'*g;
						γₚ; h .+ preference_slack; θ - γₚ'*(h .+ preference_slack);
						γₛ; preference_slack; θ - γₛ' * preference_slack
					],
				),
			)
			z = Vector{symbolic_type}(
				vcat(x, preference_slack, (isnothing(f) ? [] : λ), (isnothing(g) ? [] : γ), γₚ, γₛ),
			)
			return (; F, G, z)
		else
			J = only(preferences)(x, θ)
			L = J - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
			∇L = Symbolics.gradient(L, x)
			F = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L;
						f;
						isnothing(g) ? nothing : g .- σ;
						isnothing(g) ? nothing : σ .* γ .- ϵ;
					],
				),
			)
			z = Vector{symbolic_type}(
				vcat(x, (isnothing(f) ? [] : λ), (isnothing(g) ? [] : γ)),
			)
			return (; F, z)
		end
	end

	# Recursive call for the next level.
	(; F, z) = construct_kkt(preferences[2:end], is_prioritized_constraint[2:end], player)

	J = first(preferences)(x, θ)
	λ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("λ_$(player)_$(level)"),
		length(F),
	)
	push!(Λ, λ...)

	L = J - λ'*F
	∇L = Symbolics.gradient(L, z)
	F̃ = Vector{symbolic_type}([∇L; F])

	(; F = F̃, z = [z; λ])
end

(; F, z) = construct_kkt(goop_preferences[player][2:end], is_prioritized_constraint[player][2:end], player)
Main.@infiltrate

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
	vcat(x, s, Σ, Λ, Γ),
)

# Set up common memory
variable_dimension = length(z)
kkt_dimension = length(F)
idx = blockedrange(length.([x, s, Λ, Σ, Γ]))
primal_dims = idx[Block(1)]
preference_slack_dims = idx[Block(2)] # s
interior_point_slack_dims = idx[Block(4)] # Σ
inequality_constraint_dual_dims = idx[Block(5)] # Γ

Main.@infiltrate

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

Main.@infiltrate


status, z_sol_old_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = QuasiGOOP.solve(
	QuasiGOOP.InteriorPoint(),
	OG_kkt_system,
	parameters;
	tol = 1e-6,
	η₀ = 0.0, # no regularization
	min_stepsize = 1e-4,
	max_outer_iters = 50,
	z₀ = nothing,
	verbose = true,
)
@show status
println("[Old G] Primal solution: $(z_sol_old_goop[1:n])")
println("[Old G] Dual solution ($(length(z_sol_old_goop) - n) variables): $(z_sol_old_goop[Not(1:n)])")
println("[Old G] Objective: $(J₁(z_sol_old_goop[1:n], 0))")
