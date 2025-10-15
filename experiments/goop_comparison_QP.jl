using QuasiGOOP

include("old_goop.jl")

function rand_psd(n, r)
	# n: primal dimension, r: matrix rank (<=n)
	R = randn(r, n);
	R' * R;
end

# Trilevel Quadratic Program (Toy Example)
# --------------------------------------------------------------
# Qᵢ : Positive semi-definite matrices
# Upper-level problem:
#   min_{x} (1/2)x' Q₁ x + c₁'x
#   subject to:
#       x ∈ argmin_{x} (1/2)x' Q₂ x + c₂' x
#                         subject to: x ∈ argmin_{x} (1/2)x' Q₃ x + c₃' x
#                                                       Aₑx = bₑ
#                                                       Aᵢx ≥ bᵢ
# --------------------------------------------------------------
n = 4 # primal (x) dimension
m = 2 # equality dimension
n_digits = 4
backend = SymbolicTracingUtils.SymbolicsBackend()
# n = 10; m = 4;       

# Problem data
player = 1
Q₁ = I(n)
c₁ = [1.0, 1.0, 1.0, 1.0]
Q₂ = [0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
c₂ = [1.0, 1.0, 1.0, 1.0]
Q₃ = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]
c₃ = [1.0, 1.0, 1.0, 1.0]
A₃ = [1 0 1 0; 1 1 0 1] # A₃x = b₃
b₃ = [1.0, 1.0]

# Randomize Q, A and b (Q_i has to be positive semi-definite)
Q₁ = rand_psd(n, 1); c₁ = rand(n); 
Q₂ = rand_psd(n, 1); c₂ = rand(n);
Q₃ = rand_psd(n, 1); c₃ = rand(n);
Aₑ = rand(m, n); bₑ = rand(m);
Aᵢ = rand(m, n); bᵢ = rand(m);

J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]
J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n]
J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n]
# g_eq(x, θ) = Aₑ*x[1:n] .- bₑ
# g_ineq(x, θ) = Aᵢ*x[1:n] .- bᵢ
# g_eq(x, θ) = [x[1] + x[3] + x[4] - 1.0; x[1] + x[4] - 1.0]
g_eq(x, θ) = A₃*x[1:n] .- b₃
g_ineq(x, θ) = [x[1] - 0.5; x[2] - 0.5]

################# NEW GOOP #########################
@info "........................STARTING NEW GOOP........................"

x = BlockArray(zeros(n), [n]) # single player
parameters = BlockArray([0.0], [1])
goop_preferences = [[J₂, J₃]] # single player
is_prioritized_constraint = [[false, false]]
equality_constraints = [g_eq]
inequality_constraints = [g_ineq] #[g_ineq] # nothing
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
status, z_sol_new_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = QuasiGOOP.solve(
	QuasiGOOP.InteriorPoint(),
	GOOP_kkt_system,
	parameters;
	tol = 1e-5,
	η₀ = 0.0, # no regularization
	min_stepsize = 1e-4,
	max_outer_iters = 50,
	z₀ = nothing,
	verbose = true,
)
@show status
println("[New G] Primal solution: $(round.(z_sol_new_goop[1:n], digits = n_digits))")
println("[New G] Dual solution ($(length(z_sol_new_goop) - n) variables): $(round.(z_sol_new_goop[Not(1:n)], digits = n_digits))")
println("[New G] Objective: $(round(J₁(z_sol_new_goop[1:n], 0), digits = n_digits))")
println("[New G] number of equations: $(GOOP_kkt_system.kkt_dimension)")

################# OLD GOOP #########################
@info "........................STARTING OLD GOOP........................"

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

(; F, z) = construct_old_goop_kkt(goop_preferences[player][2:end], is_prioritized_constraint[player][2:end], player)

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

status, z_sol_old_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = QuasiGOOP.solve(
	QuasiGOOP.InteriorPoint(),
	OG_kkt_system,
	parameters;
	tol = 1e-5,
	η₀ = 0.0, # no regularization
	min_stepsize = 1e-4,
	max_outer_iters = 50,
	z₀ = nothing,
	verbose = true,
)
@show status
println("[Old G] Primal solution: $(round.(z_sol_old_goop[1:n], digits = n_digits))")
println("[Old G] Dual solution ($(length(z_sol_old_goop) - n) variables): $(round.(z_sol_old_goop[Not(1:n)], digits = n_digits))")
println("[Old G] Objective: $(round(J₁(z_sol_old_goop[1:n], 0), digits = n_digits))")
println("[Old G] number of equations: $(OG_kkt_system.kkt_dimension)")

# Reprint new goop
@show status
println("[New G] Primal solution: $(round.(z_sol_new_goop[1:n], digits = n_digits))")
println("[New G] Dual solution ($(length(z_sol_new_goop) - n) variables): $(round.(z_sol_new_goop[Not(1:n)], digits = n_digits))")
println("[New G] Objective: $(round(J₁(z_sol_new_goop[1:n], 0), digits = n_digits))")
println("[New G] number of equations: $(GOOP_kkt_system.kkt_dimension)")

# Final output
primal_new = z_sol_new_goop[1:n]
primal_old = z_sol_old_goop[1:n]
objective_diff = J₁(primal_new, 0) - J₁(primal_old, 0)
primal_diff = primal_new .- primal_old
println("Objective difference (new - old): $(round.(objective_diff, digits = n_digits))")
println("Primal solution difference (new - old): $(round.(primal_diff, digits = n_digits))")
println("Maximum absolute primal difference: $(round(maximum(abs.(primal_diff)), digits = n_digits))")

