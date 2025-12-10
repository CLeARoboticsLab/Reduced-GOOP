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
Q₂ = I(n) # [0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
c₂ = [1.0, 1.0, 1.0, 1.0]
Q₃ = I(n) # [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]
c₃ = [1.0, 1.0, 1.0, 1.0]
# Aₑ = [1 0 1 0; 1 1 0 1] # A₃x = b₃
# bₑ = [1.0, 1.0]

# Randomize Q, A and b (Q_i has to be positive semi-definite)
Random.seed!(1) # 17
#10 (problem is infeasble or unbounded)
#16 (both version in new goop same sol)
#17 (modified new goop gives wrong)
#18,19,20 (old goop wrong)
Q₁ = rand_psd(n, 1);
c₁ = rand(n);
Q₂ = rand_psd(n, 1);
c₂ = rand(n);
Q₃ = rand_psd(n, 1);
c₃ = rand(n);
Aₑ = rand(m, n);
bₑ = rand(m);
Aᵢ = [0 0 1 0; 0 0 0 1];
bᵢ = [0.5; 0.0]; 

# # Pretty-print randomized problem data in a compact table.
# println("Randomized problem data:")
# data = [
# 	"Q₁" => Q₁,
# 	"c₁" => c₁,
# 	"Q₂" => Q₂,
# 	"c₂" => c₂,
# 	"Q₃" => Q₃,
# 	"c₃" => c₃,
# 	"Aₑ" => Aₑ,
# 	"bₑ" => bₑ,
# 	"Aᵢ" => Aᵢ,
# 	"bᵢ" => bᵢ,
# ]
# name_width = maximum(length(name) for (name, _) in data)
# println("  ", rpad("Name", name_width), " | Value")
# println("  ", repeat("-", name_width), "-+-", repeat("-", 40))
# for (name, value) in data
# 	rounded_value = round.(value; digits = n_digits)
# 	value_str = sprint(io -> show(io, "text/plain", rounded_value))
# 	value_lines = split(value_str, '\n'; keepempty = false)
# 	first_line = isempty(value_lines) ? "" : value_lines[1]
# 	println("  ", rpad(name, name_width), " | ", first_line)
# 	for line in value_lines[2:end]
# 		println("  ", repeat(" ", name_width), " | ", line)
# 	end
# 	println("  ", repeat("-", name_width), "-+-", repeat("-", 40))
# end

# # Check column space inclusion
# println("column space inclusion: ", colspace_issubset(Aᵢ', hcat(Aₑ', Q₃)))
# A = [
# 	Q₃ Aᵢ'; 
# 	zeros(m, n+m)
# 	]
# B = [
# 	Q₃ Aᵢ'; 
# 	Aᵢ zeros(m, m)
# 	]
# println("column space inclusion ( NG ⊆ OG ): ", colspace_issubset(A, B))
# if !colspace_issubset(A, B)
#     error("Column space inclusion check failed: NG ⊈ OG")
# end

J₁(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]
J₂(x, θ) = 0.5x[1:n]'*Q₂*x[1:n] + c₂'*x[1:n]
J₃(x, θ) = 0.5x[1:n]'*Q₃*x[1:n] + c₃'*x[1:n]
g_eq(x, θ) = Aₑ*x[1:n] .- bₑ
# g_ineq(x, θ) = Aᵢ*x[1:n] .- bᵢ
# g_eq(x, θ) = [x[1] + x[2]+ x[3] + x[4] - 2.0]
# g_eq(x, θ) = A₃*x[1:n] .- b₃
# g_ineq(x, θ) = [x[1] - 0.5; x[2] - 0.5]

# nonlinear equality constraint 
# g_eq(x, θ) = [x[1]^4 + x[2]^4 - 1.0]
g_eq(x, θ) = [x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4 - 1.0]

# initial warmstart
warmstart_x = [0.7613, 0.5384, 0.7484, 0.7185]

################# NEW GOOP #########################
@info "........................STARTING NEW GOOP........................"

x = BlockArray(zeros(n), [n]) # single player
parameters = BlockArray([0.0], [1])
goop_preferences = [[J₁, J₂, J₃]] # single player
is_prioritized_constraint = [[false, false, false]]
equality_constraints = [g_eq]
inequality_constraints = [nothing] #[g_ineq] # [nothing]
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

NG_kkt_system = QuasiGOOP.generate_slacked_kkt_system(GOOP_trial1)
status_new_goop, z_sol_new_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = QuasiGOOP.solve(
	QuasiGOOP.InteriorPoint(),
	NG_kkt_system,
	parameters;
	tol = 1e-5,
	η₀ = 0.0, # no regularization
	min_stepsize = 1e-5,
	max_outer_iters = 200,
	z₀ = warmstart_x, #[0.5032, 2.1336, -3.1896, 0.3465], # nothing
	verbose = true,
)

println("[New G] status = $(status_new_goop)")
println("[New G] Primal solution: $(round.(z_sol_new_goop[1:n], digits = n_digits))")
println("[New G] Dual solution ($(length(z_sol_new_goop) - n) variables): $(round.(z_sol_new_goop[Not(1:n)], digits = n_digits))")
println("[New G] Objective: $(round(J₁(z_sol_new_goop[1:n], 0), digits = n_digits))")
println("[New G] number of equations: $(NG_kkt_system.kkt_dimension)")

Main.@infiltrate

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

(; F, z) = construct_kkt_old_goop(goop_preferences[player][2:end], is_prioritized_constraint[player][2:end], player)

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
	vcat(x, s, Λ, Σ, Γ),
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

status_old_goop, z_sol_old_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = QuasiGOOP.solve(
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
println("[Old G] status = $(status_old_goop)")
println("[Old G] Primal solution: $(round.(z_sol_old_goop[1:n], digits = n_digits))")
println("[Old G] Dual solution ($(length(z_sol_old_goop) - n) variables): $(round.(z_sol_old_goop[Not(1:n)], digits = n_digits))")
println("[Old G] Objective: $(round(J₁(z_sol_old_goop[1:n], 0), digits = n_digits))")
println("[Old G] number of equations: $(OG_kkt_system.kkt_dimension)")

################# OLDER GOOP via PATH #########################
x = SymbolicTracingUtils.make_variables(backend, :x, n)
θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1)) # this is relaxation here
symbolic_type = eltype(x)

(; F, G, z) = construct_kkt_older_goop(goop_preferences[player][2:end], is_prioritized_constraint[player][2:end], player)

λ = SymbolicTracingUtils.make_variables(
	backend,
	Symbol("λ_$(player)_1"),
	length(F), # equality dimension
)
γ = SymbolicTracingUtils.make_variables(
	backend,
	Symbol("γ_$(player)_1"),
	length(G), # inequality_dimension
)

# Topmost level
L = first(goop_preferences[player][1](z, θ)) - λ'*F - γ'*G
∇L = Symbolics.gradient(L, z)
Main.@infiltrate
F = Vector{symbolic_type}([∇L; F])
z̲ = [
	fill(-Inf, length(F))
	fill(0, length(G))
]
z̅ = [
	fill(Inf, length(F))
	fill(Inf, length(G))
]
Main.@infiltrate
parameter_value = [1e-4]
parametric_mcp = ParametricMCPs.ParametricMCP([F; G], [z; λ; γ], [θ], z̲, z̅; compute_sensitivities = false)
z_sol_older_goop, status_older_goop, info = ParametricMCPs.solve(
	parametric_mcp,
	parameter_value;
	initial_guess = vcat(warmstart_x, zeros(length([z;λ;γ])-n)),#zeros(length([z; λ; γ])),
	verbose = false,
	cumulative_iteration_limit = 100000,
	proximal_perturbation = 1e-2,
	# major_iteration_limit = 1000,
	# minor_iteration_limit = 2000,
	# nms_initial_reference_factor = 50,
	use_basics = true,
	use_start = true,
)
@show status_older_goop
println("[Older G via PATH] status = $(status_older_goop)")
println("[Older G via PATH] Primal solution: $(round.(z_sol_older_goop[1:n], digits = n_digits))")
println("[Older G via PATH] Dual solution ($(length(z_sol_older_goop) - n) variables): $(round.(z_sol_older_goop[Not(1:n)], digits = n_digits))")
println("[Older G via PATH] Objective: $(round(J₁(z_sol_older_goop[1:n], 0), digits = n_digits))")
println("[Older G via PATH] number of equations: $(length(F))")
println("[Older G via PATH] number of inequalities: $(length(G))")

# Check if the status is :solved
if "$(status_older_goop)" != "MCP_Solved"
	error("Older GOOP via PATH did not solve successfully. Status: $status_older_goop")
end

##################### SUMMARY ########################
# Reprint new goop
println("[New G] status = $(status_new_goop)")
println("[New G] Primal solution: $(round.(z_sol_new_goop[1:n], digits = n_digits))")
println("[New G] Dual solution ($(length(z_sol_new_goop) - n) variables): $(round.(z_sol_new_goop[Not(1:n)], digits = n_digits))")
println("[New G] Objective: $(round(J₁(z_sol_new_goop[1:n], 0), digits = n_digits))")
println("[New G] number of equations: $(NG_kkt_system.kkt_dimension)")

# Compare new goop and older goop
primal_new = z_sol_new_goop[1:n]
primal_older = z_sol_older_goop[1:n]
objective_new = J₁(primal_new, 0)
objective_older = J₁(primal_older, 0)
objective_diff = objective_new - objective_older
primal_diff = primal_new .- primal_older
println("Objective difference (new - older): $(round.(objective_diff, digits = n_digits))")
println("Primal solution difference (new - older): $(round.(primal_diff, digits = n_digits))")
max_primal_diff = maximum(abs, primal_diff)
println("Maximum absolute primal difference: $(round(max_primal_diff, digits = n_digits))")
abs_objective_diff = abs(objective_diff)
primal_tol = 5e-3
objective_tol = 5e-3
if max_primal_diff > primal_tol || abs_objective_diff > objective_tol
	error(
		"New G and Older G solutions diverge: max Δx = $(max_primal_diff), ΔJ = $(abs_objective_diff); tolerances are $(primal_tol) for primal and $(objective_tol) for objective.",
	)
else
	println(
		"New G and Older G solutions match within tolerances (max Δx = $(round(max_primal_diff, digits = n_digits)), ΔJ = $(round(abs_objective_diff, digits = n_digits))).",
	)
end

Main.@infiltrate

# for (i, expr) in enumerate(NG_kkt_system.F_symbolic)
#            println("F[$i] = $expr")
# end

# for (i, zi) in enumerate(NG_kkt_system.z_symbolic)
# 		   println("z[$i] = $zi")
# end