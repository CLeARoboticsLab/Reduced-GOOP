using ReducedGOOP

""" Trilevel Equality-Constrained Quadratic Program (Toy Example)
--------------------------------------------------------------

 Upper-level problem:
   min_{x} (1/2)x' Q₁ x + c₁'x
   subject to:#       x ∈ argmin_{x} (1/2)x' Q₂ x + c₂' x
                         subject to: x ∈ argmin_{x} (1/2)x' Q₃ x + c₃' x
                                                       A₃x = b₃

 --------------------------------------------------------------
 """
n = 4 # x dimension
m = 2 # equality dimension
backend = SymbolicTracingUtils.SymbolicsBackend()

# Problem data
Q₁ = I(n)
c₁ = [1.0, 0.0, -1.0, 2.0]
Q₂ = I(n) # [0 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 1]
c₂ = [-1.0, 2.0, 0.0, 1.0]
Q₃ = -[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
c₃ = -[0.5, -0.5, 1.0, 0.0]
A₃ = [1 0 1 1; 1 0 0 1] # A₃x = b₃
b₃ = [1.0, 1.0]

# Random.seed!(17)
#16 (both version in new goop same sol)
#17 (modified new goop gives wrong)
#18,19,20 (old goop wrong)
# Q₁ = rand_psd(n, 1);
# c₁ = rand(n);
# Q₂ = rand_psd(n, 1);
# c₂ = rand(n);
# Q₃ = rand_psd(n, 1);
# c₃ = rand(n);
# Aₑ = rand(m, n);
# bₑ = rand(m);
Aᵢ = [1 0 0 0; 0 1 0 0];
bᵢ = [0.5, 0.5] # x₁ ≥ 0.5, x₂ ≥ 0.5
f(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]


Q₁ = I(n)
c₁ = [1.0, 0.0, -1.0, 2.0]
Q₂ = 2I(n) # [0 0 0 0; 0 1 0 0; 0 0 2 0; 0 0 0 1]
c₂ = [-1.0, 2.0, 0.0, 1.0]
Q₃ = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0] # I(n)
c₃ = [0.5, -0.5, 1.0, 0.0]
Aₑ = [1 0 1 1; 0 1 1 0]
bₑ = [1.0, 2.0]
f(x, θ) = 0.5x[1:n]'*Q₁*x[1:n] + c₁'*x[1:n]


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
# g_eq(x, θ) = A₃*x[1:n] .- b₃
g_eq(x, θ) = Aₑ*x[1:n] .- bₑ
g_ineq(x, θ) = [x[1] - 0.5; x[2] - 0.5]

x = BlockArray(zeros(n), [n]) # single player
θ = BlockArray([0.0], [1])
goop_preferences = [[J₁, J₂, J₃]]
is_prioritized_constraint = [[false, false, false]]
equality_constraints = [g_eq]
inequality_constraints = [g_ineq] #[g_ineq] # [g_ineq]
shared_equality_constraint = nothing
shared_inequality_constraint = nothing

GOOP_trial1 = ReducedGOOP.ParametricGOOP(
	x,
	θ;
	preferences = goop_preferences,
	is_prioritized_constraint,
	equality_constraints,
	inequality_constraints,
	shared_equality_constraint,
	shared_inequality_constraint,
)

GOOP_kkt_system = ReducedGOOP.generate_slacked_complete_kkt_system(GOOP_trial1)
parameter_value = θ
solve_options = ReducedGOOP.InteriorPointOptions(;
	tol = 1e-3,
	η₀ = 1e-4,
	max_inner_iters = 20,
	min_stepsize = 1e-4,
	max_outer_iters = 50,
	ϵ₀ = 1e-5,
	tightening_rate = 0.1,
	loosening_rate = 0.5,
	linesearch = :backtracking,
	verbose = true,
)
status, z_sol_new_goop, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters = ReducedGOOP.solve(
	ReducedGOOP.InteriorPoint(),
	GOOP_kkt_system,
	parameter_value;
	z₀ = zeros(n),
	options = solve_options,
)
@show status
println("Primal solution: $(z_sol_new_goop[1:n])")
println("Duals from new goop: $(z_sol_new_goop[Not(1:n)])")
println("Objective: $(f(z_sol_new_goop[1:n], 0))")
# z_sol_new_goop = z

## Check if duals can be found when primals are given using NonlinearSolve
x = SymbolicTracingUtils.make_variables(backend, :x, n)
ϵ = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
η = only(SymbolicTracingUtils.make_variables(backend, :η, 1))
F_symbolic = GOOP_kkt_system.F_symbolic
z_symbolic = GOOP_kkt_system.z_symbolic
symbolic_type = eltype(x)
F_symbolic_after_sub = Vector{symbolic_type}(
	Symbolics.substitute(F_symbolic, Dict([x[1] => 0.5, x[2] => 1.0, x[3] => 1.0, x[4] => -0.5, ϵ => 0, η => 0])),
)

# Compile the numeric function (returns F given z)
F_eval = first(Symbolics.build_function(F_symbolic_after_sub, z_symbolic[Not(1:n)]; expression = Val(false)))
# Wrap it to match NonlinearSolve’s expected signature f(u, p)
test_f(u, p) = F_eval(u)

# Initial guess and parameters
z_val = zeros(length(z_symbolic) - n) #z[Not(1:n)] .+ 1e-2*rand() # z[Not(1:n)] # sanity check #zeros(length(z_symbolic) - n)

# Construct the problem (3 positional args max) and solve it
prob = isnothing(inequality_constraints[1]) ? NonlinearProblem(test_f, z_val) : NonlinearLeastSquaresProblem(test_f, z_val)
sol = NonlinearSolve.solve(prob)
@assert length(sol.u) == length(z_symbolic) - n
# compute difference between two solutions
println("Duals from nonlinear solve: $(sol.u)")
@show sol.u .- z_sol_new_goop[Not(1:n)]
@show norm(sol.u .- z_sol_new_goop[Not(1:n)], Inf)
