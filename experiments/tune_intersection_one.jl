include("Intersection.jl")

using .Intersection
using ReducedGOOP
using Random

length(ARGS) >= 6 || error("Usage: tune_intersection_one.jl eta0 tightening loosening rho_low rho_high max_inner_iters [tol] [label]")

eta0 = parse(Float64, ARGS[1])
tightening = parse(Float64, ARGS[2])
loosening = parse(Float64, ARGS[3])
rho_low = parse(Float64, ARGS[4])
rho_high = parse(Float64, ARGS[5])
max_inner_iters = parse(Int, ARGS[6])
tol = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 1e-14
label = length(ARGS) >= 8 ? ARGS[8] : "case"

Random.seed!(123)

dynamics_model = Intersection.PlanarDoubleIntegrator()
num_players = 2
planning_horizon = 12
collision_avoidance = 1.5
speed_limit = 2.0
map_end = 7
lane_width = 2
state_dimension = 4
control_dimension = 2
dt = 0.2
control_bounds = (; lb = [-10.0, -10.0], ub = [10.0, 10.0])

dynamics = Intersection.build_intersection_dynamics(
	dynamics_model;
	Δt = dt,
	state_dimension,
	control_dimension,
)
(; problem, flatten_parameters) = Intersection.get_setup(
	num_players;
	dynamics,
	control_bounds,
	planning_horizon,
	collision_avoidance,
	speed_limit,
	map_end,
	lane_width,
)
GOOP_kkt_system = ReducedGOOP.generate_slacked_reduced_kkt_system(problem)

initial_state1 = [-4.0, -1.0, 3.0, 0.0]
initial_state2 = [1.0, -5.0, 0.0, 1.5]
goal_position1 = [6.0, -1.0]
goal_position2 = [1.0, 6.0]
obstacle_position = [0.25, 0.15]

(; θ) = Intersection.build_instance_parameters(
	flatten_parameters,
	initial_state1,
	initial_state2,
	goal_position1,
	goal_position2,
	obstacle_position,
)
(; warmstart_solution) = Intersection.build_default_warmstart(
	planning_horizon,
	dynamics,
	initial_state1,
	initial_state2;
	speed_limit,
)

options = ReducedGOOP.InteriorPointOptions(;
	tol,
	η₀ = eta0,
	ϵ₀ = 0.1,
	max_inner_iters,
	max_outer_iters = 1,
	tightening_rate = tightening,
	loosening_rate = loosening,
	min_stepsize = 1e-20,
	linesearch = :backtracking,
	linear_solve_algorithm = ReducedGOOP.LinearSolve.KrylovJL_LSMR(),
	use_linsolve = false,
	record_convergence = true,
	record_condition_number = false,
	eta_retry_growth = 0.3,
	perturbation_enabled = false,
	stagnation_rtol = 1e-1,
	perturbation_scale = 1e-6,
	tsvd_threshold = 0.0,
	use_marquardt_scaling = true,
	ρ_low = rho_low,
	ρ_high = rho_high,
	verbose = false,
)

elapsed = @elapsed output = ReducedGOOP.solve(
	ReducedGOOP.InteriorPoint(),
	GOOP_kkt_system,
	θ;
	z₀ = warmstart_solution,
	options,
)

history = output.kkt_error_history
best, best_iter = isempty(history) ? (output.kkt_error, output.total_iters) : findmin(history)
n = length(history)
marker_indices = unique([1, min(100, n), min(250, n), min(300, n), min(500, n), min(750, n), n])
markers = join(["$(i):$(history[i])" for i in marker_indices if i >= 1], ",")

println(
	"RUN label=$(label) eta0=$(eta0) tightening=$(tightening) loosening=$(loosening) " *
	"rho_low=$(rho_low) rho_high=$(rho_high) tol=$(tol) status=$(output.status) " *
	"total_iters=$(output.total_iters) final=$(output.kkt_error) best=$(best) " *
	"best_iter=$(best_iter) elapsed=$(elapsed) markers=$(markers)",
)
