module RoboticArmCore

# The two-arm pot carry as a GOOP, in Scholtes form.
#
# A two-armed robot carries a hot pot by its handle toward a goal while a child (or a
# pet) chases the pot. Player 1 is the robot, ONE agent holding the pot with both
# grippers; player 2 is the child. Both are 3D single integrators, so the control is
# a velocity and the speed limits are bounds on ‖u‖.
#
# Ported from ScholtesReducedGOOP.jl `examples/robotic_arm.jl` (Jingqi Li): the
# robosuite-scale scenario, constraints, preference hierarchy, initial guesses and the
# ρ sweep, with the same numbers. Two differences in form, not in content:
#
#   - x₀ is the PARAMETER θ = [x₀¹; x₀²] rather than a constant baked into the dynamics,
#     so one compiled system serves every initial state (receding horizon, MPC).
#   - the problem is a `ParametricGOOP`; its KKT system is built with
#     `complementarity = :scholtes` and solved with `ReducedGOOP.Scholtes()`.
#
# Layout (as in the source): z^i = [x_1; …; x_T; u_1; …; u_T], x¹ = [p₁; p₂] ∈ R⁶,
# u¹ ∈ R⁶, x² = p₃ ∈ R³, u² ∈ R³.

using BlockArrays: BlockArray, Block
using LinearAlgebra: norm
using Printf: @sprintf, @printf
using ReducedGOOP

export ScenarioConfig, default_scenario_config, RoboticArmContext, build_context,
       scenario_parameters, zero_control_guess, direct_path_guess, solve_rho_sweep,
       solver_options, plan_metrics, trajectories, state, control, gripper, pot_centre,
       child, pot_tilt, safety, arm_speed, arm_reach, child_speed, child_reach,
       handle_grasp, dynamics_residual, load_balance, pot_goal, chase,
       effort, eq_columns, RHO_SWEEP, LINEAR_SOLVER, MAX_INNER, TOL, PROJ_EPS,
       ETA_OPTIONS, CONTINUE_ETA, RHO_REF, NX, NU, TO

const TO = ReducedGOOP.TO

# ── 1. The scenario ───────────────────────────────────────────────────────────────
const NX = (6, 3)
const NU = (6, 3)

const DEFAULT_HORIZON = 30
const DEFAULT_DT = 0.1
const DEFAULT_X_INIT = [
    [-0.158824691536631, -0.003733450106580549, 0.9596013071940565,
     0.1588246915365118, 0.00373345010657488, 0.959601307194051],
    [-0.5080493093427129, -0.2523146383427626, 1.1011550012508406],
]
const DEFAULT_BASES = [
    [-3.4290110376125894e-17, -0.56000000000000005, 0.91200000000000003],
    [-3.4290110376125894e-17, 0.56000000000000005, 0.91200000000000003],
    [-0.75, -0.12664, 1.3137499999999998],
]
const DEFAULT_GOAL_OFFSET = [0.0, -0.2, 0.5]
const DEFAULT_D_MIN = 0.4
const DEFAULT_V_ARM = 0.6
const DEFAULT_V_CHILD = 0.3
const DEFAULT_BALANCE = 0.05
const DEFAULT_REACH_MAX = 0.8
const DEFAULT_CHILD_REACH_MAX = 0.45

"""
The scenario's geometry and limits. `x_init` is the nominal initial state (the θ a
context is built at); `pot_goal` defaults to the start centre plus `[0, -0.2, 0.5]` and
`d_handle` to the grippers' initial separation (the handle is rigid).
"""
Base.@kwdef struct ScenarioConfig
    horizon::Int = DEFAULT_HORIZON
    dt::Float64 = DEFAULT_DT
    x_init::Vector{Vector{Float64}} = deepcopy(DEFAULT_X_INIT)
    bases::Vector{Vector{Float64}} = deepcopy(DEFAULT_BASES)
    pot_goal::Vector{Float64} =
        0.5 .* (x_init[1][1:3] .+ x_init[1][4:6]) .+ DEFAULT_GOAL_OFFSET
    d_min::Float64 = DEFAULT_D_MIN
    v_arm::Float64 = DEFAULT_V_ARM
    v_child::Float64 = DEFAULT_V_CHILD
    balance::Float64 = DEFAULT_BALANCE
    reach_max::Float64 = DEFAULT_REACH_MAX
    child_reach_max::Float64 = DEFAULT_CHILD_REACH_MAX
    d_handle::Float64 = norm(x_init[1][1:3] .- x_init[1][4:6])
end

default_scenario_config(; kwargs...) = ScenarioConfig(; kwargs...)

"θ for an initial state: `[x₀¹; x₀²]`."
scenario_parameters(x_init) = vcat(Float64.(x_init[1]), Float64.(x_init[2]))

# ── 2. Indexing (per player: states then controls) ───────────────────────────────
player_offset(sc, i) = sum((NX[j] + NU[j]) * sc.horizon for j in 1:(i - 1); init = 0)
state_start(sc, i, t) = player_offset(sc, i) + (t - 1) * NX[i]
control_start(sc, i, t) = player_offset(sc, i) + NX[i] * sc.horizon + (t - 1) * NU[i]

state(sc, z, i, t) = (k = state_start(sc, i, t); [z[k + c] for c in 1:NX[i]])
control(sc, z, i, t) = (k = control_start(sc, i, t); [z[k + c] for c in 1:NU[i]])
gripper(sc, z, t, a) = state(sc, z, 1, t)[(3a - 2):(3a)]
pot_centre(sc, z, t) = 0.5 .* (gripper(sc, z, t, 1) .+ gripper(sc, z, t, 2))
child(sc, z, t) = state(sc, z, 2, t)

x0_of(θ, i) = i == 1 ? θ[1:NX[1]] : θ[(NX[1] + 1):(NX[1] + NX[2])]

# ── 3. Equalities: dynamics (x₀ = θ is data), rigid handle ─────────────────────────
f(sc, x, u) = x .+ sc.dt .* u

function dynamics_residual(sc, z, θ, i)
    rows = eltype(z)[]
    for t in 1:(sc.horizon)
        x_prev = t == 1 ? x0_of(θ, i) : state(sc, z, i, t - 1)
        x_pred = f(sc, x_prev, control(sc, z, i, t))
        x_now = state(sc, z, i, t)
        for k in 1:NX[i]
            push!(rows, x_now[k] - x_pred[k])
        end
    end
    rows
end

function handle_grasp(sc, z)
    rows = eltype(z)[]
    for t in 1:(sc.horizon)
        d = gripper(sc, z, t, 1) .- gripper(sc, z, t, 2)
        push!(rows, sum(abs2, d) - sc.d_handle^2)
    end
    rows
end

# ── 4. Inequalities, g(z) ≥ 0 ─────────────────────────────────────────────────────
# Safety is HORIZONTAL: the child's reach is a cylinder, so lifting the pot buys no
# clearance and the robot has to go around -- never over the child's hand, which now
# moves in z.
function safety(sc, z)
    rows = eltype(z)[]
    for t in 1:(sc.horizon)
        d = pot_centre(sc, z, t)[1:2] .- child(sc, z, t)[1:2]
        push!(rows, sum(abs2, d) - sc.d_min^2)
    end
    rows
end

function arm_speed(sc, z)
    rows = eltype(z)[]
    for t in 1:(sc.horizon)
        u = control(sc, z, 1, t)
        push!(rows, sc.v_arm^2 - sum(abs2, u[1:3]))
        push!(rows, sc.v_arm^2 - sum(abs2, u[4:6]))
    end
    rows
end

# Each gripper within its arm's nominal reach of its own shoulder. Defined but OFF,
# as in the source (not part of the robot's inequalities).
function arm_reach(sc, z)
    rows = eltype(z)[]
    for t in 1:(sc.horizon), a in 1:2
        d = gripper(sc, z, t, a) .- sc.bases[a]
        push!(rows, sc.reach_max^2 - sum(abs2, d))
    end
    rows
end

# 3D: the child's height is free (no ground pin), so its vertical speed is bounded too.
function child_speed(sc, z)
    rows = eltype(z)[]
    for t in 1:(sc.horizon)
        u = control(sc, z, 2, t)
        push!(rows, sc.v_child^2 - sum(abs2, u))
    end
    rows
end

function child_reach(sc, z)
    rows = eltype(z)[]
    for t in 1:(sc.horizon)
        d = child(sc, z, t) .- sc.bases[3]
        push!(rows, sc.child_reach_max^2 - sum(abs2, d))
    end
    rows
end

robot_inequality(sc, z) = vcat(safety(sc, z), arm_speed(sc, z)) # arm_reach(sc, z)
child_inequality(sc, z) = vcat(child_speed(sc, z), child_reach(sc, z))

# ── 5. The preference hierarchy [lowest, …, highest priority] ─────────────────────
#   robot: effort, pot to the goal (running cost), keep the pot level (pot_tilt)
#   child: effort, chase the pot
pot_tilt(sc, z, t) = abs(gripper(sc, z, t, 1)[3] - gripper(sc, z, t, 2)[3])

function load_balance(sc, z)
    total = 0.0
    for t in 1:(sc.horizon)
        offset = pot_tilt(sc, z, t)^2
        total = total + ifelse(offset < sc.balance^2, 0.0, offset - sc.balance^2)
    end
    total
end

pot_goal(sc, z) = sum(sum(abs2, pot_centre(sc, z, t) .- sc.pot_goal) for t in 1:(sc.horizon))

function chase(sc, z)
    total = 0.0
    for t in 1:(sc.horizon)
        total = total + sum(abs2, child(sc, z, t) .- pot_centre(sc, z, t))
    end
    total
end

function effort(sc, z, i)
    total = 0.0
    for t in 1:(sc.horizon)
        total = total + sum(abs2, control(sc, z, i, t))
    end
    total
end

"The `ParametricGOOP` of a scenario, with θ = [x₀¹; x₀²]."
function build_problem(sc::ScenarioConfig)
    ni = [(NX[i] + NU[i]) * sc.horizon for i in 1:2]
    x = BlockArray(zeros(sum(ni)), ni)
    θ = BlockArray(scenario_parameters(sc.x_init), [NX[1], NX[2]])
    flat(x) = collect(x)
    flatθ(θ) = collect(θ)
    ReducedGOOP.ParametricGOOP(
        x,
        θ;
        preferences = [
            Function[
                (x, θ) -> effort(sc, flat(x), 1),
                (x, θ) -> pot_goal(sc, flat(x)),
                (x, θ) -> load_balance(sc, flat(x)),
            ],
            Function[
                (x, θ) -> effort(sc, flat(x), 2), 
                (x, θ) -> chase(sc, flat(x))
            ],
        ],
        is_prioritized_constraint = [[false, false, false], [false, false]],
        equality_constraints = [
            (x, θ) -> vcat(dynamics_residual(sc, flat(x), flatθ(θ), 1), handle_grasp(sc, flat(x))),
            (x, θ) -> dynamics_residual(sc, flat(x), flatθ(θ), 2),
        ],
        inequality_constraints = [
            (x, θ) -> robot_inequality(sc, flat(x)),
            (x, θ) -> child_inequality(sc, flat(x)),
        ],
    )
end

# ── 6. Initial guesses (must be STRICTLY feasible) ────────────────────────────────
function zero_control_guess(sc::ScenarioConfig, x_init = sc.x_init)
    z = zeros(sum((NX[i] + NU[i]) * sc.horizon for i in 1:2))
    for i in 1:2
        x = x_init[i]
        for t in 1:(sc.horizon)
            x = f(sc, x, zeros(NU[i]))
            k = state_start(sc, i, t)
            for c in 1:NX[i]
                z[k + c] = x[c]
            end
        end
    end
    z
end

"Straight-line pot, child closing on the safety circle (source `direct_path_guess`)."
function direct_path_guess(sc::ScenarioConfig, x_init = sc.x_init; child_speed_frac = 0.7,
                           d_floor = 1.05 * sc.d_min)
    0 < child_speed_frac < 1 || error("child_speed_frac must lie strictly in (0, 1)")
    T, DT = sc.horizon, sc.dt
    z = zeros(sum((NX[i] + NU[i]) * T for i in 1:2))
    pot_0 = 0.5 .* (x_init[1][1:3] .+ x_init[1][4:6])
    span = norm(sc.pot_goal .- pot_0)
    span / (T * DT) < sc.v_arm || error("direct_path_guess: the goal is out of reach in this horizon")
    half = 0.5 .* (x_init[1][1:3] .- x_init[1][4:6])
    reach = child_speed_frac * sc.v_child * DT
    x_robot, x_child = copy(x_init[1]), copy(x_init[2])
    for t in 1:T
        pot = pot_0 .+ (t / T) .* (sc.pot_goal .- pot_0)
        x_robot_next = vcat(pot .+ half, pot .- half)
        x_child_next = copy(x_child)
        v = x_child[1:2] .- pot[1:2]
        nv = norm(v)
        dir = nv > 1e-12 ? v ./ nv : [1.0, 0.0]
        target = pot[1:2] .+ d_floor .* dir
        d = target .- x_child[1:2]
        nd = norm(d)
        nd > 1e-12 && (x_child_next[1:2] .+= min(reach, nd) .* d ./ nd)
        u_robot = (x_robot_next .- x_robot) ./ DT
        u_child = (x_child_next .- x_child) ./ DT
        k = state_start(sc, 1, t); z[(k + 1):(k + NX[1])] .= x_robot_next
        k = control_start(sc, 1, t); z[(k + 1):(k + NU[1])] .= u_robot
        k = state_start(sc, 2, t); z[(k + 1):(k + NX[2])] .= x_child_next
        k = control_start(sc, 2, t); z[(k + 1):(k + NU[2])] .= u_child
        x_robot, x_child = x_robot_next, x_child_next
    end
    worst = min(minimum(safety(sc, z)), minimum(arm_speed(sc, z)), minimum(child_speed(sc, z)))
    worst > 0 || error("direct_path_guess: not strictly feasible (worst g = $worst)")
    z
end

# ── 7. The compiled context and the ρ sweep ───────────────────────────────────────
# Solver settings, as the source sets them (examples/robotic_arm.jl L202-245).
const RHO_SWEEP = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
const LINEAR_SOLVER = :normal # :normal, :klu
const RHO_REF = 1e-4
const MAX_INNER = 500
const TOL = 1e-5
const PROJ_EPS = 1e-8
const ETA_OPTIONS = (
    eta_schedule = parse(Bool, get(ENV, "GOOP_ETA_SCHEDULE", "true")),
    eta_init = parse(Float64, get(ENV, "GOOP_ETA_INIT", "1e-4")),
    eta_min = parse(Float64, get(ENV, "GOOP_ETA_MIN", "1e-8")),
    eta_max = parse(Float64, get(ENV, "GOOP_ETA_MAX", "1e4")),
    gain_low = parse(Float64, get(ENV, "GOOP_GAIN_LOW", "0.25")),
    gain_high = parse(Float64, get(ENV, "GOOP_GAIN_HIGH", "0.75")),
    tightening_rate = parse(Float64, get(ENV, "GOOP_TIGHTENING_RATE", "1.2")),
    loosening_rate = parse(Float64, get(ENV, "GOOP_LOOSENING_RATE", "3.0")),
)
const CONTINUE_ETA = parse(Bool, get(ENV, "GOOP_CONTINUE_ETA", "false"))

"A scenario, its `ParametricGOOP` and its compiled Scholtes KKT system."
struct RoboticArmContext{P,K}
    scenario::ScenarioConfig
    problem::P
    kkt::K
    m::Int
    build_time::Float64
end

function build_context(sc::ScenarioConfig = ScenarioConfig(); codegen = :fast_differentiation)
    @info "build_context: building the ParametricGOOP problem (horizon = $(sc.horizon))"
    problem = build_problem(sc)
    @info "build_context: generating and compiling the KKT system"
    t = @elapsed kkt = ReducedGOOP.generate_slacked_reduced_kkt_system(
        problem;
        complementarity = :scholtes,
        codegen,
    )
    RoboticArmContext(sc, problem, kkt, kkt.n_nc + 2kkt.n_comp, t)
end

"Scholtes options for one ρ row, with every bar scaled by √m (per-equation tolerances)."
solver_options(ctx::RoboticArmContext, ρ; linear_solver = LINEAR_SOLVER, proj_eps = PROJ_EPS,
               max_inner = MAX_INNER, tol = TOL, eta_init = ETA_OPTIONS.eta_init) =
    ReducedGOOP.ScholtesOptions(; ETA_OPTIONS..., eta_init, linear_solver, proj_eps,
        rho_schedule = [ρ], max_inner, tol = tol * sqrt(ctx.m), tol_inner = ρ * sqrt(ctx.m))

"Columns of y holding equality-like duals (λ, ψ): copied between solves by warm-z+eq."
eq_columns(kkt) =
    findall(v -> startswith(string(v), "lam_") || startswith(string(v), "psi_"), kkt.vars)

# The per-iteration series the convergence panels draw (the source's section 6.6): the
# direction size ‖δw‖∞, η, the complementarity slackness ‖s ⊙ γ + u‖∞ over the relaxed
# rows, the smallest s / γ / u before each step, and the accepted step length α. The
# solver's trace fields are live buffers, so each is reduced to a scalar on the spot.
new_traces() = (dwn = Float64[], eta = Float64[], comp = Float64[],
                positive = NamedTuple[], alpha = Float64[])

function convergence_trace(kkt, tr)
    tr === nothing && return nothing
    n, nc = kkt.n, kkt.n_comp
    uidx = kkt.u_index
    # over the RELAXED rows only: an exact φ pair (u_index = 0) has no u and aims at 0
    comp_peak(nt) = begin
        peak = 0.0
        @inbounds for j in 1:nc
            uj = uidx[j]
            uj == 0 && continue
            v = abs(nt.w[n + j] * nt.b[j] + nt.w[n + nc + uj])
            v > peak && (peak = v)
        end
        peak
    end
    return function (nt)
        push!(tr.dwn, norm(nt.dw, Inf))
        push!(tr.eta, nt.eta)
        push!(tr.comp, comp_peak(nt))
        push!(tr.positive, (iter = nt.iter,
                            s = minimum(view(nt.w, (n + 1):(n + nc))),
                            gamma = minimum(nt.b),
                            u = minimum(view(nt.w, (n + nc + 1):length(nt.w)))))
        return nothing
    end
end

# The accepted Armijo step length; fires after every line search, a failed one included
# (α = 0).
step_size_trace(tr) = tr === nothing ? nothing : (nt -> (push!(tr.alpha, nt.alpha); nothing))

"""
    solve_rho_sweep(ctx, θ = scenario_parameters(ctx.scenario.x_init); z0, rho_schedule,
                    linear_solver, proj_eps, max_inner, tol, stop_at_tol = false,
                    collect_traces = false, on_result = nothing) -> (cold, warm)

The source's `solve_rho_sweep`: a cold solve at every ρ, then the warm-z+eq chain from
the cold row with the smallest ‖K₀‖/√m down the ρ below it. `stop_at_tol` runs the cold
rows fine → coarse and returns at the first converged row (then at the first converged
warm row). `on_result(tag, ρ, result, elapsed, trace)` is called after every solve.
Returns `cold :: Vector{Pair{Float64,Any}}` and `warm :: Vector{Tuple{String,Float64,Any,Any}}`.
"""
function solve_rho_sweep(ctx::RoboticArmContext, θ = scenario_parameters(ctx.scenario.x_init);
                         z0 = zero_control_guess(ctx.scenario, [θ[1:6], θ[7:9]]),
                         rho_schedule = RHO_SWEEP, linear_solver = LINEAR_SOLVER,
                         proj_eps = PROJ_EPS, max_inner = MAX_INNER, tol = TOL,
                         stop_at_tol = false, collect_traces = false, on_result = nothing)
    kkt = ctx.kkt
    scale = sqrt(ctx.m)
    k0(r) = r.residual / scale
    new_trace() = collect_traces ? new_traces() : nothing
    solve_at(ρ; w₀ = nothing, tr = nothing, eta_init = ETA_OPTIONS.eta_init) =
        ReducedGOOP.solve(ReducedGOOP.Scholtes(), kkt, θ; z₀ = z0, w₀,
            options = solver_options(ctx, ρ; linear_solver, proj_eps, max_inner, tol, eta_init),
            trace = convergence_trace(kkt, tr), step_trace = step_size_trace(tr))
    report(tag, ρ, r, t, tr) = on_result === nothing || on_result(tag, ρ, r, t, tr)

    warm = Tuple{String,Float64,Any,Any}[]
    cold = Pair{Float64,Any}[]
    for ρ in (stop_at_tol ? reverse(rho_schedule) : rho_schedule)
        tr = new_trace()
        t = @elapsed r = solve_at(ρ; tr)
        report("cold", ρ, r, t, tr)
        push!(cold, ρ => r)
        stop_at_tol && k0(r) < tol && return cold, warm
    end
    isempty(cold) && return cold, warm

    seed_idx = argmin(i -> k0(cold[i].second), eachindex(cold))
    seed, seed_rho = cold[seed_idx].second, cold[seed_idx].first
    remaining = rho_schedule[(findfirst(==(seed_rho), rho_schedule) + 1):end]
    k0(seed) < tol || @warn @sprintf(
        "solve_rho_sweep: the best cold row (ρ = %.0e) did not converge (||K_0||/sqrt(m) = %.3e > %.0e)",
        seed_rho, k0(seed), tol)
    eq_cols = eq_columns(kkt)
    prev = seed
    for ρ in remaining
        tr = new_trace()
        w₀ = ReducedGOOP.scholtes_warm_start(kkt, θ, prev.z, ρ; eq_cols, eq = prev.w[eq_cols])
        t = @elapsed r = solve_at(ρ; w₀, tr, eta_init = CONTINUE_ETA ? prev.eta : ETA_OPTIONS.eta_init)
        report("warm-z+eq", ρ, r, t, tr)
        push!(warm, ("warm-z+eq", ρ, r, tr))
        stop_at_tol && k0(r) < tol && return cold, warm
        prev = r
    end
    return cold, warm
end

# ── 8. What a plan does ───────────────────────────────────────────────────────────
"Per-player trajectories of z: robot states/controls (6 × T), child states/controls (3 × T)."
function trajectories(sc::ScenarioConfig, z)
    T = sc.horizon
    (; robot_xs = reduce(hcat, [state(sc, z, 1, t) for t in 1:T]),
       robot_us = reduce(hcat, [control(sc, z, 1, t) for t in 1:T]),
       child_xs = reduce(hcat, [state(sc, z, 2, t) for t in 1:T]),
       child_us = reduce(hcat, [control(sc, z, 2, t) for t in 1:T]))
end

"The quantities a plan is judged on (goal error, gaps, speeds, tilt, drift, objectives)."
function plan_metrics(sc::ScenarioConfig, z, θ = scenario_parameters(sc.x_init))
    T = sc.horizon
    (; goal_error = norm(pot_centre(sc, z, T) .- sc.pot_goal),
       min_gap = minimum(norm(pot_centre(sc, z, t)[1:2] .- child(sc, z, t)[1:2]) for t in 1:T),
       max_tilt = maximum(pot_tilt(sc, z, t) for t in 1:T),
       min_safety = minimum(safety(sc, z)), min_arm_speed = minimum(arm_speed(sc, z)),
       min_child_speed = minimum(child_speed(sc, z)), min_child_reach = minimum(child_reach(sc, z)),
       min_arm_reach_inactive = minimum(arm_reach(sc, z)),
       max_handle_drift = maximum(abs, handle_grasp(sc, z)),
       max_dynamics = max(maximum(abs, dynamics_residual(sc, z, θ, 1)),
                          maximum(abs, dynamics_residual(sc, z, θ, 2))),
       effort1 = effort(sc, z, 1), pot_goal = pot_goal(sc, z), load_balance = load_balance(sc, z),
       effort2 = effort(sc, z, 2), chase = chase(sc, z))
end

end # module RoboticArmCore
