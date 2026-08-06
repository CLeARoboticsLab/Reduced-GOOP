module Robotic_arm_receding

# Forward BLAS/LAPACK to Apple Accelerate (AMX)
@static if Sys.isapple()
    using AppleAccelerate: AppleAccelerate
end

using CairoMakie: CairoMakie
using JLD2, Random
using Dates: Dates
using Statistics: mean, median, std
using ReducedGOOP
using TimerOutputs: @timeit, reset_timer!

# Reuse a Revise-tracked core when one is already loaded in Main. The fallback
# keeps this entry point loadable on its own in a fresh Julia process.
const ROBOTIC_ARM_CORE_PATH = joinpath(@__DIR__, "robotic_arm_core.jl")
if !isdefined(Main, :RoboticArmCore)
    Base.include(Main, ROBOTIC_ARM_CORE_PATH)
end
Main.RoboticArmCore isa Module || error("Main.RoboticArmCore exists but is not a module.")
using Main.RoboticArmCore

# Preserve qualified access to the core API from this entry-point module.
for core_symbol in names(Main.RoboticArmCore)
    core_symbol === :RoboticArmCore && continue
    @eval export $core_symbol
end

# Share the visualization layer with the open-loop entry point; neither entry
# point depends on the other.
include(joinpath(@__DIR__, "robotic_arm_visualization.jl"))

const DUAL_WARMSTART_MODES =
    (:primal_only, :equality_duals, :all_except_innermost_stationarity)

# ── Experiment entry point ─────────────────────────────────────────────────────

"""
Receding-horizon (MPC) version of the robotic-arm experiment.

The symbolic KKT system is built exactly once for a fixed `planning_horizon`.
Every MPC iteration then only updates the initial-state and initial-control
entries of the parameter vector `θ` and the primal warm start `z₀` before
re-solving, so all precomputed structures (compiled residuals, sparse Jacobian,
solver options) are reused across iterations.

At each MPC step the open-loop game is solved from the current state, the
first control is applied (the system advances to the first predicted state),
and the previous solution is shifted one stage to warm-start the next solve.
Non-converged solves are not fatal: the final iterate is used and the step is
marked as non-converged in the saved plots.
"""
function demo(;
    num_mpc_steps = 20,
    map_end = 10,
    lane_width = 2,
    verbose = false,
    rng_seed = 123,
    debug = false,
    goop_version = :quasi,     # :complete | :reduced | :quasi
    show_interactive_trajectory = false,
    record_condition_number = false,
    record_convergence = false,
    linear_solver = :klu,
    # FastDifferentiation otherwise emits all sparse-Jacobian entries as one
    # enormous RuntimeGeneratedFunction. Bounded chunks avoid pathological
    # first-call inference/lowering while preserving the same expressions.
    fd_codegen_chunk_size = 128,
    # `:primal_only` resets every non-primal variable; `:equality_duals` carries
    # λᵢₖ; `:all_except_innermost_stationarity` carries every dual except the ψ
    # segments targeting preference level Kⁱ. Only primals are horizon-shifted.
    dual_warmstart = :all_except_innermost_stationarity,
    reuse_factorization_iters = 0,
    # η_max = 1e6 lets a struggling step escalate the regularization into a
    # do-nothing limit cycle: LM steps scale like 1/η, so at η ~ 1e6 the solver
    # takes microscopic "full" steps that pass the line search but reduce the
    # residual by ~1e-5 per iteration until the budget is gone. Successful
    # steps never use η above ~1e-4, so 1e2 is a generous ceiling that makes
    # genuinely stuck solves fail fast instead.
    η_max = 1e2,
    # Growth used only after a singular KLU numeric factorization. Each retry
    # rebuilds the failed factorization cache before applying this increase.
    klu_singularity_eta_growth = 100.0,
    # Re-solve a failed step once from the default (cold) warmstart.
    rescue_failed_steps = true,
    # Armijo sufficient-decrease constant for the backtracking line search;
    # 0.0 recovers the pre-Armijo accept-any-decrease behavior.
    armijo_constant = 1e-4,
    tol = 0.008,
    # Newton iteration budget per MPC step. Raise it when a stiff grid (short
    # horizon relative to the distance the arm must cover) needs more iterations
    # than the default before the step is declared failed.
    max_inner_iters = 500,
    # Initial Levenberg-Marquardt regularization. Worth raising when solves stall
    # or diverge on ill-conditioned grids.
    η₀ = 1e-6,
    save_outputs = true,
    # Planning grid. `nothing` keeps the scenario defaults (T = 30, Δt = 0.1);
    # the Δt/T benchmark overrides both to hold Δt·T fixed.
    planning_horizon = nothing,
    Δt = nothing,
)
    reset_timer!(TO)
    @timeit TO "experiment setup" Random.seed!(rng_seed)

    dual_warmstart in DUAL_WARMSTART_MODES || throw(
        ArgumentError(
            "Unknown dual_warmstart mode $(dual_warmstart); expected one of " *
            "$(join(DUAL_WARMSTART_MODES, ", ")).",
        ),
    )

    # ── Scenario ───────────────────────────────────────────────────────────────
    scenario_overrides = (;
        (isnothing(planning_horizon) ? (;) : (; planning_horizon))...,
        (isnothing(Δt) ? (;) : (; Δt))...,
    )
    scenario_config = demo_scenario_config(; use_running_goal_cost = false, scenario_overrides...)

    # run_id uses the kwarg values (possibly nothing) before re-binding from scenario.
    run_id =
        "Robotic_arm_receding_" *
        Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS") *
        "_$(goop_version)_$(dual_warmstart)" *
        (isnothing(planning_horizon) ? "" : "_T$(planning_horizon)") *
        (isnothing(Δt) ? "" : "_dt$(Δt)")

    # Re-bind so a `nothing` kwarg picks up the scenario default.
    (;
        dynamics_model,
        dynamics,
        control_bounds,
        num_players,
        planning_horizon,
        Δt,
        base_initial_state1,
        base_initial_state2,
        goal_position1,
        goal_position2,
        initial_state3,
        goal_position3,
        collision_avoidance,
        arm_speed_limit,
        child_speed_limit,
        dₚ,
    ) = scenario_config

    ϵ₀ = 0.1  # placeholder; no inequality constraints in this scenario

    # ── Context: KKT system + options, built once ──────────────────────────────
    ctx = @timeit TO "context build" build_mpc_context(
        scenario_config;
        goop_version,
        dual_warmstart,
        fd_codegen_chunk_size,
        tol,
        max_inner_iters,
        linear_solver,
        klu_singularity_eta_growth,
        η_max,
        η₀,
        ϵ₀,
        armijo_constant,
        reuse_factorization_iters,
        record_convergence,
        record_condition_number,
        verbose,
    )
    (; GOOP_kkt_system, primal_dimensions, problem_setup_time_sec, kkt_build_time_sec) = ctx
    println(
        "one-time setup: problem construction $(round(problem_setup_time_sec; digits = 2)) s, ",
        "symbolic KKT build $(round(kkt_build_time_sec; digits = 2)) s",
    )

    # Dual warm-start metadata needed for run_metadata save
    total_equality_stationarity_dual_count =
        length(GOOP_kkt_system.all_equality_stationarity_dual_dims)
    selected_dual_count = if dual_warmstart === :primal_only
        0
    elseif dual_warmstart === :equality_duals
        length(GOOP_kkt_system.equality_constraint_dual_dims)
    else
        total_equality_stationarity_dual_count -
        length(GOOP_kkt_system.innermost_stationarity_dual_dims)
    end
    println(
        "dual warm-start mode: $(dual_warmstart) " *
        "($(selected_dual_count)/$(total_equality_stationarity_dual_count) " *
        "equality/stationarity dual coordinates carried)",
    )

    # ── Output directories ─────────────────────────────────────────────────────
    dirs = if save_outputs
        @timeit TO "output directory setup" prepare_receding_output_dirs(run_id; debug)
    else
        nothing
    end
    visualization_config =
        save_outputs ? VisualizationConfig(; dirs, show_interactive_trajectory) : nothing

    # ── MPC state: starts from scenario initial conditions ─────────────────────
    state = @timeit TO "warmstart construction" init_mpc_state(ctx)
    solver = ReducedGOOP.InteriorPoint()

    closed_loop_xs = [
        [vcat(copy(state.arm_state1), copy(state.arm_state2))],
        [copy(state.child_state)],
    ]
    closed_loop_us = [Vector{Float64}[] for _ in 1:num_players]

    step_statuses = Symbol[]
    step_kkt_errors = Float64[]
    step_solve_times = Float64[]
    step_total_iters = Int[]
    step_primary_statuses = Symbol[]
    step_primary_kkt_errors = Float64[]
    step_primary_iters = Int[]
    step_rescue_statuses = Symbol[]
    step_rescue_kkt_errors = Float64[]
    step_rescue_iters = Int[]
    step_attempt_iters = Int[]
    step_klu_singular_retries = Int[]
    step_svd_fallback_counts = Int[]
    θ1_initial = θ2_initial = θ3_initial = nothing

    # ── MPC loop ───────────────────────────────────────────────────────────────
    for k in 1:num_mpc_steps
        # Refresh the state/control boundary parameters without rebuilding the
        # KKT system
        instance_states = (;
            initial_state1 = state.arm_state1,
            initial_state2 = state.arm_state2,
            initial_state3 = state.child_state,
            initial_control1 = state.ctrl1,
            initial_control2 = state.ctrl2,
            initial_control3 = state.ctrl3,
        )
        stage_warmstart = save_outputs ? copy(state.warmstart) : nothing
        instance_parameters =
            @timeit TO "instance parameter construction" build_instance_parameters(
                ctx.flatten_parameters,
                instance_states,
                scenario_config,
            )
        (; θ1, θ2, θ3, θ) = instance_parameters
        if k == 1
            θ1_initial, θ2_initial, θ3_initial = θ1, θ2, θ3
        end

        # Plot the warm start used for this step's solve.
        # The interactive export of the previous step leaves WGLMakie active; static PDF saving needs the CairoMakie backend.
        if save_outputs
            CairoMakie.activate!()
            # A full-z warmstart carries duals and slacks; the visualization only
            # understands the primal block.
            stage_warmstart_primal =
                length(stage_warmstart) == GOOP_kkt_system.variable_dimension ?
                stage_warmstart[GOOP_kkt_system.primal_dims] : stage_warmstart
            @timeit TO "warmstart visualization" save_warmstart_visualizations(
                stage_warmstart_primal;
                filename_tag = "step_$(k)",
                primal_dimensions,
                instance_parameters,
                scenario_config,
                visualization_config,
            )
        end

        result = @timeit TO "MPC step" step_mpc!(
            state, ctx, solver;
            rescue_failed_steps,
            step_label = "step $(k): ",
        )
        (;
            output,
            primary_output,
            strategies,
            θ1,
            θ2,
            θ3,
            elapsed_time,
            rescue_status,
            rescue_kkt_error,
            rescue_iters,
            rescue_klu_singular_retries,
            rescue_svd_fallback_count,
        ) = result

        converged = output.status == :solved
        push!(step_statuses, output.status)
        push!(step_kkt_errors, output.kkt_error)
        push!(step_solve_times, elapsed_time)
        push!(step_total_iters, output.total_iters)
        push!(step_primary_statuses, primary_output.status)
        push!(step_primary_kkt_errors, primary_output.kkt_error)
        push!(step_primary_iters, primary_output.total_iters)
        push!(step_rescue_statuses, rescue_status)
        push!(step_rescue_kkt_errors, rescue_kkt_error)
        push!(step_rescue_iters, rescue_iters)
        push!(step_attempt_iters, primary_output.total_iters + rescue_iters)
        push!(
            step_klu_singular_retries,
            primary_output.klu_singular_retries + rescue_klu_singular_retries,
        )
        push!(
            step_svd_fallback_counts,
            primary_output.svd_fallback_count + rescue_svd_fallback_count,
        )
        attempt_summary =
            rescue_iters > 0 ?
            ", primary_iters=$(primary_output.total_iters), rescue_iters=$(rescue_iters)" :
            ""
        println(
            "MPC step $(k)/$(num_mpc_steps): status=$(output.status), ",
            "iters=$(output.total_iters), ",
            "kkt_error=$(round(output.kkt_error; sigdigits = 4)), ",
            "time=$(round(elapsed_time; digits = 3)) sec",
            attempt_summary,
        )
        if !converged
            println(
                "  step $(k) did not converge within max_inner_iters=$(max_inner_iters); ",
                "continuing with the final iterate.",
            )
        end

        # Apply the first control: advance every player to its first predicted
        # state (equivalent to applying us[1] through the dynamics constraints).
        for player in 1:num_players
            push!(closed_loop_us[player], collect(strategies[player].us[1]))
            push!(closed_loop_xs[player], collect(strategies[player].xs[2]))
        end

        if save_outputs
            @timeit TO "solution output and plotting" begin
                solution_dict = Dict(
                    "mpc_step" => k,
                    "dual_warmstart" => dual_warmstart,
                    "strategies" => strategies,
                    "z" => output.z,
                    "x" => output.x,
                    "warmstart" => stage_warmstart,
                    "solve_time_sec" => elapsed_time,
                    "kkt_error" => output.kkt_error,
                    "ϵ" => output.ϵ,
                    "status" => output.status,
                    "total_iters" => output.total_iters,
                    "primary_status" => primary_output.status,
                    "primary_kkt_error" => primary_output.kkt_error,
                    "primary_iters" => primary_output.total_iters,
                    "rescue_status" => rescue_status,
                    "rescue_kkt_error" => rescue_kkt_error,
                    "rescue_iters" => rescue_iters,
                    "attempt_iters" => primary_output.total_iters + rescue_iters,
                    "klu_singular_retries" =>
                        primary_output.klu_singular_retries + rescue_klu_singular_retries,
                    "svd_fallback_count" =>
                        primary_output.svd_fallback_count + rescue_svd_fallback_count,
                    "kkt_error_history" => get(output, :kkt_error_history, Float64[]),
                    "condition_number_history" =>
                        get(output, :condition_number_history, Float64[]),
                    "eta_history" => get(output, :eta_history, Float64[]),
                    "alpha_history" => get(output, :alpha_history, Float64[]),
                    "rho_history" => get(output, :rho_history, Float64[]),
                    "arm_speed_limit" => arm_speed_limit,
                    "child_speed_limit" => child_speed_limit,
                )
                # The interactive export of the previous step leaves WGLMakie active;
                # static PDF saving needs the CairoMakie backend.
                CairoMakie.activate!()
                status_suffix = converged ? "" : "_notconverged"
                JLD2.save_object(
                    joinpath(
                        dirs.solution_data_dir,
                        "solution_dict_step_$(k)_eps$(ϵ₀)$(status_suffix).jld2",
                    ),
                    solution_dict,
                )
                save_convergence_diagnostics(
                    solution_dict,
                    dirs.convergence_plots_dir,
                    k,
                    ϵ₀;
                    filename_suffix = status_suffix,
                )
                save_step_plots(
                    strategies,
                    k,
                    num_mpc_steps,
                    converged,
                    output.kkt_error,
                    dirs;
                    map_end,
                    lane_width,
                    θ1,
                    θ2,
                    θ3,
                    goal_position1,
                    goal_position2,
                    goal_position3,
                    collision_avoidance,
                    dₚ,
                    arm_speed_limit,
                    child_speed_limit,
                    control_bounds,
                    dynamics_model,
                    save_interactive = show_interactive_trajectory,
                )
            end
        end
    end

    # ── Closed-loop outputs ────────────────────────────────────────────────────
    failed_steps = findall(==(:failed), step_statuses)
    println(
        "\nMPC run finished: $(num_mpc_steps - length(failed_steps))/$(num_mpc_steps) steps converged",
        isempty(failed_steps) ? "." :
        "; non-converged steps: $(join(failed_steps, ", ")).",
    )

    closed_loop_strategies = map(1:num_players) do player
        # Pad with a terminal zero control so xs/us have equal length, matching
        # the trajectory layout the plotting helpers expect.
        (;
            xs = closed_loop_xs[player],
            us = vcat(
                closed_loop_us[player],
                [zeros(dynamics[player].control_dimension)],
            ),
        )
    end

    if save_outputs
        @timeit TO "closed-loop output and plotting" begin
            JLD2.save_object(
                joinpath(dirs.problem_data_dir, "closed_loop_trajectory.jld2"),
                Dict(
                    "closed_loop_strategies" => closed_loop_strategies,
                    "closed_loop_xs" => closed_loop_xs,
                    "closed_loop_us" => closed_loop_us,
                    "step_statuses" => step_statuses,
                    "step_kkt_errors" => step_kkt_errors,
                    "step_solve_times" => step_solve_times,
                    "step_total_iters" => step_total_iters,
                    "step_primary_statuses" => step_primary_statuses,
                    "step_primary_kkt_errors" => step_primary_kkt_errors,
                    "step_primary_iters" => step_primary_iters,
                    "step_rescue_statuses" => step_rescue_statuses,
                    "step_rescue_kkt_errors" => step_rescue_kkt_errors,
                    "step_rescue_iters" => step_rescue_iters,
                    "step_attempt_iters" => step_attempt_iters,
                    "step_klu_singular_retries" => step_klu_singular_retries,
                    "step_svd_fallback_counts" => step_svd_fallback_counts,
                    "dual_warmstart" => dual_warmstart,
                ),
            )
            save_closed_loop_plots(
                closed_loop_strategies,
                failed_steps,
                num_mpc_steps,
                dirs;
                map_end,
                lane_width,
                θ1 = θ1_initial,
                θ2 = θ2_initial,
                θ3 = θ3_initial,
                goal_position1,
                goal_position2,
                goal_position3,
                collision_avoidance,
                dₚ,
                arm_speed_limit,
                child_speed_limit,
                control_bounds,
                dynamics_model,
                save_interactive = true,
            )
        end
    end

    # ── Runtime report ─────────────────────────────────────────────────────────
    timing_stats = @timeit TO "runtime report" begin
        stats = print_runtime_report(
            problem_setup_time_sec,
            kkt_build_time_sec,
            step_solve_times,
            step_statuses,
        )
        if save_outputs
            save_solve_time_plots(step_solve_times, step_statuses, dirs.timing_plots_dir)
            JLD2.save_object(
                joinpath(dirs.problem_data_dir, "timing_summary.jld2"),
                Dict(
                    "problem_setup_time_sec" => problem_setup_time_sec,
                    "kkt_build_time_sec" => kkt_build_time_sec,
                    "step_solve_times" => step_solve_times,
                    "step_statuses" => step_statuses,
                    "all_solve_stats" => stats.all_stats,
                    "warm_solve_stats" => stats.warm_stats,
                ),
            )
        end
        stats
    end

    if save_outputs
        @timeit TO "run metadata save" begin
            JLD2.save_object(
                joinpath(dirs.problem_data_dir, "run_metadata.jld2"),
                Dict(
                    "run_id" => run_id,
                    "debug" => debug,
                    "run_dir" => dirs.run_dir,
                    "rng_seed" => rng_seed,
                    "dual_warmstart" => dual_warmstart,
                    "selected_dual_count" => selected_dual_count,
                    "total_equality_stationarity_dual_count" =>
                        total_equality_stationarity_dual_count,
                    "dynamics_model" => dynamics_model_name(dynamics_model),
                    "planning_horizon" => planning_horizon,
                    "num_mpc_steps" => num_mpc_steps,
                    "tol" => tol,
                    "klu_singularity_eta_growth" => klu_singularity_eta_growth,
                    "ϵ₀" => ϵ₀,
                    "max_inner_iters" => max_inner_iters,
                    "fd_codegen_chunk_size" => fd_codegen_chunk_size,
                    "arm_speed_limit" => arm_speed_limit,
                    "child_speed_limit" => child_speed_limit,
                    "collision_avoidance" => collision_avoidance,
                    "dₚ" => dₚ,
                    "initial_state1" => base_initial_state1,
                    "initial_state2" => base_initial_state2,
                    "initial_state3" => initial_state3,
                    "goal_position1" => goal_position1,
                    "goal_position2" => goal_position2,
                    "goal_position3" => goal_position3,
                    "step_statuses" => step_statuses,
                    "step_kkt_errors" => step_kkt_errors,
                    "step_solve_times" => step_solve_times,
                    "step_total_iters" => step_total_iters,
                    "step_primary_statuses" => step_primary_statuses,
                    "step_primary_kkt_errors" => step_primary_kkt_errors,
                    "step_primary_iters" => step_primary_iters,
                    "step_rescue_statuses" => step_rescue_statuses,
                    "step_rescue_kkt_errors" => step_rescue_kkt_errors,
                    "step_rescue_iters" => step_rescue_iters,
                    "step_attempt_iters" => step_attempt_iters,
                    "step_klu_singular_retries" => step_klu_singular_retries,
                    "step_svd_fallback_counts" => step_svd_fallback_counts,
                    "problem_setup_time_sec" => problem_setup_time_sec,
                    "kkt_build_time_sec" => kkt_build_time_sec,
                    "total_solve_time_sec" => sum(step_solve_times),
                ),
            )
        end
    end

    save_outputs && println("outputs saved under: ", dirs.run_dir)
    println("\nTiming summary:")
    show(TO)
    println()

    (;
        closed_loop_strategies,
        step_statuses,
        step_kkt_errors,
        step_solve_times,
        step_total_iters,
        step_primary_statuses,
        step_primary_kkt_errors,
        step_primary_iters,
        step_rescue_statuses,
        step_rescue_kkt_errors,
        step_rescue_iters,
        step_attempt_iters,
        step_klu_singular_retries,
        step_svd_fallback_counts,
        dual_warmstart,
        selected_dual_count,
        timing_stats,
        planning_horizon,
        Δt,
        kkt_dimension = GOOP_kkt_system.kkt_dimension,
        variable_dimension = GOOP_kkt_system.variable_dimension,
        problem_setup_time_sec,
        kkt_build_time_sec,
        run_dir = isnothing(dirs) ? nothing : dirs.run_dir,
    )
end

# ── MPC utilities ──────────────────────────────────────────────────────────────

"""
Assemble the next MPC warm start. Primals are shifted separately; selected
equality and stationarity duals retain their existing flat coordinates. All
other variables use the same defaults as `ReducedGOOP.solve` (zero, with
positive slacks/inequality duals).
"""
function build_receding_warmstart(shifted_primal, previous_z, kkt_system, mode::Symbol)
    mode in DUAL_WARMSTART_MODES ||
        throw(ArgumentError("Unknown dual_warmstart mode $(mode)."))
    length(shifted_primal) == length(kkt_system.primal_dims) || throw(
        DimensionMismatch(
            "Shifted primal has length $(length(shifted_primal)); " *
            "expected $(length(kkt_system.primal_dims)).",
        ),
    )
    length(previous_z) == kkt_system.variable_dimension || throw(
        DimensionMismatch(
            "Previous KKT point has length $(length(previous_z)); " *
            "expected $(kkt_system.variable_dimension).",
        ),
    )

    mode === :primal_only && return copy(shifted_primal)

    T = promote_type(eltype(shifted_primal), eltype(previous_z))
    warmstart = zeros(T, kkt_system.variable_dimension)
    warmstart[kkt_system.preference_slack_dims] .= one(T)
    warmstart[kkt_system.interior_point_slack_dims] .= one(T)
    warmstart[kkt_system.inequality_constraint_dual_dims] .= one(T)
    warmstart[kkt_system.primal_dims] .= shifted_primal
    if mode === :equality_duals
        dims = kkt_system.equality_constraint_dual_dims
        warmstart[dims] .= previous_z[dims]
    else
        # Carry all equality/stationarity duals except those tied to the innermost level.
        dims = kkt_system.all_equality_stationarity_dual_dims
        warmstart[dims] .= previous_z[dims]
        warmstart[kkt_system.innermost_stationarity_dual_dims] .= zero(T)
    end
    warmstart
end

function build_mpc_context(
    obs;
    planning_horizon::Integer = 30,
    kwargs...,
)
    base_initial_state1 = collect(Float64, obs["robot0_eef_pos"])
    base_initial_state2 = collect(Float64, obs["robot1_eef_pos"])
    initial_state3      = collect(Float64, obs["robot2_eef_pos"])
    scenario_config = demo_scenario_config(;
        planning_horizon,
        base_initial_state1,
        base_initial_state2,
        initial_state3,
    )
    # Print geometry before the (slow) KKT build so the user can verify
    _print_scenario_config(scenario_config)
    build_mpc_context(scenario_config; kwargs...)
end

function _print_scenario_config(scenario_config::ScenarioConfig)
    (; base_initial_state1, base_initial_state2, goal_position1, goal_position2,
       initial_state3, goal_position3, arm_speed_limit, child_speed_limit,
       Δt, planning_horizon, collision_avoidance, dₚ) = scenario_config
    total_plan_time = planning_horizon * Δt
    max_reach = arm_speed_limit * total_plan_time
    dist1 = sqrt(sum(abs2, goal_position1 .- base_initial_state1))
    dist2 = sqrt(sum(abs2, goal_position2 .- base_initial_state2))
    println("\n── MPC scenario configuration ──────────────────────────────────────")
    println("  horizon:      T=$(planning_horizon) × Δt=$(Δt) s = $(total_plan_time) s total")
    println("  arm speed:    ≤ $(arm_speed_limit) m/s  →  max reach $(round(max_reach; digits = 3)) m per horizon")
    println("  child speed:  ≤ $(child_speed_limit) m/s")
    println("  safety dist:  $(collision_avoidance) m  (grip separation dₚ=$(round(dₚ; digits = 4)) m)")
    println("  arm 1  pos:   $(round.(base_initial_state1; digits = 4))")
    println("         goal:  $(round.(goal_position1; digits = 4))  dist=$(round(dist1; digits = 4)) m")
    println("  arm 2  pos:   $(round.(base_initial_state2; digits = 4))")
    println("         goal:  $(round.(goal_position2; digits = 4))  dist=$(round(dist2; digits = 4)) m")
    println("  child  pos:   $(round.(initial_state3; digits = 4))")
    println("         goal:  $(round.(goal_position3; digits = 4))")
    dist1 > max_reach && @warn "arm 1 goal $(round(dist1; digits=3)) m exceeds horizon reach $(round(max_reach; digits=3)) m"
    dist2 > max_reach && @warn "arm 2 goal $(round(dist2; digits=3)) m exceeds horizon reach $(round(max_reach; digits=3)) m"
    println("─────────────────────────────────────────────────────────────────────")
end

"""Immutable context for one MPC instance: KKT system, options, and timing metadata."""
struct MpcContext
    GOOP_kkt_system::Any
    scenario_config::Any
    flatten_parameters::Any
    primal_dimensions::Vector{Int}
    options::Any
    planning_horizon::Int
    arm_state_dimension::Int
    arm_control_dimension::Int
    dual_warmstart::Symbol
    problem_setup_time_sec::Float64
    kkt_build_time_sec::Float64
end

"""Mutable per-step planner state: arm/child positions, boundary controls, warmstart."""
mutable struct MpcState
    arm_state1::Vector{Float64}
    arm_state2::Vector{Float64}
    child_state::Vector{Float64}
    ctrl1::Vector{Float64}
    ctrl2::Vector{Float64}
    ctrl3::Vector{Float64}
    warmstart::Vector{Float64}
    step::Int
end

"""
Initialise an `MpcState` from the scenario defaults or supplied initial positions.

Override `arm_state1`, `arm_state2`, or `child_state` to start from a pose
other than the one baked into `ctx.scenario_config` (e.g. actual post-grip
EEF positions read from the simulator rather than the nominal scenario start).
"""
function init_mpc_state(
    ctx::MpcContext;
    arm_state1 = ctx.scenario_config.base_initial_state1,
    arm_state2 = ctx.scenario_config.base_initial_state2,
    child_state = ctx.scenario_config.initial_state3,
)
    initial_states = (;
        initial_state1 = arm_state1,
        initial_state2 = arm_state2,
        initial_state3 = child_state,
    )
    (; warmstart_solution) = build_default_warmstart(initial_states, ctx.scenario_config)
    controls = extract_initial_controls(
        warmstart_solution, ctx.primal_dimensions, ctx.scenario_config.dynamics,
    )
    MpcState(
        copy(arm_state1),
        copy(arm_state2),
        copy(child_state),
        copy(controls.initial_control1),
        copy(controls.initial_control2),
        copy(controls.initial_control3),
        warmstart_solution,
        0,
    )
end
export init_mpc_state

"""
Advance the planner by one MPC step: solve (with optional rescue), update
`state` in place, and return the full step result for logging or trajectory
recording.
"""
function step_mpc!(state::MpcState, ctx::MpcContext, solver; kwargs...)
    instance_states = (;
        initial_state1  = state.arm_state1,
        initial_state2  = state.arm_state2,
        initial_state3  = state.child_state,
        initial_control1 = state.ctrl1,
        initial_control2 = state.ctrl2,
        initial_control3 = state.ctrl3,
    )
    result = _mpc_solve(
        solver, ctx.GOOP_kkt_system, ctx.flatten_parameters, ctx.scenario_config,
        ctx.options, ctx.primal_dimensions,
        ctx.arm_state_dimension, ctx.arm_control_dimension,
        ctx.dual_warmstart, instance_states, state.warmstart; kwargs...,
    )
    state.arm_state1  = result.new_state1
    state.arm_state2  = result.new_state2
    state.child_state = result.new_state_child
    state.ctrl1 = result.new_ctrl1
    state.ctrl2 = result.new_ctrl2
    state.ctrl3 = result.new_ctrl3
    state.warmstart = result.next_warmstart
    state.step += 1
    result
end
export step_mpc!

"""
Solve the problem associated to one receding-horizon MPC step.

Builds parameters from `instance_states`, runs the primary solve, optionally
rescues from the current-state default warmstart on failure, advances the state
to the first predicted next knot, and shifts the warmstart forward one stage.

Both `demo` and `create_planner_from_context` call this so the solve, rescue,
and warmstart-shift logic is shared across entry points.
"""
function _mpc_solve(
    solver,
    GOOP_kkt_system,
    flatten_parameters,
    scenario_config,
    options,
    primal_dimensions,
    arm_state_dimension,
    arm_control_dimension,
    dual_warmstart,
    instance_states,
    stage_warmstart;
    rescue_failed_steps::Bool = true,
    step_label::String = "",
)
    (; dynamics, planning_horizon) = scenario_config

    instance_parameters =
        build_instance_parameters(flatten_parameters, instance_states, scenario_config)
    (; θ1, θ2, θ3, θ) = instance_parameters

    elapsed_time = @elapsed primary_output =
        ReducedGOOP.solve(solver, GOOP_kkt_system, θ; z₀ = stage_warmstart, options)
    output = primary_output
    rescue_status = :not_run
    rescue_kkt_error = NaN
    rescue_iters = 0
    rescue_klu_singular_retries = 0
    rescue_svd_fallback_count = 0

    if output.status == :failed && rescue_failed_steps
        println(
            "  $(step_label)solve from shifted warmstart failed ",
            "(kkt_error=$(round(output.kkt_error; sigdigits = 3))); ",
            "retrying from the default warmstart.",
        )
        rescue_warmstart =
            build_default_warmstart(instance_states, scenario_config).warmstart_solution
        elapsed_time += @elapsed rescue_output =
            ReducedGOOP.solve(solver, GOOP_kkt_system, θ; z₀ = rescue_warmstart, options)
        rescue_status = rescue_output.status
        rescue_kkt_error = rescue_output.kkt_error
        rescue_iters = rescue_output.total_iters
        rescue_klu_singular_retries = rescue_output.klu_singular_retries
        rescue_svd_fallback_count = rescue_output.svd_fallback_count
        if rescue_output.status == :solved || rescue_output.kkt_error < output.kkt_error
            output = rescue_output
        end
    end

    strategies = extract_player_strategies(output.x, primal_dimensions, dynamics)

    combined_arm_state = collect(Float64, strategies[1].xs[2])
    new_state1 = combined_arm_state[1:arm_state_dimension]
    new_state2 = combined_arm_state[(arm_state_dimension + 1):(2 * arm_state_dimension)]
    new_state_child = collect(Float64, strategies[2].xs[2])

    combined_arm_control = collect(Float64, strategies[1].us[2])
    new_ctrl1 = combined_arm_control[1:arm_control_dimension]
    new_ctrl2 = combined_arm_control[(arm_control_dimension + 1):(2 * arm_control_dimension)]
    new_ctrl3 = collect(Float64, strategies[2].us[2])

    shifted_strategies = shift_strategies(strategies, dynamics, planning_horizon)
    shifted_primal = flatten_warmstart_solution(
        planning_horizon,
        [s.xs for s in shifted_strategies],
        [s.us for s in shifted_strategies],
    )
    next_warmstart =
        build_receding_warmstart(shifted_primal, output.z, GOOP_kkt_system, dual_warmstart)

    (;
        output,
        primary_output,
        instance_parameters,
        strategies,
        θ1,
        θ2,
        θ3,
        elapsed_time,
        new_state1,
        new_state2,
        new_state_child,
        new_ctrl1,
        new_ctrl2,
        new_ctrl3,
        next_warmstart,
        rescue_status,
        rescue_kkt_error,
        rescue_iters,
        rescue_klu_singular_retries,
        rescue_svd_fallback_count,
    )
end

"""
Shift a per-player trajectory one stage forward to warm-start the next MPC
solve. The terminal stage is padded with a zero control passed through the
dynamics (trivially feasible); for the single-integrator
dynamics the appended terminal state repeats the previous final state.
"""
function shift_strategies(strategies, dynamics, planning_horizon)
    map(collect(enumerate(strategies))) do (player, strategy)
        xs = [collect(Float64, x) for x in strategy.xs[2:end]]
        us = [collect(Float64, u) for u in strategy.us[2:end]]
        # terminal_control = copy(us[end])
        terminal_control = zeros(dynamics[player].control_dimension)
        push!(xs, dynamics[player].step(xs[end], terminal_control, planning_horizon))
        push!(us, terminal_control)
        (; xs, us)
    end
end

# ── Runtime reporting ──────────────────────────────────────────────────────────

function _solve_time_stats(times)
    (;
        total = sum(times),
        mean = mean(times),
        median = median(times),
        min = minimum(times),
        max = maximum(times),
        std = length(times) > 1 ? std(times) : 0.0,
    )
end

function _format_stats(stats)
    "total $(round(stats.total; digits = 2)) s, " *
    "mean $(round(stats.mean; digits = 2)) s, " *
    "median $(round(stats.median; digits = 2)) s, " *
    "min $(round(stats.min; digits = 2)) s, " *
    "max $(round(stats.max; digits = 2)) s, " *
    "std $(round(stats.std; digits = 2)) s"
end

"""
Print a runtime report that separates one-time compilation/setup cost from the
repeated per-step solve cost, and return the solve-time statistics.
"""
function print_runtime_report(
    problem_setup_time_sec,
    kkt_build_time_sec,
    step_solve_times,
    step_statuses,
)
    num_steps = length(step_solve_times)
    println("\n── Runtime report ──────────────────────────────────────────────")
    println("One-time setup (paid once, reused by all $(num_steps) solves):")
    println("  problem construction:      $(round(problem_setup_time_sec; digits = 2)) s")
    println("  symbolic KKT system build: $(round(kkt_build_time_sec; digits = 2)) s")
    println(
        "  setup total:               $(round(problem_setup_time_sec + kkt_build_time_sec; digits = 2)) s",
    )
    println("Per-step solve times:")
    for (k, (t, status)) in enumerate(zip(step_solve_times, step_statuses))
        println("  step $(lpad(k, 2)): $(lpad(round(t; digits = 3), 9)) s  ($(status))")
    end
    all_stats = _solve_time_stats(step_solve_times)
    warm_stats = num_steps > 1 ? _solve_time_stats(step_solve_times[2:end]) : nothing
    println("All $(num_steps) solves: $(_format_stats(all_stats))")
    if !isnothing(warm_stats)
        jit_estimate = step_solve_times[1] - warm_stats.median
        println(
            "First solve (step 1) additionally pays the one-time JIT compilation ",
            "of the solver code: $(round(step_solve_times[1]; digits = 2)) s",
        )
        println(
            "  estimated JIT overhead ≈ $(round(jit_estimate; digits = 2)) s ",
            "(step 1 minus median warm solve)",
        )
        println("Warm solves (steps 2–$(num_steps)): $(_format_stats(warm_stats))")
    end
    println("────────────────────────────────────────────────────────────────")
    (; all_stats, warm_stats)
end

"""
Save a histogram of per-step solve times and a per-step bar chart colored by
convergence status. The first solve is excluded from the histogram when it is
dominated by one-time JIT compilation (> 3× the median warm solve); its value
is reported in the title instead so the warm-solve distribution stays readable.
"""
function save_solve_time_plots(step_solve_times, step_statuses, timing_plots_dir)
    CairoMakie.activate!()
    num_steps = length(step_solve_times)

    warm_times = step_solve_times[min(2, num_steps):end]
    exclude_first = num_steps > 2 && step_solve_times[1] > 3 * median(warm_times)
    hist_times = exclude_first ? step_solve_times[2:end] : step_solve_times
    hist_title =
        exclude_first ?
        "Warm-solve times (step 1 excluded: $(round(step_solve_times[1]; digits = 1)) s incl. one-time JIT)" :
        "Per-step solve times"
    hist_fig = serif_figure(size = (900, 600))
    hist_ax = CairoMakie.Axis(
        hist_fig[1, 1];
        xlabel = "solve time [s]",
        ylabel = "number of MPC steps",
        title = hist_title,
    )
    CairoMakie.hist!(
        hist_ax,
        hist_times;
        bins = max(6, ceil(Int, 2 * sqrt(length(hist_times)))),
        color = (:dodgerblue, 0.85),
        strokecolor = :black,
        strokewidth = 1,
    )
    save_figure(joinpath(timing_plots_dir, "solve_time_histogram.pdf"), hist_fig)

    bar_fig = serif_figure(size = (950, 600))
    bar_ax = CairoMakie.Axis(
        bar_fig[1, 1];
        xlabel = "MPC step",
        ylabel = "solve time [s]",
        title = "Solve time per MPC step (green = converged, red = not converged)",
    )
    CairoMakie.barplot!(
        bar_ax,
        1:num_steps,
        step_solve_times;
        color = [status == :solved ? :seagreen : :firebrick for status in step_statuses],
        strokecolor = :black,
        strokewidth = 0.5,
    )
    save_figure(joinpath(timing_plots_dir, "solve_time_per_step.pdf"), bar_fig)
end

# ── Output / plotting helpers ──────────────────────────────────────────────────

function _add_title!(figure, title_text)
    CairoMakie.Label(figure[0, :], title_text; fontsize = 20, tellwidth = false)
end

function _step_title(k, num_mpc_steps, converged, kkt_error)
    title = "MPC step $(k)/$(num_mpc_steps)"
    converged ? title :
    title * " — NOT CONVERGED (KKT error $(round(kkt_error; sigdigits = 3)))"
end

function save_step_plots(
    strategies,
    k,
    num_mpc_steps,
    converged,
    kkt_error,
    dirs;
    map_end,
    lane_width,
    θ1,
    θ2,
    θ3,
    goal_position1,
    goal_position2,
    goal_position3,
    collision_avoidance,
    dₚ,
    arm_speed_limit,
    child_speed_limit,
    control_bounds,
    dynamics_model,
    save_interactive,
)
    # Plots keep the per-arm view of the combined two-arm agent.
    plot_strategies = split_arm_strategies(strategies)
    status_suffix = converged ? "" : "_notconverged"
    title = _step_title(k, num_mpc_steps, converged, kkt_error)

    trajectory_fig, _ = plot_single_integrator_3d_trajectories(;
        map_end,
        lane_width,
        strategy = plot_strategies,
        θ1,
        θ2,
        θ3,
        goal_position1,
        goal_position2,
        goal_position3,
        collision_avoidance,
    )
    _add_title!(trajectory_fig, title)
    speed_fig, _ = speed_plot(;
        strategy = plot_strategies,
        speed_limit = arm_speed_limit,
        dynamics_model,
        speed_limit_players = 1:2,
        additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
    )
    _add_title!(speed_fig, title)
    control_fig, _ = control_plot(;
        strategy = plot_strategies,
        control_lb = control_bounds.lb,
        control_ub = control_bounds.ub,
    )
    _add_title!(control_fig, title)
    distance_fig, _ = inter_player_distance_plot(;
        strategy = plot_strategies,
        reference_distance = dₚ,
        safety_distance = collision_avoidance,
    )
    _add_title!(distance_fig, title)

    save_figure(
        joinpath(dirs.trajectory_plots_dir, "trajectory_step_$(k)$(status_suffix).pdf"),
        trajectory_fig,
    )
    save_figure(
        joinpath(dirs.speed_plots_dir, "speed_step_$(k)$(status_suffix).pdf"),
        speed_fig,
    )
    save_figure(
        joinpath(dirs.control_plots_dir, "control_step_$(k)$(status_suffix).pdf"),
        control_fig,
    )
    save_figure(
        joinpath(dirs.distance_plots_dir, "distance_step_$(k)$(status_suffix).pdf"),
        distance_fig,
    )

    if save_interactive
        interactive_path = joinpath(
            dirs.trajectory_plots_dir,
            "trajectory_interactive_step_$(k)$(status_suffix).html",
        )
        plot_trajectory_3d_interactive(;
            map_end,
            lane_width,
            strategy = plot_strategies,
            θ1,
            θ2,
            θ3,
            goal_position1,
            goal_position2,
            goal_position3,
            collision_avoidance,
            reference_distance = dₚ,
            display_figure = false,
            save_path = interactive_path,
        )
        println("saved interactive trajectory browser file: ", interactive_path)
    end
end

function save_closed_loop_plots(
    closed_loop_strategies,
    failed_steps,
    num_mpc_steps,
    dirs;
    map_end,
    lane_width,
    θ1,
    θ2,
    θ3,
    goal_position1,
    goal_position2,
    goal_position3,
    collision_avoidance,
    dₚ,
    arm_speed_limit,
    child_speed_limit,
    control_bounds,
    dynamics_model,
    save_interactive,
)
    plot_strategies = split_arm_strategies(closed_loop_strategies)
    title = "Closed-loop receding-horizon execution ($(num_mpc_steps) steps)"
    if !isempty(failed_steps)
        title *= " — non-converged steps: $(join(failed_steps, ", "))"
    end
    trajectory_fig, _ = plot_single_integrator_3d_trajectories(;
        map_end,
        lane_width,
        strategy = plot_strategies,
        θ1,
        θ2,
        θ3,
        goal_position1,
        goal_position2,
        goal_position3,
        collision_avoidance,
    )
    _add_title!(trajectory_fig, title)
    speed_fig, _ = speed_plot(;
        strategy = plot_strategies,
        speed_limit = arm_speed_limit,
        dynamics_model,
        speed_limit_players = 1:2,
        additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
    )
    _add_title!(speed_fig, title)
    control_fig, _ = control_plot(;
        strategy = plot_strategies,
        control_lb = control_bounds.lb,
        control_ub = control_bounds.ub,
    )
    _add_title!(control_fig, title)
    distance_fig, _ = inter_player_distance_plot(;
        strategy = plot_strategies,
        reference_distance = dₚ,
        safety_distance = collision_avoidance,
    )
    _add_title!(distance_fig, title)

    save_figure(
        joinpath(dirs.closed_loop_plots_dir, "trajectory_closed_loop.pdf"),
        trajectory_fig,
    )
    save_figure(joinpath(dirs.closed_loop_plots_dir, "speed_closed_loop.pdf"), speed_fig)
    save_figure(
        joinpath(dirs.closed_loop_plots_dir, "control_closed_loop.pdf"),
        control_fig,
    )
    save_figure(
        joinpath(dirs.closed_loop_plots_dir, "distance_closed_loop.pdf"),
        distance_fig,
    )

    if save_interactive
        interactive_path = joinpath(
            dirs.closed_loop_plots_dir,
            "trajectory_interactive_closed_loop.html",
        )
        plot_trajectory_3d_interactive(;
            map_end,
            lane_width,
            strategy = plot_strategies,
            θ1,
            θ2,
            θ3,
            goal_position1,
            goal_position2,
            goal_position3,
            collision_avoidance,
            reference_distance = dₚ,
            display_figure = false,
            save_path = interactive_path,
        )
        println(
            "saved closed-loop interactive trajectory browser file: ",
            interactive_path,
        )
    end
end

function prepare_receding_output_dirs(run_id; debug)
    run_dir = if debug
        joinpath("data", "robotic_arm_receding_horizon", "debug", run_id)
    else
        joinpath("data", "robotic_arm_receding_horizon", "runs", run_id)
    end

    data_dir = joinpath(run_dir, "data")
    problem_data_dir = joinpath(data_dir, "problem")
    solution_data_dir = joinpath(problem_data_dir, "solution")
    plots_dir = joinpath(run_dir, "plots")
    trajectory_plots_dir = joinpath(plots_dir, "trajectories")
    convergence_plots_dir = joinpath(plots_dir, "convergence")
    speed_plots_dir = joinpath(plots_dir, "speed")
    control_plots_dir = joinpath(plots_dir, "controls")
    distance_plots_dir = joinpath(plots_dir, "distance")
    warmstart_plots_dir = joinpath(plots_dir, "warmstart")
    closed_loop_plots_dir = joinpath(plots_dir, "closed_loop")
    timing_plots_dir = joinpath(plots_dir, "timing")

    for dir in (
        problem_data_dir,
        solution_data_dir,
        trajectory_plots_dir,
        convergence_plots_dir,
        speed_plots_dir,
        control_plots_dir,
        distance_plots_dir,
        warmstart_plots_dir,
        closed_loop_plots_dir,
        timing_plots_dir,
    )
        mkpath(dir)
    end

    (;
        run_dir,
        data_dir,
        problem_data_dir,
        solution_data_dir,
        plots_dir,
        trajectory_plots_dir,
        convergence_plots_dir,
        speed_plots_dir,
        control_plots_dir,
        distance_plots_dir,
        warmstart_plots_dir,
        closed_loop_plots_dir,
        timing_plots_dir,
    )
end

# ── MPC types and shared API ──────────────────────────────────────────────────
#
# Both the Julia demo() entry point and the Python closure share the same
# context / state / step machinery:
#
#   Julia:  scenario = demo_scenario_config(...)
#           ctx      = build_mpc_context(scenario; kwargs...)
#           state    = init_mpc_state(ctx)
#           while ...; result = step_mpc!(state, ctx, solver); end
#
#   Python: ctx      = jl.Robotic_arm_receding.build_mpc_context(obs; kwargs...)
#           plan_fn  = jl.Robotic_arm_receding.create_planner_from_context(ctx, 20; ...)
#           action   = np.array(plan_fn(obs))   # one interpolated waypoint per call
#           # returns Float64[] after num_mpc_steps MPC solves → done
#
# The only structural difference is the output mechanism: demo() collects a
# full closed-loop trajectory and saves plots; create_planner_from_context
# streams linearly-interpolated waypoints for OSC tracking.
#
# Observation feedback hook: plan_fn(obs) receives the simulator observation
# on every call. State is propagated from the plan by default. To enable
# closed-loop sim correction, update state.arm_state1/2 / state.child_state
# from obs before calling step_mpc! (see the comment inside create_planner_from_context).

"""
Build the KKT system and solver options from a pre-constructed `ScenarioConfig`.

This is the shared implementation used by both `demo` (Julia entry point) and
the Python-facing `build_mpc_context(obs; ...)` wrapper below.
"""
function build_mpc_context(
    scenario_config::ScenarioConfig;
    goop_version::Symbol = :quasi,
    dual_warmstart::Symbol = :all_except_innermost_stationarity,
    fd_codegen_chunk_size::Integer = 128,
    tol::Float64 = 0.008,
    max_inner_iters::Integer = 500,
    linear_solver::Symbol = :klu,
    klu_singularity_eta_growth::Float64 = 100.0,
    η_max::Float64 = 1e2,
    η₀::Float64 = 1e-6,
    ϵ₀::Float64 = 0.1,
    armijo_constant::Float64 = 1e-4,
    reuse_factorization_iters::Integer = 0,
    record_convergence::Bool = false,
    record_condition_number::Bool = false,
    verbose::Bool = false,
)
    dual_warmstart in DUAL_WARMSTART_MODES || throw(
        ArgumentError(
            "Unknown dual_warmstart mode $(dual_warmstart); expected one of " *
            "$(join(DUAL_WARMSTART_MODES, ", ")).",
        ),
    )

    (; dynamics, planning_horizon) = scenario_config
    arm_state_dimension, _ = divrem(dynamics[1].state_dimension, 2)
    arm_control_dimension, _ = divrem(dynamics[1].control_dimension, 2)

    problem_setup_time_sec = @elapsed (; problem, flatten_parameters) = get_setup(scenario_config)

    kkt_generators = Dict(
        :complete => ReducedGOOP.generate_slacked_complete_kkt_system,
        :reduced => ReducedGOOP.generate_slacked_reduced_kkt_system,
        :quasi => ReducedGOOP.generate_slacked_quasi_kkt_system,
    )
    GOOP_kkt_generator = get(kkt_generators, goop_version, nothing)
    isnothing(GOOP_kkt_generator) && error("Unknown GOOP version: $(goop_version)")

    @info "Building $(goop_version) KKT system (T = $(planning_horizon))..."
    kkt_build_time_sec = @elapsed GOOP_kkt_system = GOOP_kkt_generator(
        problem;
        backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
        backend_options = (;),
        codegen = :fast_differentiation,
        fd_codegen_chunk_size,
    )
    println(
        "context build: problem $(round(problem_setup_time_sec; digits = 2)) s, ",
        "KKT $(round(kkt_build_time_sec; digits = 2)) s",
    )
    println("[Primal-Dual] KKT dimension: ", GOOP_kkt_system.kkt_dimension)
    println("[Primal-Dual] variable dimension: ", GOOP_kkt_system.variable_dimension)
    total_eq_stat_duals = length(GOOP_kkt_system.all_equality_stationarity_dual_dims)
    selected_duals = if dual_warmstart === :primal_only
        0
    elseif dual_warmstart === :equality_duals
        length(GOOP_kkt_system.equality_constraint_dual_dims)
    else
        total_eq_stat_duals - length(GOOP_kkt_system.innermost_stationarity_dual_dims)
    end
    println(
        "dual warm-start: $(dual_warmstart) ",
        "($(selected_duals)/$(total_eq_stat_duals) equality/stationarity duals carried)",
    )

    primal_dimensions = [
        (dyn.state_dimension + dyn.control_dimension) * planning_horizon
        for dyn in dynamics
    ]

    options = ReducedGOOP.InteriorPointOptions(;
        tol,
        η₀,
        η_max,
        ϵ₀,
        max_inner_iters,
        max_outer_iters = 1,
        tightening_rate = 1.2,
        loosening_rate = 3.0,
        min_stepsize = 1e-20,
        linesearch = :backtracking,
        linear_solver,
        klu_singularity_eta_growth,
        armijo_constant,
        reuse_factorization_iters,
        eta_retry_growth = 2.0,
        ρ_low = 0.75,
        ρ_high = 0.75,
        tsvd_threshold = 0.0,
        use_marquardt_scaling = false,
        record_convergence,
        record_condition_number,
        verbose,
    )

    MpcContext(
        GOOP_kkt_system,
        scenario_config,
        flatten_parameters,
        primal_dimensions,
        options,
        Int(planning_horizon),
        Int(arm_state_dimension),
        Int(arm_control_dimension),
        dual_warmstart,
        problem_setup_time_sec,
        kkt_build_time_sec,
    )
end

"""
Return a stateful `plan_fn(obs) -> Vector{Float64}` closure backed by an `MpcState`.
To be used in Python as a callable that returns one interpolated waypoint per call for OSC tracking.

Each call drains one pre-computed waypoint from an internal interpolation buffer.
The buffer is refilled by one `step_mpc!` call every
`ratio = low_level_freq ÷ planner_freq` simulator steps, producing `ratio`
linearly interpolated positions for smooth OSC tracking.

Returns `Float64[]` after `num_mpc_steps` MPC solves to signal completion.
"""
function create_planner_from_context(
    ctx::MpcContext,
    num_mpc_steps::Integer = 20;
    planner_freq::Integer = 10,
    low_level_freq::Integer = 10,
)
    low_level_freq >= planner_freq ||
        error("low_level_freq ($(low_level_freq)) must be ≥ planner_freq ($(planner_freq))")
    ratio = div(low_level_freq, planner_freq)
    low_level_freq == planner_freq * ratio ||
        @warn "low_level_freq $(low_level_freq) is not an exact multiple of planner_freq $(planner_freq); using ratio=$(ratio)"

    state  = init_mpc_state(ctx)
    solver = ReducedGOOP.InteriorPoint()
    buffer = Vector{Float64}[]

    function plan_next_fn(obs)
        if isempty(buffer)
            state.step >= num_mpc_steps && return Float64[]

            prev_arm1  = copy(state.arm_state1)
            prev_arm2  = copy(state.arm_state2)
            prev_child = copy(state.child_state)

            # ── Sim-feedback correction hook ───────────────────────────────────
            # Uncomment to re-anchor state from actual EEF positions each step:
            # state.arm_state1  = collect(Float64, obs["robot0_eef_pos"])
            # state.arm_state2  = collect(Float64, obs["robot1_eef_pos"])
            # state.child_state = collect(Float64, obs["robot2_eef_pos"])

            result = step_mpc!(
                state, ctx, solver;
                rescue_failed_steps = true,
                step_label = "step $(state.step + 1): ",
            )

            (; goal_position1, goal_position2) = ctx.scenario_config
            dist1 = sqrt(sum(abs2, state.arm_state1 .- goal_position1))
            dist2 = sqrt(sum(abs2, state.arm_state2 .- goal_position2))
            output = result.output
            println(
                "[mpc] step=$(state.step)/$(num_mpc_steps), status=$(output.status), ",
                "iters=$(output.total_iters), ",
                "kkt_err=$(round(output.kkt_error; sigdigits = 4)), ",
                "time=$(round(result.elapsed_time; digits = 3)) s | ",
                "arm1→goal=$(round(dist1; digits = 4)) m  ",
                "arm2→goal=$(round(dist2; digits = 4)) m",
            )

            # Create a linear interpolation buffer of `ratio` waypoints
            # between the previous and current states for smooth OSC tracking
            for i in 1:ratio
                α = i / ratio
                push!(buffer, vcat(
                    prev_arm1  .+ α .* (state.arm_state1  .- prev_arm1),  [1.0],
                    prev_arm2  .+ α .* (state.arm_state2  .- prev_arm2),  [1.0],
                    prev_child .+ α .* (state.child_state .- prev_child),
                ))
            end
        end
        popfirst!(buffer)
    end

    plan_next_fn
end

"""Return goal positions for overlay visualization."""
function get_current_goal_positions(ctx::MpcContext, obs)::Vector{Vector{Float64}}
    (; goal_position1, goal_position2) = ctx.scenario_config
    eef0 = collect(Float64, obs["robot0_eef_pos"])
    eef1 = collect(Float64, obs["robot1_eef_pos"])
    [collect(Float64, goal_position1), collect(Float64, goal_position2), 0.5 .* (eef0 .+ eef1)]
end
export get_current_goal_positions

end
