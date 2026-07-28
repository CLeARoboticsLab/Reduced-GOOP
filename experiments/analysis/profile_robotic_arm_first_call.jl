using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)

using BlockArrays
using Profile
using ReducedGOOP
using SparseArrays
using TimerOutputs: reset_timer!

include(joinpath(REPO_ROOT, "experiments", "Robotic_arm.jl"))

function initialized_kkt_point(mcp, primal_warmstart)
    z = zeros(mcp.variable_dimension)
    z[mcp.preference_slack_dims] .= 1.0
    z[mcp.interior_point_slack_dims] .= 1.0
    z[mcp.inequality_constraint_dual_dims] .= 1.0
    z[mcp.primal_dims] .= primal_warmstart
    z
end

function measure_call(label, f; profile = false)
    Base.@nospecialize f
    GC.gc()
    profile && Profile.clear()
    compile_before = Base.cumulative_compile_time_ns()
    result = if profile
        Profile.@profile @timed Base.invokelatest(f)
    else
        @timed Base.invokelatest(f)
    end
    compile_after = Base.cumulative_compile_time_ns()
    compile_delta = (compile_after[1] - compile_before[1]) / 1e9
    recompile_delta = (compile_after[2] - compile_before[2]) / 1e9
    println(
        label,
        ": wall=",
        round(result.time; digits = 6),
        " s, @timed_compile=",
        round(result.compile_time; digits = 6),
        " s, cumulative_compile_delta=",
        round(compile_delta; digits = 6),
        " s, recompile_delta=",
        round(recompile_delta; digits = 6),
        " s, allocations=",
        Base.format_bytes(result.bytes),
        ", gc=",
        round(result.gctime; digits = 6),
        " s",
    )
    if profile
        data = Profile.fetch()
        println("profile buffer entries: ", length(data))
        Profile.print(
            stdout,
            data;
            format = :flat,
            C = true,
            combine = true,
            sortedby = :count,
            mincount = max(5, length(data) ÷ 2000),
        )
    end
    result
end

function finite_difference_jacobian_check(mcp, J, z, θ; step = 1e-6)
    base = zeros(mcp.kkt_dimension)
    trial = similar(base)
    z_trial = copy(z)
    mcp.F!(base, z; θ, ϵ = 0.1, η = 0.0)
    rows = rowvals(J)
    values = nonzeros(J)
    structural_rows = falses(mcp.kkt_dimension)
    max_structural_error = 0.0
    max_scaled_error = 0.0
    max_off_pattern_derivative = 0.0
    for col in axes(J, 2)
        z_trial[col] += step
        mcp.F!(trial, z_trial; θ, ϵ = 0.1, η = 0.0)
        z_trial[col] = z[col]
        fill!(structural_rows, false)
        for ptr in nzrange(J, col)
            row = rows[ptr]
            structural_rows[row] = true
            finite_difference = (trial[row] - base[row]) / step
            error = abs(values[ptr] - finite_difference)
            max_structural_error = max(max_structural_error, error)
            max_scaled_error = max(
                max_scaled_error,
                error / max(1.0, abs(values[ptr]), abs(finite_difference)),
            )
        end
        for row in eachindex(trial)
            if !structural_rows[row]
                max_off_pattern_derivative =
                    max(max_off_pattern_derivative, abs((trial[row] - base[row]) / step))
            end
        end
    end
    (; max_structural_error, max_scaled_error, max_off_pattern_derivative)
end

function main()
    reset_timer!(ReducedGOOP.TO)
    Base.cumulative_compile_timing(true)

    scenario_config = Robotic_arm.demo_scenario_config()
    (; problem, flatten_parameters) = Robotic_arm.get_setup(scenario_config)
    (; dynamics, planning_horizon) = scenario_config
    println("planning horizon: ", planning_horizon)

    states = (;
        initial_state1 = scenario_config.base_initial_state1,
        initial_state2 = scenario_config.base_initial_state2,
        initial_state3 = scenario_config.initial_state3,
    )
    warmstart =
        Robotic_arm.build_default_warmstart(states, scenario_config).warmstart_solution
    primal_dimensions = [
        (dyn.state_dimension + dyn.control_dimension) * planning_horizon for
        dyn in dynamics
    ]
    initial_controls =
        Robotic_arm.extract_initial_controls(warmstart, primal_dimensions, dynamics)
    parameters = Robotic_arm.build_instance_parameters(
        flatten_parameters,
        merge(states, initial_controls),
        scenario_config,
    )

    println("building symbolic KKT system...")
    chunk_size = let value = get(ENV, "FD_CODEGEN_CHUNK_SIZE", "")
        isempty(value) ? nothing : parse(Int, value)
    end
    println("FastDifferentiation chunk size: ", something(chunk_size, "monolithic"))
    build = @timed ReducedGOOP.generate_slacked_reduced_kkt_system(
        problem;
        backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
        backend_options = (;),
        codegen = :fast_differentiation,
        fd_codegen_chunk_size = chunk_size,
    )
    mcp = build.value
    println(
        "KKT build: wall=",
        round(build.time; digits = 3),
        " s, compile=",
        round(build.compile_time; digits = 3),
        " s, allocations=",
        Base.format_bytes(build.bytes),
    )
    println(
        "dimensions: residual=",
        mcp.kkt_dimension,
        ", variables=",
        mcp.variable_dimension,
        ", Jacobian nnz=",
        nnz(mcp.∇F_z!.result_buffer),
    )
    if get(ENV, "KKT_BUILD_ONLY", "0") == "1"
        println("\nTimerOutputs breakdown:")
        show(ReducedGOOP.TO)
        println()
        Base.cumulative_compile_timing(false)
        return
    end

    z = initialized_kkt_point(mcp, warmstart)
    θ = parameters.θ
    F = zeros(mcp.kkt_dimension)
    J = mcp.∇F_z!.result_buffer

    measure_call("residual first call", () -> mcp.F!(F, z; θ, ϵ = 0.1, η = 0.0))
    measure_call("residual second call", () -> mcp.F!(F, z; θ, ϵ = 0.1, η = 0.0))
    measure_call(
        "Jacobian first call",
        () -> mcp.∇F_z!(J, z; θ, ϵ = 0.1, η = 0.0);
        profile = true,
    )
    measure_call("Jacobian second call", () -> mcp.∇F_z!(J, z; θ, ϵ = 0.1, η = 0.0))
    verification = @timed finite_difference_jacobian_check(mcp, J, z, θ)
    println(
        "finite-difference verification: wall=",
        round(verification.time; digits = 3),
        " s, max_structural_error=",
        verification.value.max_structural_error,
        ", max_scaled_error=",
        verification.value.max_scaled_error,
        ", max_off_pattern_derivative=",
        verification.value.max_off_pattern_derivative,
    )

    println("\nTimerOutputs breakdown:")
    show(ReducedGOOP.TO)
    println()
    Base.cumulative_compile_timing(false)
end

main()
