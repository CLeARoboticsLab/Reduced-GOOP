using Dates

function get_problem_for_experiment(::Type{ParametricGamePenalty}, experiment::HighwayExperiment; α)
    (;
        num_players,
        objectives,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        is_prioritized_constraint,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        shared_equality_dimension,
        shared_inequality_dimension,
    ) = experiment

    num_levels = length(prioritized_preferences)
    penalty_weights = [[Float64(α^i) for i in (num_levels - 1):-1:0] for _ in 1:num_players]

    # TODO PHM: It might be nice to do
    # ParametricGamePenalty(; penalty_weights, experiment...)

    problem = ParametricGamePenalty(;
        objectives,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        is_prioritized_constraint,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        shared_equality_dimension,
        shared_inequality_dimension,
        penalty_weights,
    )

    return problem
end

# Make call more handy
function run(
    ::Type{ParametricGamePenalty};
    num_samples = 10,
    pareto = false,
    out_prefix = string(Dates.format(now(), "yyyymmdd_HHMMSS"), "_baseline"),
    problem = nothing,
)
    data_folder = "./data/"
    relaxably_feasible_folder = joinpath(data_folder, "relaxably_feasible")
    problems_folder = joinpath(relaxably_feasible_folder, "problems")

    main_output_folder = joinpath(relaxably_feasible_folder, out_prefix)
    plots_folder = joinpath(main_output_folder, "plots")
    solutions_folder = joinpath(main_output_folder, "solutions")
    runtime_folder = joinpath(main_output_folder, "runtime")

    @info "Output directory: $main_output_folder"

    # Algorithm setting
    if !pareto
        alfas = [1, 10, 20, 30, 40, 50]
    else
        alfas = [α for α in 1:50]
    end

    # Tracking not-converged instances
    not_converged = []

    # Run-time record
    runtime = Float64[]

    experiment = HighwayExperiment()

    # Handy access to whatever information about experiment conditions we require down the line
    (; dynamics, planning_horizon) = experiment

    # Run the baseline experiment
    @showprogress desc = "Running problem instances using baseline..." for i_sample in 1:num_samples
        @info "Solving problem instance #$i_sample..."

        @showprogress desc = "    Using different penalty weight factors α..." for (i_version, α) in
                                                                                   enumerate(alfas)
            @info "        Using baseline with penalty weight factor α = $α (#$i_version)..."

            if isnothing(problem)
                # Get the problem modeled as a ParametricGamePenalty
                problem = get_problem_for_experiment(ParametricGamePenalty, experiment; α)
            end

            # Load problem parameters
            (; θ) = load_problem_parameters(problems_folder, "rfp_$i_sample.jld2")

            # Measure run time
            elapsed_time = @elapsed begin
                result = get_receding_horizon_solution(problem, θ, planning_horizon, dynamics;)
            end
            push!(runtime, elapsed_time)

            # If not solved, then continue to next problem instance
            if isnothing(result)
                @info "Baseline #$i_version could not find a solution...moving on to the next problem"
                push!(not_converged, i_sample)
                continue
            end

            (; strategies, info) = result

            # Save solution
            file = joinpath(solutions_folder, "rfp_$(i_sample)_$(i_version)_sol.jld2")
            save_object(file, info)

            # Plot and save relevant representations of the strategies found
            if !pareto
                plot_filename_base = "$(i_sample)_$i_version.png"
                plot_strategies(experiment, strategies, plots_folder, plot_filename_base)
            end

            # Save not-converged instances
            if !isempty(not_converged)
                file = joinpath(data_folder, "rfp_$(i_sample)_$(i_version)_not_converged.jld2")
                save_object(file, not_converged)
                not_converged = []
            end
        end

        # Save runtime
        file = joinpath(runtime_folder, "rfp_runtime_$(i_sample).jld2")
        save_object(file, runtime)
    end

    return (; problem)
end

function get_receding_horizon_solution(
    problem::ParametricGamePenalty,
    θ,
    planning_horizon,
    dynamics;
    initial_guess = nothing,
)
    solution = solve_penalty(problem, θ; initial_guess, return_primals = true)

    if !solution.solved
        return nothing
    end

    # Extra information needed to unflatten the trajectories
    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    # Build strategies from the solution
    strategies = reduce(
        vcat,
        unflatten_trajectory(
            primals[1:primal_dimension],
            state_dim(dynamics),
            control_dim(dynamics),
        ) for primals in solution.primals
    )

    # Build dictionary with relevant information about the solution
    info = Dict(
        "slacks" => solution.slacks,
        "strategy1" => strategies[1],
        "strategy2" => strategies[2],
        "strategy3" => strategies[3],
        "primals" => solution.primals,
        #"experiment" => experiment, # Not everything can be saved, maybe not needed?
    )

    (; solution, strategies, info)
end