function get_problem_for_experiment(
    ::Type{ParametricGameClassifier},
    experiment::HighwayExperiment,
)
    (;
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

    problem = ParametricGameClassifier(;
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
    )

    return problem
end

function generate_samples(num_samples = 10; problem = nothing)
    relaxably_feasible_problems_folder = "./data/relaxably_feasible/problems/"
    trivially_feasible_problems_folder = "./data/trivially_feasible/problems/"
    inspection_needed_problems_folder = "./data/inspection_needed/problems/"

    experiment = HighwayExperiment()

    if isnothing(problem)
        problem = get_problem_for_experiment(ParametricGameClassifier, experiment)
    end

    Random.seed!(123)

    samples = []
    num_relaxably_feasible, num_trivially_feasible = 0, 0

    obstacle_position = [0.25, 0.15]

    while num_relaxably_feasible < num_samples
        # Sample a random number between x and y = x + rand()*(y-x)
        initial_state1 = [-0.3, -0.2 + rand() * 0.4, 0.4 + rand() * (0.8 - 0.4), 0.0]
        initial_state2 = [0.1, -0.2 + rand() * 0.4, 0.2, 0.0]
        initial_state3 = [0.4, -0.2 + rand() * 0.4, 0.2, 0.0]

        # Goal positions
        goal_position1 = [0.9, 0.0]
        goal_position2 = [0.9, 0.0]
        goal_position3 = [0.9, 0.0]

        # Parameters
        θ1 = flatten_parameters(;
            initial_state = initial_state1,
            goal_position = goal_position1,
            obstacle_position = obstacle_position,
        )

        θ2 = flatten_parameters(;
            initial_state = initial_state2,
            goal_position = goal_position2,
            obstacle_position = obstacle_position,
        )

        θ3 = flatten_parameters(;
            initial_state = initial_state3,
            goal_position = goal_position3,
            obstacle_position = obstacle_position,
        )

        θ = [θ1..., θ2..., θ3...]

        # Classify problem instance
        @info "Classifying problem..."
        solution = classify_game(problem, θ; return_primals = true)

        # Define data for saving
        saved_data = Dict(
            "obstacle_position" => obstacle_position,
            "initial_state1" => initial_state1,
            "initial_state2" => initial_state2,
            "initial_state3" => initial_state3,
            "goal_position1" => goal_position1,
            "goal_position2" => goal_position2,
            "goal_position3" => goal_position3,
        )

        # 1. If no sol, this is infeasible problem and we drop the instance
        if !solution.solved
            @info "Infeasible problem...moving on to the next problem"
            continue

            # 2. If sol with positive objective, this is a relaxably feasible problem and goes to "Relaxably Feasible Set" 
        elseif any(x -> x > 0, solution.objectives)
            num_relaxably_feasible += 1
            @info "Relaxably feasible problem...saving the instance #$num_relaxably_feasible"
            file_name = joinpath(
                relaxably_feasible_problems_folder,
                "rfp_$num_relaxably_feasible.jld2",
            )

            # 3. If sol with zero objective, this is a trivially feasible problem and goes to "Trivially Feasible Set"
        elseif all(x -> isapprox(x, 0.0; atol = 1e-4), solution.objectives)
            num_trivially_feasible += 1

            # Limit the number of trivially feasible solutions saved
            if num_trivially_feasible >= num_samples
                @info "Trivially feasible problem...skipping the instance #$num_trivially_feasible"
                continue
            end

            @info "Trivially feasible problem...saving the instance #$num_trivially_feasible"
            file_name = joinpath(
                trivially_feasible_problems_folder,
                "tfp_$num_trivially_feasible.jld2",
            )

        else
            @info "Inspection needed for this problem..."
            file_name =
                joinpath(inspection_needed_problems_folder, "inspection_needed.jld2")
        end

        @info solution.objectives

        mkpath(dirname(file_name))
        JLD2.save_object(file_name, saved_data)
    end

    for sample in samples
        @info sample
    end

    return (; problem)
end
