using Dates

function get_problem_for_experiment(
    ::Type{ParametricOrderedPreferencesMPCCGame},
    experiment::HighwayExperiment;
    relaxation_mode = :standard,
    solver = "PATH"
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
        equality_dimensions,
        inequality_dimensions,
    ) = experiment

    problem = ParametricOrderedPreferencesMPCCGame(;
        objectives,
        equality_constraints,
        inequality_constraints,
        prioritized_preferences,
        is_prioritized_constraint,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        equality_dimensions,
        inequality_dimensions,
        relaxation_mode,
        solver,
    )

    return problem
end

function run(
    ::Type{ParametricOrderedPreferencesMPCCGame};
    num_samples = 10,
    warmstart_samples = 20,
    should_check_equilibrium = false,
    out_prefix = string(Dates.format(now(), "yyyymmdd_HHMMSS"), "_goop"),
    verbose = false,
    problem = nothing,
    solver = "PATH", # Change to IP to try the other one
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
    ϵ = 1.1
    κ = 0.1
    max_iterations = 6
    tolerance = 5e-2
    num_perturb = 20

    # Tracking not-converged instances
    not_converged = []

    # Record equilibria tally of found solutions
    equilbrium_tallies = []

    # Run-time record
    runtime = Float64[]

    experiment = HighwayExperiment()

    if isnothing(problem)
        # Get the problem modeled as a ParametricOrderedPreferencesMPCCGame
        problem = get_problem_for_experiment(ParametricOrderedPreferencesMPCCGame, experiment; solver)
    end

    warmstart_solution = nothing

    # Handy access to whatever information about experiment conditions we require down the line
    (; num_players, dynamics, planning_horizon) = experiment

    # Run the experiment
    @showprogress desc = "Running problem instances using GOOP..." for i_sample in 1:num_samples
        @info "Solving problem instance #$i_sample..."

        # Load problem data
        (; θ, initial_states) = load_problem_parameters(problems_folder, "rfp_$i_sample.jld2")

        # Generate multiple equilibrium solutions
        @showprogress desc = "    Using different initial guesses..." for i_version in
                                                                          1:warmstart_samples
            @info "        Using GOOP with warmstart solution $i_version..."

            # initial guess is all zeros
            if i_version > 1
                # using a random control sequence
                warmstart_x = [[initial_states[1]], [initial_states[2]], [initial_states[3]]]
                warmstart_u = [[], [], []]
                warmstart_solution = []

                rand_u = hcat(2.0 * rand(num_players), 4 * rand(num_players) .- 2.0)
                for k in 1:num_players
                    for i in 1:(planning_horizon - 1)
                        push!(warmstart_u[k], rand_u[k, :])
                        push!(warmstart_x[k], dynamics(warmstart_x[k][i], rand_u[k, :]))
                    end
                    push!(warmstart_u[k], [0.0, 0.0])

                    warmstart_primals = mapreduce(vcat, 1:planning_horizon) do i
                        vcat(warmstart_x[k][i], warmstart_u[k][i])
                    end
                    push!(warmstart_solution, warmstart_primals)
                end
                warmstart_solution = vcat(warmstart_solution...)
            end

            # Measure run time
            elapsed_time = @elapsed begin
                result = get_receding_horizon_solution(
                    problem,
                    θ,
                    planning_horizon,
                    dynamics;
                    ϵ,
                    κ,
                    max_iterations,
                    tolerance,
                    warmstart_solution,
                    verbose
                )
            end
            push!(runtime, elapsed_time)

            # If not solved, then continue to next problem instance
            if isnothing(result)
                @warn "GOOP could not find a solution...moving on to the next problem"
                push!(not_converged, i_sample)
                continue
            end

            (; strategies, info) = result

            # Save solution
            file = joinpath(solutions_folder, "rfp_$(i_sample)_$(i_version)_sol.jld2")
            save_object(file, info)

            if should_check_equilibrium
                # Check if the solution is an equilibrium
                tally = check_equilibrium(experiment, θ, strategies; num_perturb)

                # Check goop equilibrium data
                @info "goop solution for prob #$(i_sample)_w$i_version is equilibrium in $tally cases (out of $num_perturb)"

                push!(equilbrium_tallies, tally)
            end

            # Plot and save relevant representations of the strategies found
            plot_filename_base = "$(i_sample)_$i_version.png"
            plot_strategies(experiment, strategies, plots_folder, plot_filename_base)
        end
    end

    # Save not-converged instances
    if !isempty(not_converged)
        file = joinpath(data_folder, "rfp_GOOP_not_converged.jld2")
        save_object(file, not_converged)
    end
    @info string("not-converged instances: ", length(not_converged))

    # Save equilibrium tally
    file = joinpath(solutions_folder, "rfp_equilibrium.jld2")
    save_object(file, equilbrium_tallies)

    # Save runtime
    file = joinpath(runtime_folder, "rfp_runtime.jld2")
    save_object(file, runtime)

    return (; problem)
end

function check_equilibrium(experiment, θ, strategy; num_perturb::Int = 20, tol = 2e-2)
    (;
        num_players,
        planning_horizon,
        dynamics,
        inequality_constraints,
        shared_inequality_constraints,
        shared_equality_constraints,
        prioritized_preferences,
    ) = experiment

    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    # Distribution for sampling feasible trajectories
    Random.seed!(123)
    dist = Normal(0.0, 0.01)

    count_equilibrium_goop = 0
    goop_z = BlockArray(
        mapreduce(vcat, 1:num_players) do i
            mapreduce(vcat, 1:planning_horizon) do k
                vcat(strategy[i].xs[k], strategy[i].us[k])
            end
        end,
        fill(primal_dimension, num_players),
    )

    θ_blocked = BlockArray(θ, fill(Int(length(θ) / num_players), num_players))

    @showprogress desc = "  Checking equilibrium..." for ll in 1:num_perturb
        perturbed_x = [[strategy[1].xs[1]], [strategy[2].xs[1]], [strategy[3].xs[1]]]
        perturbed_u = [[], [], []]

        for kk in 1:num_players
            @info "   Checking x*...@perturbation #$ll, player #$kk"

            perturbed_z_block = []
            check_inequality, check_shared_equality, check_shared_inequality = false, false, false
            while !(check_inequality && check_shared_equality && check_shared_inequality)
                # Step 1: Perturb control sequence u by ω = rand(dist, n_size) and generate perturbed trajectory x
                for i in 1:(planning_horizon - 1)
                    local u = strategy[kk].us[i] + rand(dist, control_dim(dynamics))
                    push!(perturbed_u[kk], u)
                    push!(perturbed_x[kk], dynamics(perturbed_x[kk][i], u))
                end
                push!(perturbed_u[kk], [0.0, 0.0])
                # Step 2: Check if the perturbed trajectory x satisfies shared constraints and inequality constraints
                # Rejection sampling. Feasible perturbations.Fix others' strategy constant and perturb one player's strategy
                perturbed_z_block = mapreduce(vcat, 1:planning_horizon) do i
                    vcat(perturbed_x[kk][i], perturbed_u[kk][i])
                end
                check_inequality =
                    all(inequality_constraints[kk](perturbed_z_block, θ_blocked) .≥ -tol)

                perturbed_z = let
                    z_temp = copy(goop_z)
                    z = blocks(z_temp)
                    z[kk] = perturbed_z_block
                    mortar(z)
                end
                check_shared_inequality =
                    all(shared_inequality_constraints(perturbed_z, θ_blocked) .≥ -tol)
                check_shared_equality =
                    all(shared_equality_constraints(perturbed_z, θ_blocked) .== 0.0)

                # Initialize perturbed trajectory for next iteration
                perturbed_x = [[strategy[1].xs[1]], [strategy[2].xs[1]], [strategy[3].xs[1]]]
                perturbed_u = [[], [], []]
            end

            # Step 3: Check if f₃(x*, θ) < f₃(x, θ) in the neighborhood of x*
            f₃_star = sum(max.(0, -prioritized_preferences[kk][1](goop_z[Block(kk)], θ_blocked)))
            f₃ = sum(max.(0, -prioritized_preferences[kk][1](perturbed_z_block, θ_blocked)))
            if f₃_star < f₃
                @info "    f₃(x*, θ) < f₃(x, θ) in the neighborhood of x* for player #$kk"
            elseif isapprox(f₃_star, f₃, atol = tol)
                @info "    f₃(x*, θ) close to f₃(x, θ) for player #$kk"
                @info "    |f₃(x*, θ) - f₃(x, θ)| = $(abs(f₃_star - f₃))"

                # Step 4: Check if f₂(x*, θ) > f₂(x, θ) in the neighborhood of x*
                f₂_star =
                    sum(max.(0, -prioritized_preferences[kk][2](goop_z[Block(kk)], θ_blocked)))
                f₂ = sum(max.(0, -prioritized_preferences[kk][2](perturbed_z_block, θ_blocked)))

                @info begin
                    if f₂_star < f₂
                        "    f₂(x*, θ) < f₂(x, θ) in the neighborhood of x* for player #$kk"
                    elseif isapprox(f₂_star, f₂, atol = tol)
                        "    f₂(x*, θ) close to f₂(x, θ) for player #$kk"
                        "    |f₂(x*, θ) - f₂(x, θ)| = $(abs(f₂_star - f₂))"
                    else
                        "    f₂(x*, θ) > f₂(x, θ) SOMETHING IS WRONG for player #$kk"
                    end
                end
            else
                @info "    f₃(x*, θ) > f₃(x, θ) SOMETHING IS WRONG for player #$kk"
            end

            # Step 5: Check if x* is an equilibrium
            if f₃_star < f₃ || (
                isapprox(f₃_star, f₃, atol = tol) &&
                (f₂_star < f₂ || isapprox(f₂_star, f₂, atol = tol))
            )
                @info "   x* is a GOOP equilibrium...@perturbation #$ll for player #$kk"
                count_equilibrium_goop += 1
            else
                @info "   x* is not a GOOP equilibrium...@perturbation #$ll for player #$kk"
            end
        end
    end

    return count_equilibrium_goop / num_players
end

function get_receding_horizon_solution(
    problem::ParametricOrderedPreferencesMPCCGame,
    θ,
    planning_horizon,
    dynamics;
    ϵ,
    κ,
    max_iterations,
    tolerance,
    warmstart_solution,
    verbose
)
    (; relaxation, solution, residual) =
        solve_relaxed_pop_game(problem, warmstart_solution, θ; ϵ, κ, max_iterations, tolerance, verbose)

    if all([!solution[i].solved for i in 1:first(size(solution))])
        return nothing
    end

    # Extra information needed to unflatten the trajectories
    dynamics_dimension = state_dim(dynamics) + control_dim(dynamics)
    primal_dimension = dynamics_dimension * planning_horizon

    # Choose the solution with best complementarity residual
    min_residual_idx = argmin(residual)

    # Build strategies from the solution
    strategies = reduce(
        vcat,
        unflatten_trajectory(
            primals[1:primal_dimension],
            state_dim(dynamics),
            control_dim(dynamics),
        ) for primals in solution[min_residual_idx].primals
    )

    # Build dictionary with relevant information about the solution
    info = Dict(
        "residual" => residual[min_residual_idx],
        "relaxation" => relaxation[min_residual_idx],
        "slacks" => solution[min_residual_idx].slacks,
        "strategy1" => strategies[1],
        "strategy2" => strategies[2],
        "strategy3" => strategies[3],
        "primals" => solution[min_residual_idx].primals,
    )

    (; solution, strategies, info)
end
