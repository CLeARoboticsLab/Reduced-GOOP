"Generic description of a constrained parametric game problem."
struct ParametricGamePenalty{T1,T2,T3,T4,T5,T6,T7,T8,T9}
    "Objective functions for all players"
    objectives::T1
    "Equality constraints for all players"
    equality_constraints::T2
    "Inequality constraints for all players"
    inequality_constraints::T3
    "Shared equality constraint"
    shared_equality_constraint::T4
    "Shared inequality constraint"
    shared_inequality_constraint::T5

    "Dimension of parameter vector"
    parameter_dimension::T6
    "Dimension of primal variables for all players"
    primal_dimensions::T7
    "Dimension of equality constraints for all players"
    equality_dimensions::T7
    "Dimension of inequality constraints for all players"
    inequality_dimensions::T7
    "Dimension of shared equality constraint"
    shared_equality_dimension::T8
    "Dimension of shared inequality constraint"
    shared_inequality_dimension::T8

    "Corresponding ParametricMCP."
    parametric_mcp::ParametricMCP

    "Sum of slacks for each preference"
    sum_slacks::T9
end

function ParametricGamePenalty(;
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
    @assert !isnothing(equality_constraints)
    @assert !isnothing(inequality_constraints)

    dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions) #Block(1), Block(2),... original primals for each player
    dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

    ordered_priority_levels = eachindex(prioritized_preferences[1]) # assume same number of levels for all players
    num_levels = length(ordered_priority_levels) + 1
    num_players = length(objectives)

    # Store objectives
    objectives_w_penalty = Symbolics.Num[]
    inequality_constraints_w_preferences = Vector{Symbolics.Num}[]
    equality_constraints_encoded = Vector{Symbolics.Num}[]

    # Store dimensions
    private_primals = [[dim] for dim in primal_dimensions]

    # Store (callable) slacks
    sum_slacks = Function[]

    primal_dimension_ii = 0
    start_idx = 1

    # Main.@infiltrate

    function set_up_penalty(priority_level, player_idx)
        # Account for slack variables in the primal dimension at each priority level
        prioritized_constraints_ii = prioritized_preferences[player_idx][priority_level]
        slack_dimension_ii =
            length(prioritized_constraints_ii(dummy_primals, dummy_parameters))
        primal_dimension_ii += slack_dimension_ii
        append!(private_primals[player_idx], slack_dimension_ii)

        # Set up symbolic variables and reformulate the preferences as penalty terms in the objective functions.
        # Define symbolic variables for primals.
        z̃ = Symbolics.scalarize(
            only(
                Symbolics.@variables(z̃[start_idx:(primal_dimension_ii + start_idx - 1)])
            ),
        )
        x = BlockArray(z̃, private_primals[player_idx])
        θ̃ = Symbolics.scalarize(
            only(Symbolics.@variables(θ̃[1:sum(parameter_dimensions)])),
        )
        θ = BlockArray(θ̃, parameter_dimensions)

        # Define symbolic variables for the slack variables.
        slacks_ii = last(z̃, slack_dimension_ii)

        # Define objective with penalty terms.
        if priority_level == 1
            append!(
                objectives_w_penalty,
                penalty_weights[player_idx][num_levels]*objectives[player_idx](x, θ),
            )
        end
        objectives_w_penalty[player_idx] +=
            penalty_weights[player_idx][priority_level]*sum(slacks_ii)

        # Append auxillary constraints into inequality constraints: fᵢ(x,θ) + sᵢ ≥ 0 , sᵢ ≥ 0
        auxillary_constraints = prioritized_constraints_ii(x, θ) .+ slacks_ii
        if priority_level == 1
            push!(
                inequality_constraints_w_preferences,
                inequality_constraints[player_idx](x, θ),
            )
        end
        append!(inequality_constraints_w_preferences[player_idx], auxillary_constraints)
        append!(inequality_constraints_w_preferences[player_idx], slacks_ii)

        # Store sum_slacks
        x_temp = let
            Symbolics.scalarize(
                only(Symbolics.@variables(z̃[1:(primal_dimension_ii + start_idx - 1)])),
            )
        end
        push!(
            sum_slacks,
            Symbolics.build_function(sum(slacks_ii), x_temp, θ, expression = Val{false}),
        )

        # Define equality constraints
        if priority_level == 1
            push!(equality_constraints_encoded, equality_constraints[player_idx](x, θ))
        end

        # Update start_idx
        if priority_level == last(ordered_priority_levels)
            start_idx += primal_dimension_ii
        end
    end

    for player in 1:num_players
        primal_dimension_ii = sum(private_primals[player])
        for priority_level in ordered_priority_levels
            set_up_penalty(priority_level, player)
        end
    end

    # Update primal_dimensions, equality_dimenions, inequality_dimensions
    primal_dimensions = map(x->sum(x), private_primals)
    equality_dimensions = map(x->length(x), equality_constraints_encoded)
    inequality_dimensions = map(x->length(x), inequality_constraints_w_preferences)

    # Set up for parametric game

    N = length(objectives)
    @assert N ==
            length(equality_constraints) ==
            length(inequality_constraints) ==
            length(primal_dimensions) ==
            length(equality_dimensions) ==
            length(inequality_dimensions)

    total_dimension =
        sum(primal_dimensions) +
        sum(equality_dimensions) +
        sum(inequality_dimensions) +
        shared_equality_dimension +
        shared_inequality_dimension

    # Define symbolic variables for this MCP.
    z̃ = Symbolics.scalarize(only(Symbolics.@variables(z̃[1:total_dimension])))
    z = BlockArray(
        Symbolics.scalarize(z̃),
        [
            sum(primal_dimensions),
            sum(equality_dimensions),
            sum(inequality_dimensions),
            shared_equality_dimension,
            shared_inequality_dimension,
        ],
    )
    x = BlockArray(z[Block(1)], primal_dimensions)
    μ = BlockArray(z[Block(2)], equality_dimensions)
    λ = BlockArray(z[Block(3)], inequality_dimensions)
    μₛ = z[Block(4)]
    λₛ = z[Block(5)]

    # Define a symbolic variable for the parameters.
    θ̃ = Symbolics.scalarize(only(Symbolics.@variables(θ̃[1:sum(parameter_dimensions)])))
    θ = BlockArray(θ̃, parameter_dimensions)

    # Retrieve trajectory_x 
    trajectory_primals = [x[Block(i)][1:private_primals[i][1]] for i in 1:num_players]
    trajectory_x = BlockArray(
        vcat(trajectory_primals...),
        [private_primals[i][1] for i in 1:num_players],
    )

    # Build symbolic expressions for objectives and constraints for all players
    # (and shared constraints).
    fs = objectives_w_penalty
    gs = equality_constraints_encoded
    hs = inequality_constraints_w_preferences
    g̃ = shared_equality_constraints(trajectory_x, θ)
    h̃ = shared_inequality_constraints(trajectory_x, θ)

    # Build Lagrangians for all players.
    Ls = map(zip(1:N, fs, gs, hs)) do (i, f, g, h)
        f - μ[Block(i)]' * g - λ[Block(i)]' * h - μₛ' * g̃ - λₛ' * h̃
    end

    # Build F = [∇ₓLs, gs, hs, g̃, h̃]'.
    ∇ₓLs = map(zip(Ls, blocks(x))) do (L, xᵢ)
        Symbolics.gradient(L, xᵢ)
    end

    F_symbolic = [reduce(vcat, ∇ₓLs); reduce(vcat, gs); reduce(vcat, hs); g̃; h̃]

    # Set lower and upper bounds for z.
    z̲ = [
        fill(-Inf, sum(primal_dimensions))
        fill(-Inf, sum(equality_dimensions))
        fill(0, sum(inequality_dimensions))
        fill(-Inf, shared_equality_dimension)
        fill(0, shared_inequality_dimension)
    ]
    z̅ = [
        fill(Inf, sum(primal_dimensions))
        fill(Inf, sum(equality_dimensions))
        fill(Inf, sum(inequality_dimensions))
        fill(Inf, shared_equality_dimension)
        fill(Inf, shared_inequality_dimension)
    ]

    # Build parametric MCP.
    # parametric_mcp = ParametricMCP(F, z̲, z̅, parameter_dimension)
    parametric_mcp = ParametricMCP(
        F_symbolic,
        Symbolics.scalarize(z̃),
        θ̃,
        z̲,
        z̅;
        compute_sensitivities = true,
    )

    ParametricGamePenalty(
        objectives_w_penalty,
        equality_constraints_encoded,
        inequality_constraints_w_preferences,
        shared_equality_constraints,
        shared_inequality_constraints,
        parameter_dimensions,
        primal_dimensions,
        equality_dimensions,
        inequality_dimensions,
        shared_equality_dimension,
        shared_inequality_dimension,
        parametric_mcp,
        sum_slacks,
    )
end

function total_dim(problem::ParametricGamePenalty)
    sum(problem.primal_dimensions) +
    sum(problem.equality_dimensions) +
    sum(problem.inequality_dimensions) +
    problem.shared_equality_dimension +
    problem.shared_inequality_dimension
end

"Solve a constrained parametric game."
function solve_penalty(
    problem::ParametricGamePenalty,
    parameter_value = zeros(problem.parameter_dimension);
    initial_guess = nothing,
    verbose = false,
    return_primals = true,
)
    z0 = if !isnothing(initial_guess)
        initial_guess
    else
        zeros(total_dim(problem))
    end

    z, status, info = ParametricMCPs.solve(
        problem.parametric_mcp,
        parameter_value;
        initial_guess = z0,
        verbose,
        cumulative_iteration_limit = 400000,
        proximal_perturbation = 1e-2,
        major_iteration_limit = 5000,
        minor_iteration_limit = 10000,
        convergence_tolerance = 2e-2, #1e-6
        nms_initial_reference_factor = 45000, #20
        nms_maximum_watchdogs = 8000, #5
        nms_memory_size = 16000, #10
        nms_mstep_frequency = 3000, #10
        lemke_start_type = "advanced",
        restart_limit = 100,
        gradient_step_limit = 100,
        use_basics = true,
        use_start = true,
    )

    solved = string(status) == "MCP_Solved"

    # Compute slacks for each preference
    slacks = vcat(map(sum_slacks -> sum_slacks(z, parameter_value), problem.sum_slacks))

    if return_primals
        primals = blocks(
            BlockArray(z[1:sum(problem.primal_dimensions)], problem.primal_dimensions),
        )
        return (; primals, variables = z, slacks, solved, info)
    else
        return (; variables = z, slacks, solved, info)
    end
end
