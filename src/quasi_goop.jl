mutable struct Lagrangian_term{T<:Symbolics.Num}
    expr::Union{Vector{T}, T}
    duals::Union{Vector{T}, Nothing}
    deriv_order_x::Int
    deriv_order_s::Int
end

function Symbolics.gradient(f::Lagrangian_term, x::AbstractVector{<:Symbolics.Num}; var_tag::Symbol = :x)
    if f.deriv_order_x > 2 || f.deriv_order_s > 2
        zero.(x)
    end
    if var_tag == :x
        f.deriv_order_x += 1
    elseif var_tag == :s
        f.deriv_order_s += 1
    else
        error("Invalid variable tag. Use :x or :s, got $var_tag")
    end
    expr = isnothing(f.duals) ? f.expr : -f.expr' * f.duals
    Symbolics.gradient(expr, x)
end

struct ordered_preferences
    "Vector of callable preference functions"
    preferences::Vector{Vector{Function}}

    "Vector of boolean values indicating if the preference is a constraint"
    is_prioritized_constraint::Vector{Vector{Bool}}
end

struct PrimalDualSysEqn{T1,T2}
    "A callable function that computes F!(val, x, λ; θ, μ) in-place for 'val'"
    F!::T1
    "A callable function that computes ∇F!(val, x, λ; θ, μ) in-place for 'val'"
    ∇F!::T2
    #TODO dimensions
end

"""
Function that constructs a `ParametricQuasiGOOP` object from callable functins of objectives, constraints, and preferences.
"""
function ParametricQuasiGOOP(;
    objectives,
    equality_constraints,
    inequality_constraints,
    preferences::ordered_preferences,
    shared_equality_constraints,
    shared_inequality_constraints,
    primal_dimensions,
    parameter_dimensions,
    equality_dimensions,
    inequality_dimensions,
)

    (;
        objectives_symbolic,
        equality_symbolic,
        inequality_symbolic,
        preferences_symbolic,
        shared_equality_symbolic,
        shared_inequality_symbolic,
    ) = QuasiGOOP_to_PDSyst(;
        objectives,
        equality_constraints,
        inequality_constraints,
        preferences,
        shared_equality_constraints,
        shared_inequality_constraints,
        primal_dimensions,
        parameter_dimensions,
        equality_dimensions,
        inequality_dimensions,
    )

    PrimalDualSysEqn(;
        objectives_symbolic,
        equality_symbolic,
        inequality_symbolic,
        preferences_symbolic,
        shared_equality_symbolic,
        shared_inequality_symbolic,
    )
end

"Helper function to create components of the Primal-Dual system of equations from QuasiGOOP."
function QuasiGOOP_to_PDSyst(;
    objectives::Vector{<:Function},
    equality_constraints::Vector{<:Function},
    inequality_constraints::Vector{<:Function},
    preferences::ordered_preferences,
    shared_equality_constraints,
    shared_inequality_constraints,
    primal_dimensions::Vector{Int},
    parameter_dimensions::Vector{Int},
    equality_dimensions::Vector{Int},
    inequality_dimensions::Vector{Int},
)
    println("Make a PrimalDualSysEqn object from callable functions + implement approximations")
    backend = SymbolicTracingUtils.SymbolicsBackend()

    # Problem data
    ordered_priority_levels = eachindex(preferences.preferences[1]) # assume all players have the same number of preferences
    num_players = length(preferences.preferences)

    dual_dimension = 0
    primal_dimension_ii = 0
    equality_dimension_ii = copy(equality_dimensions)
    inequality_dimension_ii = copy(inequality_dimensions)

    # Account for annealing parameter, σ → 0
    augmented_parameter_dimension = sum(parameter_dimensions) + 1 

    dummy_primals = BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
    dummy_parameters = BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

    # Preallocate arrays of symbolic functions for each player
    private_inner_equality_constraints = Vector{Symbolics.Num}[]
    Lagrangian_terms = Lagrangian_term[] 

    # Store dimensions
    private_primals = [[dim] for dim in primal_dimensions]

    # Store (callable) slacks
    private_slacks = Function[]

    start_idx = 1
    
    function set_up_level(priority_level, player_idx)
        # Main.@infiltrate
        # Step 1. Reformulation for prioritized constraints
        if preferences.is_prioritized_constraint[player_idx][priority_level]
            prioritized_constraints_ii =
                preferences.preferences[player_idx][priority_level] # fᵢ(x,θ) ≥ 0
            preference_slack_dimension_ii =
                length(prioritized_constraints_ii(dummy_primals, dummy_parameters))
            primal_dimension_ii += preference_slack_dimension_ii
            append!(private_primals[player_idx], preference_slack_dimension_ii)
            inequality_dimension_ii[player_idx] += 2preference_slack_dimension_ii # account for sᵢ ≥ 0
        end
        # Main.@infiltrate

        # Step 2. Reformulate inequality constraints as equality constraints (via additional slacks)
        equality_dimension_ii[player_idx] += inequality_dimension_ii[player_idx]
        barrier_slacks_dimension_ii = copy(inequality_dimension_ii[player_idx])
        primal_dimension_ii += barrier_slacks_dimension_ii
        append!(private_primals[player_idx], barrier_slacks_dimension_ii)
        @assert sum(private_primals[player_idx]) == primal_dimension_ii

        # Main.@infiltrate

        # Step 2. Define symbolic variables for primals
        total_dimension =
            primal_dimension_ii +
            equality_dimension_ii[player_idx]
        
        z̃ = Symbolics.scalarize(
                only(Symbolics.@variables(z̃[start_idx:(total_dimension + start_idx - 1)])),
            )        
        z = BlockArray(
            z̃,
            [
                primal_dimension_ii,
                equality_dimension_ii[player_idx],
            ],
        )
        θ̃ = SymbolicTracingUtils.make_variables(backend, :θ̃, augmented_parameter_dimension)
        θ = BlockArray(
            θ̃,
            vcat(
                parameter_dimensions,
                [1],
            ),
        )
        μ = θ[end] # annealing parameter

        x = BlockArray(z[Block(1)], private_primals[player_idx])
        λ = z[Block(2)] # dual variables for equality constraints
        # Main.@infiltrate

        # Step 3. Define symbolic expression for objective and (equality) constraints
        if priority_level == first(ordered_priority_levels) ||
           isempty(private_inner_equality_constraints)
            push!(
                private_inner_equality_constraints,
                equality_constraints[player_idx](x, θ)
            )
        end

        # Main.@infiltrate

        preference_slacks_ii = x[Block(2)] # These slacks are introduced before reformulation slacks
        barrier_slacks_ii = BlockArray(x[Block(3)], [2preference_slack_dimension_ii, inequality_dimensions[player_idx]])
        objective_ii = sum(preference_slacks_ii)
        barrier_objective_ii = μ * sum(log.(barrier_slacks_ii))

        # Main.@infiltrate
        # Replace inequality constriants: fg(x,sₚ) = sₚ + g(x) ≥ 0, fg(x,sₚ) - sᵦ = 0
        auxillary_constraints = prioritized_constraints_ii(x, θ) .+ preference_slacks_ii
        append!(
            private_inner_equality_constraints[player_idx],
            vcat(
                auxillary_constraints,
                preference_slacks_ii) - barrier_slacks_ii[Block(1)] 
        )
        
        # Note: We need to replace inequality constraints with equality constraints at every priority level
        # Fact: inequality constraints exist at every priority level due to reformulation of preferences

        if preferences.is_prioritized_constraint[player_idx][priority_level]
            append!(
                private_inner_equality_constraints[player_idx], # f(x) = 0
                inequality_constraints[player_idx](x, θ) - barrier_slacks_ii[Block(2)] # fg(x,s) = g(x) - s = 0
            ) 
        end

        # store private_slacks
        x_temp = let
            Symbolics.scalarize(
                only(Symbolics.@variables(z̃[1:(total_dimension + start_idx + 1)])),
            ) 
        end
        sum_slacks = Symbolics.build_function(
            sum(preference_slacks_ii),
            x_temp,
            θ,
            expression = Val{false},
        )
        push!(private_slacks, sum_slacks)

        # Main.@infiltrate
        # Step 4. Define symbolic expression for Lagrangian and stationarity conditions/constraints
        dims = [equality_dimensions[player_idx], 2preference_slack_dimension_ii, inequality_dimensions[player_idx]]
        @assert sum(dims) == length(private_inner_equality_constraints[player_idx])
        λ = BlockArray(λ, dims)
        push!(
            Lagrangian_terms,
            Lagrangian_term(
                objective_ii - barrier_objective_ii, nothing, 0, 0)
        )
        push!(
            Lagrangian_terms,
            Lagrangian_term(equality_constraints[player_idx](x, θ), λ[Block(1)], 0, 0)
        )
        push!(
            Lagrangian_terms,
            Lagrangian_term(
                vcat(
                auxillary_constraints,
                preference_slacks_ii) - barrier_slacks_ii[Block(1)], λ[Block(2)], 0, 0)
        )
        push!(
            Lagrangian_terms,
            Lagrangian_term(inequality_constraints[player_idx](x, θ) - barrier_slacks_ii[Block(2)], λ[Block(3)], 0, 0)
        )
        # Lagrangian + drop higher-order (≥3) terms
        x_primal, x_slack = vcat(x[Block(1)], x[Block(2)]), x[Block(3)]
        stationarity_x = zero.(x_primal)
        stationarity_s = zero.(x_slack)
        for term in Lagrangian_terms
            stationarity_x .+= Symbolics.gradient(term, x_primal; var_tag = :x)
            stationarity_s .+= Symbolics.gradient(term, x_slack; var_tag = :s)
        end
        Main.@infiltrate
        g_ii = private_inner_equality_constraints[player_idx]
        L = objective_ii - barrier_objective_ii - λ' * g_ii
        stationarity_check_x = Symbolics.gradient(L, x_primal)
        stationarity_check_s = Symbolics.gradient(L, x_slack)
        println("isequal(stationarity_x, stationarity_check_x) = ", isequal(stationarity_x, stationarity_check_x)) # true
        println("isequal(stationarity_s, stationarity_check_s) = ", isequal(stationarity_s, stationarity_check_s)) # true
    end

    # Build KKT system for each priority level for each player's own problem
    for player in 1:num_players
        primal_dimension_ii = sum(private_primals[player])
        for priority_level in ordered_priority_levels
            if !isnothing(preferences.is_prioritized_constraint[player][priority_level])
                set_up_level(priority_level, player)
            end
        end
    end

end

"Helper function to create a 'PrimalDualSysEqn' object from symbolic functions."
function PrimalDualSysEqn(;
    objectives::Vector{<:Symbolics.Num},
    equality_constraints::Vector{<:Symbolics.Num},
    inequality_constraints::Vector{<:Symbolics.Num},
    preferences::ordered_preferences,
    shared_equality_constraints,
    shared_inequality_constraints,
    primal_dimensions::Vector{Int},
    parameter_dimensions::Vector{Int},
    equality_dimensions::Vector{Int},
    inequality_dimensions::Vector{Int},
)
    println("Make a PrimalDualSysEqn object from symbolic functions")
    #TODO

    F! = nothing
    ∇F! = nothing
    PrimaldualSysEqn(F!, ∇F!)
end

# struct ParametricQuasiGOOP{T1, T2, T3, T4, T5, T6, T7, T8}
#     "Objective functions for all players"
#     objectives::T1
#     "Equality constraints for all players"
#     private_inner_equality_constraints::T2
#     "Inequality constraints for all players"
#     private_inner_inequality_constraints::T3
#     "Shared equality constraint"
#     shared_equality_constraints::T4
#     "Shared inequality constraint"
#     shared_inequality_constraints::T5

#     "Dimension of parameter vector"
#     parameter_dimensions::T6
#     "Dimension of primal variables for all players"
#     primal_dimensions::T7
#     "Dimension of equality constraints for all players"
#     equality_dimensions::T7
#     "Dimension of inequality constraints for all players"
#     inequality_dimensions::T7
#     "Dimension of shared equality constraint"
#     shared_equality_dimension::T8
#     "Dimension of shared inequality constraint"
#     shared_inequality_dimension::T8

#     "Corresponding Primal Dual System Representation."
#     pd_system::PrimalDualSysEqn 
# end

# "Helper function to create a 'PrimalDualSysEqn' object from callable functions."
# function PrimalDualSysEqn(;
#     objectives::Vector{<:Function},
#     equality_constraints::Vector{<:Function},
#     inequality_constraints::Vector{<:Function},
#     preferences::ordered_preferences,
#     shared_equality_constraints::Vector{<:Function},
#     shared_inequality_constraints::Vector{<:Function},
#     primal_dimensions::Vector{Int},
#     parameter_dimensions::Vector{Int},
#     equality_dimensions::Vector{Int},
#     inequality_dimensions::Vector{Int},
#     )
#     println("Make a PrimalDualSysEqn object from callable functions")
#     #TODO

# end
