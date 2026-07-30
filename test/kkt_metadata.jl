using Test

using BlockArrays: Block, BlockArray
using ReducedGOOP

function _sentinel_semantic_problem(; annotated = true)
    horizon = 4
    state_dimension = 1
    control_dimension = 1
    primal_dimension = horizon * (state_dimension + control_dimension)

    primal_specs = ReducedGOOP.PrimalCoordinateSpec[]
    for stage in 1:horizon
        push!(
            primal_specs,
            ReducedGOOP.PrimalCoordinateSpec(;
                player = 1,
                variable = :state,
                stage,
                component = 1,
                shift_rule = :successor,
                tail_role = stage == horizon ? :dynamic_completion : :shifted,
            ),
        )
        push!(
            primal_specs,
            ReducedGOOP.PrimalCoordinateSpec(;
                player = 1,
                variable = :control,
                stage,
                component = 1,
                shift_rule = :successor,
                tail_role = stage == horizon ? :zero_completion : :shifted,
            ),
        )
    end

    equality_specs = ReducedGOOP.EqualityCoordinateSpec[
        ReducedGOOP.EqualityCoordinateSpec(;
            scope = :player,
            player = 1,
            equation_class = :initial_condition,
            equation_type = :initial_state,
            stage = 1,
            component = 1,
            shift_rule = :reset,
            tail_role = :initial_condition,
        ),
        ReducedGOOP.EqualityCoordinateSpec(;
            scope = :player,
            player = 1,
            equation_class = :initial_condition,
            equation_type = :initial_control,
            stage = 1,
            component = 1,
            shift_rule = :reset,
            tail_role = :initial_condition,
        ),
    ]
    for stage in 1:(horizon-1)
        push!(
            equality_specs,
            ReducedGOOP.EqualityCoordinateSpec(;
                scope = :player,
                player = 1,
                equation_class = :dynamics,
                equation_type = :dynamics,
                stage,
                successor_stage = stage + 1,
                component = 1,
                shift_rule = :successor,
                tail_role = stage == horizon - 1 ? :terminal_dynamics : :shifted,
            ),
        )
    end
    for stage in 1:horizon
        push!(
            equality_specs,
            ReducedGOOP.EqualityCoordinateSpec(;
                scope = :player,
                player = 1,
                equation_class = :time_indexed_path_equality,
                equation_type = :path,
                stage,
                component = 1,
                shift_rule = :successor,
                tail_role = stage == horizon ? :terminal_path : :shifted,
            ),
        )
    end

    layout = annotated ?
             ReducedGOOP.GOOPSemanticLayout(;
        primal_by_player = [primal_specs],
        equality_by_player = [equality_specs],
    ) : nothing
    x_template = BlockArray(zeros(primal_dimension), [primal_dimension])
    θ_template = BlockArray(zeros(2), [2])

    function equality(x, θ)
        trajectory = reshape(x[Block(1)], state_dimension + control_dimension, :)
        states = @view trajectory[1, :]
        controls = @view trajectory[2, :]
        vcat(
            states[1] - θ[Block(1)][1],
            controls[1] - θ[Block(1)][2],
            [
                states[stage+1] - states[stage] - controls[stage] for
                stage in 1:(horizon-1)
            ],
            [states[stage] + controls[stage] for stage in 1:horizon],
        )
    end

    preferences = [Function[
        (x, θ) -> sum(abs2, x[Block(1)]),
        (x, θ) -> sum(abs2, x[Block(1)] .- θ[Block(1)][1]),
        (x, θ) -> sum(abs2, x[Block(1)] .+ θ[Block(1)][2]),
    ]]
    problem = ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences,
        is_prioritized_constraint = [[false, false, false]],
        equality_constraints = [equality],
        inequality_constraints = [nothing],
        shared_equality_constraint = nothing,
        shared_inequality_constraint = nothing,
        semantic_layout = layout,
    )
    (; problem, horizon, primal_specs, equality_specs)
end

function _semantic_transport_lookup(records)
    Dict(
        (
            record.family,
            record.scope,
            record.player,
            record.owner_level,
            record.target_level,
            record.equation_class,
            record.equation_type,
            record.primal_variable,
            record.stage,
            record.component,
        ) => record for record in records
    )
end

function _two_player_shared_semantic_problem()
    horizon = 4
    state_dimensions = [1, 2]
    control_dimensions = [1, 1]
    primal_dimensions = [
        horizon * (state_dimensions[player] + control_dimensions[player]) for
        player in 1:2
    ]
    parameter_dimensions = copy(state_dimensions)

    primal_by_player = map(1:2) do player
        specs = ReducedGOOP.PrimalCoordinateSpec[]
        for stage in 1:horizon
            for component in 1:state_dimensions[player]
                push!(
                    specs,
                    ReducedGOOP.PrimalCoordinateSpec(;
                        player,
                        variable = :state,
                        stage,
                        component,
                        shift_rule = :successor,
                        tail_role =
                            stage == horizon ? :dynamic_completion : :shifted,
                    ),
                )
            end
            for component in 1:control_dimensions[player]
                push!(
                    specs,
                    ReducedGOOP.PrimalCoordinateSpec(;
                        player,
                        variable = :control,
                        stage,
                        component,
                        shift_rule = :successor,
                        tail_role =
                            stage == horizon ? :zero_completion : :shifted,
                    ),
                )
            end
        end
        specs
    end
    equality_by_player = map(1:2) do player
        specs = ReducedGOOP.EqualityCoordinateSpec[]
        for component in 1:state_dimensions[player]
            push!(
                specs,
                ReducedGOOP.EqualityCoordinateSpec(;
                    scope = :player,
                    player,
                    equation_class = :initial_condition,
                    equation_type = :initial_state,
                    stage = 1,
                    component,
                    shift_rule = :reset,
                    tail_role = :initial_condition,
                ),
            )
        end
        for stage in 1:(horizon-1), component in 1:state_dimensions[player]
            push!(
                specs,
                ReducedGOOP.EqualityCoordinateSpec(;
                    scope = :player,
                    player,
                    equation_class = :dynamics,
                    equation_type = :dynamics,
                    stage,
                    successor_stage = stage + 1,
                    component,
                    shift_rule = :successor,
                    tail_role =
                        stage == horizon - 1 ? :terminal_dynamics : :shifted,
                ),
            )
        end
        specs
    end
    shared_equality = ReducedGOOP.EqualityCoordinateSpec[
        ReducedGOOP.EqualityCoordinateSpec(;
            scope = :shared,
            player = nothing,
            equation_class = :global_non_time_indexed,
            equation_type = :coupling_budget,
            stage = nothing,
            component = 1,
            shift_rule = :identity,
            tail_role = :global_identity,
        ),
    ]
    semantic_layout = ReducedGOOP.GOOPSemanticLayout(;
        primal_by_player,
        equality_by_player,
        shared_equality,
    )
    x_template =
        BlockArray(zeros(sum(primal_dimensions)), primal_dimensions)
    θ_template =
        BlockArray(zeros(sum(parameter_dimensions)), parameter_dimensions)

    equality_constraints = map(1:2) do player
        function (x, θ)
            nx = state_dimensions[player]
            nu = control_dimensions[player]
            trajectory = reshape(x[Block(player)], nx + nu, :)
            states = @view trajectory[1:nx, :]
            controls = @view trajectory[(nx+1):end, :]
            dynamics = mapreduce(vcat, 1:(horizon-1)) do stage
                states[:, stage+1] .- states[:, stage] .- controls[1, stage]
            end
            vcat(states[:, 1] .- θ[Block(player)], dynamics)
        end
    end
    preferences = [
        Function[
            (x, θ) -> sum(abs2, x[Block(1)]),
            (x, θ) -> sum(abs2, x[Block(1)] .- 1),
            (x, θ) -> sum(abs2, x[Block(1)] .+ 2),
        ],
        Function[
            (x, θ) -> sum(abs2, x[Block(2)]),
            (x, θ) -> sum(abs2, x[Block(2)] .- 3),
        ],
    ]
    problem = ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences,
        is_prioritized_constraint = [[false, false, false], [false, false]],
        equality_constraints,
        inequality_constraints = [nothing, nothing],
        shared_equality_constraint =
            (x, θ) -> [x[Block(1)][1] + x[Block(2)][1]],
        shared_inequality_constraint = nothing,
        semantic_layout,
    )
    (; problem, horizon, primal_dimensions)
end

@testset "Generated KKT semantic metadata and T=4 transport" begin
    setup = _sentinel_semantic_problem()
    generated = map(
        (
            ReducedGOOP.generate_slacked_reduced_kkt_system,
            ReducedGOOP.generate_slacked_quasi_kkt_system,
        ),
    ) do generator
        generator(setup.problem)
    end

    for kkt in generated
        variables = ReducedGOOP.kkt_variable_metadata(kkt)
        equations = ReducedGOOP.kkt_equation_metadata(kkt)
        blocks = ReducedGOOP.kkt_variable_blocks(kkt)

        @test length(blocks.z) == 8
        @test length(blocks.λ) == 27
        @test length(blocks.ψ_out) == 8
        @test length(blocks.ψ_in) == 16
        @test kkt.variable_dimension == 59
        @test kkt.kkt_dimension == 33
        @test length(equations) == kkt.kkt_dimension
        @test getfield.(equations, :row) == collect(1:kkt.kkt_dimension)

        primal = filter(record -> record.family === :primal, variables)
        @test [
            (record.primal_variable, record.stage, record.component) for
            record in primal
        ] == [
            (:state, 1, 1),
            (:control, 1, 1),
            (:state, 2, 1),
            (:control, 2, 1),
            (:state, 3, 1),
            (:control, 3, 1),
            (:state, 4, 1),
            (:control, 4, 1),
        ]

        λ = filter(record -> record.family === :equality_multiplier, variables)
        @test unique(getfield.(λ, :owner_level)) == Union{Nothing,Int}[1, 2, 3]
        @test [record.owner_level for record in λ[1:9:27]] == [1, 2, 3]
        @test [record.equation_type for record in λ[1:9]] == [
            :initial_state,
            :initial_control,
            :dynamics,
            :dynamics,
            :dynamics,
            :path,
            :path,
            :path,
            :path,
        ]
        @test getfield.(λ[1:9], :equation_class) == [
            :initial_condition,
            :initial_condition,
            :dynamics,
            :dynamics,
            :dynamics,
            :time_indexed_path_equality,
            :time_indexed_path_equality,
            :time_indexed_path_equality,
            :time_indexed_path_equality,
        ]

        ψ = filter(
            record -> record.family === :stationarity_multiplier,
            variables,
        )
        @test [
            (ψ[first(range)].owner_level, ψ[first(range)].target_level) for
            range in (1:8, 9:16, 17:24)
        ] == [(2, 3), (1, 2), (1, 3)]
        @test all(record -> record.target_level == 3, ψ[1:8])
        @test all(record -> record.target_level == 2, ψ[9:16])
        @test all(record -> record.target_level == 3, ψ[17:24])
        @test blocks.ψ_in == vcat(
            getfield.(ψ[1:8], :index),
            getfield.(ψ[17:24], :index),
        )

        @test [
            (record.family, record.level) for record in equations[1:24]
        ] == vcat(
            fill((:stationarity, 1), 8),
            fill((:stationarity, 2), 8),
            fill((:stationarity, 3), 8),
        )
        @test getfield.(equations[25:33], :equation_type) == [
            :initial_state,
            :initial_control,
            :dynamics,
            :dynamics,
            :dynamics,
            :path,
            :path,
            :path,
            :path,
        ]
        @test Set(getfield.(equations, :horizon_role)) == Set((
            :inherited_interior,
            :initial_boundary,
            :terminal_boundary,
        ))
        @test count(
            record -> record.horizon_role === :initial_boundary,
            equations,
        ) == 8
        @test count(
            record -> record.horizon_role === :inherited_interior,
            equations,
        ) == 17
        @test count(
            record -> record.horizon_role === :terminal_boundary,
            equations,
        ) == 8
        @test all(equations) do record
            record.exact_shift_invariant ==
                (record.horizon_role === :inherited_interior)
        end
        @test all(equations[1:24]) do record
            expected = record.stage == 1 ? :initial_boundary :
                       record.stage == setup.horizon ? :terminal_boundary :
                       :inherited_interior
            record.horizon_role === expected
        end
        @test all(equations[25:26]) do record
            record.horizon_role === :initial_boundary
        end
        @test all(equations[27:28]) do record
            record.horizon_role === :inherited_interior
        end
        @test equations[29].horizon_role === :terminal_boundary
        @test all(equations[30:32]) do record
            record.horizon_role === :inherited_interior
        end
        @test equations[33].horizon_role === :terminal_boundary

        dual_records = filter(
            record ->
                record.family in (:equality_multiplier, :stationarity_multiplier),
            variables,
        )
        lookup = _semantic_transport_lookup(dual_records)
        source = collect(1001.0:(1000.0+kkt.variable_dimension))
        zero_tail =
            ReducedGOOP.transport_receding_duals(source, kkt; tail = :zero)
        hold_tail =
            ReducedGOOP.transport_receding_duals(source, kkt; tail = :hold)

        for destination in dual_records
            if destination.shift_rule === :reset
                @test !destination.successor_exists
                @test iszero(zero_tail[destination.index])
                @test iszero(hold_tail[destination.index])
            else
                @test destination.shift_rule === :successor
                successor_key = (
                    destination.family,
                    destination.scope,
                    destination.player,
                    destination.owner_level,
                    destination.target_level,
                    destination.equation_class,
                    destination.equation_type,
                    destination.primal_variable,
                    destination.stage + 1,
                    destination.component,
                )
                @test destination.successor_exists == haskey(lookup, successor_key)
                if haskey(lookup, successor_key)
                    @test zero_tail[destination.index] ==
                          source[lookup[successor_key].index]
                    @test hold_tail[destination.index] ==
                          source[lookup[successor_key].index]
                else
                    @test iszero(zero_tail[destination.index])
                    @test hold_tail[destination.index] == source[destination.index]
                end
            end
        end
        terminal_control_ψ = filter(ψ) do record
            record.primal_variable === :control &&
                record.stage == setup.horizon
        end
        @test !isempty(terminal_control_ψ)
        @test all(record -> record.shift_rule === :reset, terminal_control_ψ)
        @test all(record -> !record.successor_exists, terminal_control_ψ)
        @test all(
            record -> iszero(hold_tail[record.index]),
            terminal_control_ψ,
        )
        penultimate_control_ψ = filter(ψ) do record
            record.primal_variable === :control &&
                record.stage == setup.horizon - 1
        end
        @test all(record -> record.successor_exists, penultimate_control_ψ)
        @test all(penultimate_control_ψ) do record
            source_record = lookup[(
                record.family,
                record.scope,
                record.player,
                record.owner_level,
                record.target_level,
                record.equation_class,
                record.equation_type,
                record.primal_variable,
                setup.horizon,
                record.component,
            )]
            zero_tail[record.index] == source[source_record.index]
        end
        @test source == collect(1001.0:(1000.0+kkt.variable_dimension))

        shifted_primal = collect(-8.0:-1.0)
        identity = ReducedGOOP.build_selective_warmstart(
            shifted_primal,
            source,
            kkt;
            mode = :all_duals,
            dual_transport = :identity_copy,
        )
        shifted_zero = ReducedGOOP.build_selective_warmstart(
            shifted_primal,
            source,
            kkt;
            mode = :all_duals,
            dual_transport = :stage_shift_zero_tail,
        )
        shifted_hold = ReducedGOOP.build_selective_warmstart(
            shifted_primal,
            source,
            kkt;
            mode = :all_duals,
            dual_transport = :stage_shift_hold_tail,
        )
        dual_indices = vcat(blocks.λ, blocks.ψ_out, blocks.ψ_in)
        @test identity[blocks.z] == shifted_primal
        @test identity[dual_indices] == source[dual_indices]
        @test shifted_zero[dual_indices] == zero_tail[dual_indices]
        @test shifted_hold[dual_indices] == hold_tail[dual_indices]
    end

    @test ReducedGOOP.kkt_variable_metadata(generated[1]) ==
          ReducedGOOP.kkt_variable_metadata(generated[2])
    @test ReducedGOOP.kkt_equation_metadata(generated[1]) ==
          ReducedGOOP.kkt_equation_metadata(generated[2])
end

@testset "Semantic validation and unsupported generic transport" begin
    setup = _sentinel_semantic_problem(; annotated = false)
    kkt = ReducedGOOP.generate_slacked_reduced_kkt_system(setup.problem)
    source = ones(kkt.variable_dimension)
    @test all(
        record -> record.horizon_role === :unclassified,
        ReducedGOOP.kkt_equation_metadata(kkt),
    )
    @test all(
        record -> !record.exact_shift_invariant,
        ReducedGOOP.kkt_equation_metadata(kkt),
    )
    @test_throws ArgumentError ReducedGOOP.transport_receding_duals(source, kkt)
    @test_throws ArgumentError ReducedGOOP.build_selective_warmstart(
        zeros(length(kkt.primal_dims)),
        source,
        kkt;
        mode = :all_duals,
        dual_transport = :unknown_policy,
    )

    bad_layout = ReducedGOOP.GOOPSemanticLayout(
        primal_by_player = [ReducedGOOP.PrimalCoordinateSpec[]],
        equality_by_player = [ReducedGOOP.EqualityCoordinateSpec[]],
    )
    x_template = BlockArray(zeros(1), [1])
    θ_template = BlockArray(zeros(1), [1])
    @test_throws DimensionMismatch ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences = [[(x, θ) -> sum(abs2, x[Block(1)])]],
        is_prioritized_constraint = [[false]],
        equality_constraints = [nothing],
        inequality_constraints = [nothing],
        shared_equality_constraint = nothing,
        shared_inequality_constraint = nothing,
        semantic_layout = bad_layout,
    )

    invalid_class_layout = ReducedGOOP.GOOPSemanticLayout(
        primal_by_player = [[
            ReducedGOOP.PrimalCoordinateSpec(;
                player = 1,
                component = 1,
                shift_rule = :identity,
            ),
        ]],
        equality_by_player = [[
            ReducedGOOP.EqualityCoordinateSpec(;
                scope = :player,
                player = 1,
                equation_class = :shared_time_indexed,
                equation_type = :bad_shared_scope,
                stage = 1,
                component = 1,
                shift_rule = :successor,
            ),
        ]],
    )
    @test_throws ArgumentError ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences = [[(x, θ) -> sum(abs2, x[Block(1)])]],
        is_prioritized_constraint = [[false]],
        equality_constraints = [(x, θ) -> [x[Block(1)][1]]],
        inequality_constraints = [nothing],
        shared_equality_constraint = nothing,
        shared_inequality_constraint = nothing,
        semantic_layout = invalid_class_layout,
    )
end

@testset "Two-player isolation and shared-global identity transport" begin
    setup = _two_player_shared_semantic_problem()
    generated = [
        generator(setup.problem) for generator in (
            ReducedGOOP.generate_slacked_reduced_kkt_system,
            ReducedGOOP.generate_slacked_quasi_kkt_system,
        )
    ]

    for kkt in generated
        variables = ReducedGOOP.kkt_variable_metadata(kkt)
        by_index = Dict(record.index => record for record in variables)
        map = ReducedGOOP.receding_dual_transport_map(kkt; tail = :zero)
        source = collect(2001.0:(2000.0+kkt.variable_dimension))
        transported =
            ReducedGOOP.transport_receding_duals(source, kkt; tail = :zero)

        @test setup.primal_dimensions == [8, 12]
        @test length(ReducedGOOP.kkt_variable_blocks(kkt).λ) == 32
        @test length(ReducedGOOP.kkt_variable_blocks(kkt).ψ_out) == 8
        @test length(ReducedGOOP.kkt_variable_blocks(kkt).ψ_in) == 28

        for (source_index, destination_index) in
            zip(map.source_indices, map.destination_indices)
            source_record = by_index[source_index]
            destination_record = by_index[destination_index]
            @test source_record.family === destination_record.family
            @test source_record.scope === destination_record.scope
            @test source_record.player === destination_record.player
            @test source_record.owner_level === destination_record.owner_level
            @test source_record.target_level === destination_record.target_level
            @test source_record.equation_class ===
                  destination_record.equation_class
            @test source_record.equation_type === destination_record.equation_type
            @test source_record.primal_variable ===
                  destination_record.primal_variable
            @test source_record.component == destination_record.component
            @test transported[destination_index] == source[source_index]
        end

        player_ψ = Dict(
            player => filter(variables) do record
                record.family === :stationarity_multiplier &&
                    record.player == player
            end for player in 1:2
        )
        @test length(player_ψ[1]) == 24
        @test length(player_ψ[2]) == 12
        @test Set(
            (record.owner_level, record.target_level) for record in player_ψ[1]
        ) == Set([(2, 3), (1, 2), (1, 3)])
        @test Set(
            (record.owner_level, record.target_level) for record in player_ψ[2]
        ) == Set([(1, 2)])

        shared_global = only(filter(variables) do record
            record.family === :equality_multiplier &&
                record.scope === :shared_innermost
        end)
        @test shared_global.player === nothing
        @test shared_global.owner_level === nothing
        @test shared_global.equation_class === :global_non_time_indexed
        @test shared_global.equation_type === :coupling_budget
        @test shared_global.shift_rule === :identity
        @test !shared_global.successor_exists
        @test transported[shared_global.index] == source[shared_global.index]

        shared_row = only(filter(
            record ->
                record.family === :equality_feasibility &&
                record.scope === :shared,
            ReducedGOOP.kkt_equation_metadata(kkt),
        ))
        @test shared_row.equation_class === :global_non_time_indexed
        @test shared_row.equation_type === :coupling_budget
        @test shared_row.horizon_role === :global
        @test !shared_row.exact_shift_invariant
    end

    @test ReducedGOOP.kkt_variable_metadata(generated[1]) ==
          ReducedGOOP.kkt_variable_metadata(generated[2])
    @test ReducedGOOP.kkt_equation_metadata(generated[1]) ==
          ReducedGOOP.kkt_equation_metadata(generated[2])
end
