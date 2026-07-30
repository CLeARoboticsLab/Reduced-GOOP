using Test
using Random

using BlockArrays: Block, BlockArray
using ReducedGOOP

@testset "Selective KKT warm starts" begin
    equality_constraint_dual_dims = collect(6:8)
    stationarity_dual_dims = collect(9:14)
    innermost_stationarity_dual_dims = [10, 13]
    kkt = (;
        primal_dims = 1:3,
        preference_slack_dims = [4],
        interior_point_slack_dims = [5],
        inequality_constraint_dual_dims = [15, 16],
        equality_constraint_dual_dims,
        stationarity_dual_dims,
        all_equality_stationarity_dual_dims =
            vcat(equality_constraint_dual_dims, stationarity_dual_dims),
        innermost_stationarity_dual_dims,
        variable_dimension = 18,
    )

    blocks = ReducedGOOP.kkt_variable_blocks(kkt)
    @test blocks.z == [1, 2, 3]
    @test blocks.λ == [6, 7, 8]
    @test blocks.ψ_out == [9, 11, 12, 14]
    @test blocks.ψ_in == [10, 13]
    @test isempty(intersect(blocks.ψ_out, blocks.ψ_in))

    # Block lists are safe for caller-side manipulation.
    blocks.z[1] = 18
    fresh_blocks = ReducedGOOP.kkt_variable_blocks(kkt)
    @test fresh_blocks.z == [1, 2, 3]
    @test kkt.primal_dims == 1:3

    shifted_primal = [10.0, 20.0, 30.0]
    previous_z = collect(1001.0:1018.0)
    shifted_snapshot = copy(shifted_primal)
    source_snapshot = copy(previous_z)

    retained_by_mode = Dict(
        :primal_only => Int[],
        :equality_duals => fresh_blocks.λ,
        :all_except_innermost_stationarity =>
            vcat(fresh_blocks.λ, fresh_blocks.ψ_out),
        :all_duals =>
            vcat(fresh_blocks.λ, fresh_blocks.ψ_out, fresh_blocks.ψ_in),
    )
    reference_starts = Dict{Symbol,Vector{Float64}}()
    for mode in ReducedGOOP.SELECTIVE_WARMSTART_MODES
        point = ReducedGOOP.build_selective_warmstart(
            shifted_primal,
            previous_z,
            kkt,
            mode,
        )
        reference_starts[mode] = point

        @test length(point) == kkt.variable_dimension
        @test point[fresh_blocks.z] == shifted_primal
        @test point[retained_by_mode[mode]] == previous_z[retained_by_mode[mode]]

        reset_duals = setdiff(
            vcat(fresh_blocks.λ, fresh_blocks.ψ_out, fresh_blocks.ψ_in),
            retained_by_mode[mode],
        )
        @test all(iszero, point[reset_duals])
        @test point[kkt.preference_slack_dims] == [1.0]
        @test point[kkt.interior_point_slack_dims] == [1.0]
        @test point[kkt.inequality_constraint_dual_dims] == [1.0, 1.0]
        @test all(iszero, point[[17, 18]])
    end

    @test shifted_primal == shifted_snapshot
    @test previous_z == source_snapshot
    @test reference_starts[:all_duals][fresh_blocks.ψ_in] ==
          previous_z[fresh_blocks.ψ_in]

    # Every mode gets independent storage, so solving or perturbing one point
    # cannot change a later mode's initialization.
    equality_before = copy(reference_starts[:equality_duals])
    reference_starts[:primal_only][1] = -999.0
    @test reference_starts[:equality_duals] == equality_before
    @test shifted_primal == shifted_snapshot
    @test previous_z == source_snapshot

    # Construction is deterministic under randomized execution order.
    canonical = Dict(
        mode => ReducedGOOP.build_selective_warmstart(
            shifted_primal,
            previous_z,
            kkt,
            mode,
        ) for mode in ReducedGOOP.SELECTIVE_WARMSTART_MODES
    )
    for seed in (2, 11, 97)
        order =
            shuffle(MersenneTwister(seed), collect(ReducedGOOP.SELECTIVE_WARMSTART_MODES))
        randomized = Dict(
            mode => ReducedGOOP.build_selective_warmstart(
                shifted_primal,
                previous_z,
                kkt;
                mode,
            ) for mode in order
        )
        @test randomized == canonical
    end

    @test_throws ArgumentError ReducedGOOP.build_selective_warmstart(
        shifted_primal,
        previous_z,
        kkt,
        :unknown,
    )
    @test_throws DimensionMismatch ReducedGOOP.build_selective_warmstart(
        shifted_primal[1:2],
        previous_z,
        kkt,
        :primal_only,
    )
    @test_throws DimensionMismatch ReducedGOOP.build_selective_warmstart(
        shifted_primal,
        previous_z[1:end-1],
        kkt,
        :primal_only,
    )
end

@testset "One-stage receding trajectory shift" begin
    strategies = [
        (;
            xs = [[1.0], [2.0], [3.0]],
            us = [[10.0], [20.0], [30.0]],
        ),
    ]
    dynamics = [
        (;
            state_dimension = 1,
            control_dimension = 1,
            step = (x, u, t) -> x .+ t .* u,
        ),
    ]
    source_snapshot = deepcopy(strategies)

    shifted = ReducedGOOP.shift_receding_trajectories(strategies, dynamics, 3)
    @test length(shifted) == 1
    @test length(shifted[1].xs) == 3
    @test length(shifted[1].us) == 3
    @test shifted[1].xs == [[2.0], [3.0], [63.0]]
    @test shifted[1].us == [[20.0], [30.0], [0.0]]
    @test shifted[1].xs[end] ==
          dynamics[1].step(shifted[1].xs[end-1], shifted[1].us[end-1], 2)
    @test strategies == source_snapshot

    shifted[1].xs[1][1] = -10.0
    shifted[1].us[1][1] = -20.0
    @test strategies == source_snapshot

    @test_throws ArgumentError ReducedGOOP.shift_receding_trajectories(
        strategies,
        dynamics,
        1,
    )
    @test_throws DimensionMismatch ReducedGOOP.shift_receding_trajectories(
        strategies,
        Any[],
        3,
    )
end

@testset "Reduced and quasi block metadata" begin
    x_template = BlockArray(zeros(2), [2])
    θ_template = BlockArray(zeros(1), [1])
    preferences = [Function[
        (x, θ) -> sum(abs2, x[Block(1)]),
        (x, θ) -> sum(abs2, x[Block(1)] .- 1.0),
        (x, θ) -> sum(abs2, x[Block(1)] .+ 0.5),
    ]]
    problem = ReducedGOOP.ParametricGOOP(
        x_template,
        θ_template;
        preferences,
        is_prioritized_constraint = [[false, false, false]],
        equality_constraints = [
            (x, θ) -> [x[Block(1)][1] - θ[Block(1)][1]],
        ],
        inequality_constraints = [nothing],
        shared_equality_constraint =
            (x, θ) -> [x[Block(1)][2] + θ[Block(1)][1]],
        shared_inequality_constraint = nothing,
    )

    partitions = NamedTuple[]
    for generator in (
        ReducedGOOP.generate_slacked_reduced_kkt_system,
        ReducedGOOP.generate_slacked_quasi_kkt_system,
    )
        kkt = generator(problem)
        blocks = ReducedGOOP.kkt_variable_blocks(kkt)
        push!(partitions, blocks)

        @test kkt.variable_dimension == 14
        @test blocks.z == collect(1:2)
        @test length(blocks.λ) == 6
        @test length(blocks.ψ_out) == 2
        @test length(blocks.ψ_in) == 4
        @test last(blocks.λ) == kkt.variable_dimension # packed shared λₛ
        @test sort(vcat(blocks.z, blocks.λ, blocks.ψ_out, blocks.ψ_in)) ==
              collect(1:kkt.variable_dimension)

        shifted_primal = [-1.0, 2.0]
        source = collect(101.0:(100.0+kkt.variable_dimension))
        all_duals = ReducedGOOP.build_selective_warmstart(
            shifted_primal,
            source,
            kkt,
            :all_duals,
        )
        @test all_duals[blocks.z] == shifted_primal
        @test all_duals[vcat(blocks.λ, blocks.ψ_out, blocks.ψ_in)] ==
              source[vcat(blocks.λ, blocks.ψ_out, blocks.ψ_in)]
    end

    @test partitions[1] == partitions[2]
end
