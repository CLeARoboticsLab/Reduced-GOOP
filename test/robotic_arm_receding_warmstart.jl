using Test

include(joinpath(@__DIR__, "..", "experiments", "Robotic_arm_receding.jl"))
const RAR = Robotic_arm_receding

@testset "Robotic-arm receding-horizon dual warm starts" begin
    # Two toy players with K¹ = 4, K² = 2 and primal dimensions 2 and 1.
    # The stationarity block contains ψ_1_3, ψ_1_2, ψ_1_1, and ψ_2_1,
    # whose final primal-sized segments target each player's innermost level.
    equality_constraint_dual_dims = collect(6:13)
    stationarity_dual_dims = collect(14:26)
    all_equality_stationarity_dual_dims =
        vcat(equality_constraint_dual_dims, stationarity_dual_dims)
    innermost_stationarity_dual_dims = [14, 15, 18, 19, 24, 25, 26]
    inequality_constraint_dual_dims = collect(27:28)
    kkt = (;
        primal_dims = 1:3,
        preference_slack_dims = [4],
        interior_point_slack_dims = [5],
        inequality_constraint_dual_dims,
        equality_constraint_dual_dims,
        stationarity_dual_dims,
        all_equality_stationarity_dual_dims,
        innermost_stationarity_dual_dims,
        variable_dimension = 28,
    )

    @test length(kkt.equality_constraint_dual_dims) == 8
    @test length(kkt.stationarity_dual_dims) == 13
    @test kkt.all_equality_stationarity_dual_dims == collect(6:26)
    @test length(kkt.innermost_stationarity_dual_dims) == 7
    @test isempty(intersect(
        kkt.all_equality_stationarity_dual_dims,
        kkt.inequality_constraint_dual_dims,
    ))

    shifted_primal = [10.0, 20.0, 30.0]
    previous_z = collect(1001.0:(1000.0+kkt.variable_dimension))

    primal_only = RAR.build_receding_warmstart(
        shifted_primal,
        previous_z,
        kkt,
        :primal_only,
    )
    @test primal_only == shifted_primal

    equality_only = RAR.build_receding_warmstart(
        shifted_primal,
        previous_z,
        kkt,
        :equality_duals,
    )
    @test equality_only[kkt.primal_dims] == shifted_primal
    @test equality_only[kkt.equality_constraint_dual_dims] ==
          previous_z[kkt.equality_constraint_dual_dims]
    @test all(iszero, equality_only[kkt.stationarity_dual_dims])
    @test all(==(1.0), equality_only[kkt.preference_slack_dims])
    @test all(==(1.0), equality_only[kkt.interior_point_slack_dims])
    @test all(==(1.0), equality_only[kkt.inequality_constraint_dual_dims])

    without_innermost = RAR.build_receding_warmstart(
        shifted_primal,
        previous_z,
        kkt,
        :all_except_innermost_stationarity,
    )
    carried_dims = setdiff(
        kkt.all_equality_stationarity_dual_dims,
        kkt.innermost_stationarity_dual_dims,
    )
    @test without_innermost[carried_dims] == previous_z[carried_dims]
    @test all(iszero, without_innermost[kkt.innermost_stationarity_dual_dims])
    @test all(==(1.0), without_innermost[kkt.inequality_constraint_dual_dims])
end
