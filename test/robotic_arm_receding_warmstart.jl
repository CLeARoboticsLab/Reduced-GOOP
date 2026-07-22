using Test

include(joinpath(@__DIR__, "..", "experiments", "Robotic_arm_receding.jl"))
const RAR = Robotic_arm_receding

symbolic_block(name, count) = [Symbol("$(name)[$(i)]") for i in 1:count]

@testset "Robotic-arm receding-horizon dual warm starts" begin
    # Two toy players with K¹ = 4, K² = 2 and primal dimensions 2 and 1.
    # ψ_i_k is laid out as one primal-sized segment per target level k+1:Kⁱ.
    z_symbolic = Symbol[]
    append!(z_symbolic, symbolic_block("x", 3))
    append!(z_symbolic, symbolic_block("s", 1))
    append!(z_symbolic, symbolic_block("σ", 1))
    for level in 1:4
        append!(z_symbolic, symbolic_block("λ_1_$(level)", 1))
    end
    for level in 1:2
        append!(z_symbolic, symbolic_block("λ_2_$(level)", 2))
    end
    append!(z_symbolic, symbolic_block("ψ_1_3", 2))
    append!(z_symbolic, symbolic_block("ψ_1_2", 4))
    append!(z_symbolic, symbolic_block("ψ_1_1", 6))
    append!(z_symbolic, symbolic_block("ψ_2_1", 1))
    append!(z_symbolic, symbolic_block("γ_1_1", 2))

    groups = RAR._symbolic_variable_groups(z_symbolic)
    kkt = (;
        z_symbolic,
        primal_dims = 1:3,
        preference_slack_dims = [4],
        interior_point_slack_dims = [5],
        inequality_constraint_dual_dims = groups["γ_1_1"],
        variable_dimension = length(z_symbolic),
    )
    problem = (;
        num_players = 2,
        preferences = [fill(identity, 4), fill(identity, 2)],
        primal_dims = [2, 1],
        equality_dims = [1, 2],
    )

    layout = RAR.build_dual_warmstart_layout(kkt, problem)
    @test length(layout.equality_dual_indices) == 8
    @test length(layout.all_dual_indices) == 23
    @test length(layout.innermost_stationarity_dual_indices) == 7

    expected_innermost = sort!(vcat(
        groups["ψ_1_3"],
        groups["ψ_1_2"][(end-1):end],
        groups["ψ_1_1"][(end-1):end],
        groups["ψ_2_1"],
    ))
    @test layout.innermost_stationarity_dual_indices == expected_innermost

    shifted_primal = [10.0, 20.0, 30.0]
    previous_z = collect(1001.0:(1000.0+kkt.variable_dimension))

    primal_only = RAR.build_receding_warmstart(
        shifted_primal,
        previous_z,
        kkt,
        layout,
        :primal_only,
    )
    @test primal_only == shifted_primal

    equality_only = RAR.build_receding_warmstart(
        shifted_primal,
        previous_z,
        kkt,
        layout,
        :equality_duals,
    )
    @test equality_only[kkt.primal_dims] == shifted_primal
    @test equality_only[layout.equality_dual_indices] ==
          previous_z[layout.equality_dual_indices]
    @test all(iszero, equality_only[setdiff(
        layout.innermost_stationarity_dual_indices,
        layout.equality_dual_indices,
    )])
    @test all(==(1.0), equality_only[kkt.preference_slack_dims])
    @test all(==(1.0), equality_only[kkt.interior_point_slack_dims])
    @test all(==(1.0), equality_only[kkt.inequality_constraint_dual_dims])

    without_innermost = RAR.build_receding_warmstart(
        shifted_primal,
        previous_z,
        kkt,
        layout,
        :all_except_innermost_stationarity,
    )
    selected = RAR.selected_dual_indices(layout, :all_except_innermost_stationarity)
    @test without_innermost[selected] == previous_z[selected]
    @test all(iszero, without_innermost[layout.innermost_stationarity_dual_indices])

end
