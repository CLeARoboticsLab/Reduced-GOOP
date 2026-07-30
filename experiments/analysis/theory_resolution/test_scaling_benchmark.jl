using Test

include(joinpath(@__DIR__, "ScalingBenchmark.jl"))
using .ScalingBenchmark

@testset "Controlled semantic-transport scaling benchmark" begin
    summary = validate_scaling_benchmark()
    @test summary["result_rows"] == 48
    @test summary["rank_rows"] == 12
    @test summary["constant_rows"] == 2
    @test summary["comparison_rows"] == 24
    @test summary["maximum_reference_residual_norm2"] <= 1e-8
    @test summary["maximum_semantic_bound_violation"] <= 1e-10
    @test summary["minimum_fixed_index_interior_limit_norm2"] > 1e-3
    @test summary["maximum_formulation_difference"] <= 1e-10
end
