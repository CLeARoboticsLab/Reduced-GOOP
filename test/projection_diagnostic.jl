using LinearAlgebra
using SparseArrays
using Test

include(
    joinpath(
        @__DIR__,
        "..",
        "experiments",
        "analysis",
        "dual_transport",
        "ProjectionDiagnostic.jl",
    ),
)
using .ProjectionDiagnostic

@testset "Sparse row/null projection diagnostic" begin
    @testset "exact unscaled projection" begin
        Jd = sparse([1.0 0.0 0.0; 0.0 1.0 0.0])
        d = [3.0, 4.0, 5.0]
        Jd_before = copy(Jd)
        d_before = copy(d)

        result = project_row_null(Jd, d; ordering = :fixed)

        @test result.row_component ≈ [3.0, 4.0, 0.0] atol = 1e-12
        @test result.null_component ≈ [0.0, 0.0, 5.0] atol = 1e-12
        @test result.row_component + result.null_component ≈ d atol = 1e-12
        @test result.rank == 2
        @test result.nullity == 1
        @test result.energies.euclidean.total ≈ 50.0
        @test result.energies.euclidean.row ≈ 25.0
        @test result.energies.euclidean.null ≈ 25.0
        @test result.orthogonality.euclidean_cosine ≤ 1e-12
        @test result.idempotence.euclidean_relative ≤ 1e-12
        @test result.null_action.relative_to_input ≤ 1e-12
        @test Jd == Jd_before
        @test d == d_before
    end

    @testset "relative rank threshold" begin
        Jd = sparse([1.0 0.0; 0.0 1e-12])
        result = project_row_null(
            Jd,
            [1.0, 1.0];
            rank_rtol = 1e-10,
            ordering = :fixed,
        )

        @test result.rank == 1
        @test result.nullity == 1
        @test result.thresholds.rank_threshold ≈ 1e-10
        @test result.row_component ≈ [1.0, 0.0] atol = 1e-10
        @test result.null_component ≈ [0.0, 1.0] atol = 1e-10
    end

    @testset "scale-aware metric" begin
        Jd = sparse(reshape([1.0, 1.0], 1, 2))
        d = [10.0, 1.0]
        unscaled =
            project_row_null(Jd, d; ordering = :fixed)
        scaled = project_row_null(
            Jd,
            d;
            scale_mode = :scale_aware,
            ordering = :fixed,
        )

        @test unscaled.row_component ≈ [5.5, 5.5] atol = 1e-12
        @test scaled.coordinate_scales == [10.0, 1.0]
        @test scaled.row_component ≈ [1100 / 101, 11 / 101] atol = 1e-12
        @test scaled.row_component + scaled.null_component ≈ d atol = 1e-12
        @test scaled.orthogonality.metric_cosine ≤ 1e-12
        @test scaled.idempotence.metric_relative ≤ 1e-12
        @test !isapprox(
            scaled.row_component,
            unscaled.row_component;
            atol = 1e-6,
        )

        custom = project_row_null(
            Jd,
            d;
            scale_mode = :relative,
            coordinate_scales = [2.0, 4.0],
            ordering = :fixed,
        )
        @test custom.scale_mode == :scale_aware
        @test custom.coordinate_scales == [2.0, 4.0]
    end

    @testset "regularized filtered projection" begin
        Jd = sparse(reshape([1.0, 0.0], 1, 2))
        result = project_row_null(
            Jd,
            [2.0, 3.0];
            regularization_atol = 1.0,
            ordering = :fixed,
        )

        @test result.regularized
        @test result.rank == 1
        @test result.row_component ≈ [1.0, 0.0] atol = 1e-12
        @test result.null_component ≈ [1.0, 3.0] atol = 1e-12
        @test result.null_action.absolute ≈ 1.0 atol = 1e-12
        @test result.idempotence.euclidean_relative ≈ 0.5 atol = 1e-12
        @test result.orthogonality.euclidean_cosine > 0.0
    end

    @testset "zero matrix and determinism" begin
        Jd = spzeros(3, 2)
        d = [2.0, -1.0]
        first_result =
            project_row_null(Jd, d; ordering = :colamd)
        second_result =
            project_row_null(Jd, d; ordering = :colamd)

        @test first_result.rank == 0
        @test first_result.row_component == zeros(2)
        @test first_result.null_component == d
        @test first_result.row_component == second_result.row_component
        @test first_result.null_component == second_result.null_component
        @test first_result.energies == second_result.energies
        @test first_result.orthogonality == second_result.orthogonality
        @test first_result.idempotence == second_result.idempotence
    end

    @testset "input validation" begin
        Jd = sparse([1.0 0.0])
        @test_throws DimensionMismatch project_row_null(Jd, [1.0])
        @test_throws ArgumentError project_row_null(
            Jd,
            [1.0, 2.0];
            rank_rtol = -1.0,
        )
        @test_throws ArgumentError project_row_null(
            Jd,
            [1.0, 2.0];
            scale_mode = :unknown,
        )
        @test_throws ArgumentError project_row_null(
            Jd,
            [1.0, 2.0];
            coordinate_scales = [1.0, 1.0],
        )
        @test_throws DimensionMismatch project_row_null(
            Jd,
            [1.0, 2.0];
            scale_mode = :scale_aware,
            coordinate_scales = [1.0],
        )
        @test_throws ArgumentError project_row_null(
            Jd,
            [1.0, 2.0];
            scale_mode = :scale_aware,
            coordinate_scales = [1.0, 0.0],
        )
        @test_throws ArgumentError project_row_null(
            sparse([NaN 0.0]),
            [1.0, 2.0],
        )
        @test_throws ArgumentError project_row_null(
            Jd,
            [1.0, Inf],
        )
        @test_throws ArgumentError project_row_null(
            Jd,
            [1.0, 2.0];
            ordering = :unknown,
        )
    end
end
