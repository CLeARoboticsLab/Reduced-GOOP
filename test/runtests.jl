using Test: @testset, @test

using QuasiGOOP
using ParametricMCPs: ParametricMCPs

@testset "TODO: @DongHo put in a real test" begin
    """ TODO
       Test for the following QP:
                                  min_x 0.5 xᵀ M x - θᵀ x
                                  s.t.  Ax - b ≥ 0.
       Taking `y ≥ 0` as a Lagrange multiplier, we obtain the KKT conditions:
                                    G(x, y) = Mx - Aᵀy - θ = 0
                                    0 ≤ y ⟂ H(x, y) = Ax - b ≥ 0.
    """

    @test true
end
