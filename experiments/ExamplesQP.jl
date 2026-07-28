module ExamplesQP
using ReducedGOOP

using LinearAlgebra: I, norm
using Symbolics
using SymbolicTracingUtils
using BlockArrays: BlockArray
using NonlinearSolve
using InvertedIndices: Not
using Random

function rand_psd(n, r)
    # n: primal dimension, r: matrix rank (<=n)
    R = randn(r, n);
    R' * R;
end

# include("Intersection.jl")
# include("goop_comparison_QP.jl")
include("trilevel_QP.jl")
# include("nonlinear_goop_test.jl")

# include("run_nonlinear_goop_exp.jl")

end
