module ExamplesQP
using QuasiGOOP

using ParametricMCPs
using LinearAlgebra: I, norm, pinv
using Symbolics
using SymbolicTracingUtils
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes, blockedrange
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