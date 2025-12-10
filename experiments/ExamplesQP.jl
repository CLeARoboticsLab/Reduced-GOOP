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

# include("Intersection.jl")
# include("goop_comparison_QP.jl")
# include("trilevel_QP.jl")
include("nonlinear_goop_test.jl")

end