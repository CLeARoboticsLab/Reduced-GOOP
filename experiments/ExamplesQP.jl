module ExamplesQP
using QuasiGOOP

using ParametricMCPs
using LinearAlgebra: I, norm, pinv
using Symbolics
using SymbolicTracingUtils
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes, blockedrange
using NonlinearSolve
using InvertedIndices: Not

# include("trilevel_QP.jl")
# include("Intersection.jl")
include("goop_comparison_QP.jl")
end