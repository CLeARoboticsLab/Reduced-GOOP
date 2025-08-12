module ExamplesQP
using QuasiGOOP

using ParametricMCPs
using LinearAlgebra: I, norm, pinv
using Symbolics
using SymbolicTracingUtils
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes

include("trilevel_QP.jl")
# include("Intersection.jl")
end