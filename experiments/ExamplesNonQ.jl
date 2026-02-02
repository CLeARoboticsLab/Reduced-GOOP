module ExamplesNonQ
using QuasiGOOP

using ParametricMCPs
using LinearAlgebra: I, norm, pinv
using Symbolics
using SymbolicTracingUtils
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes, blockedrange
using NonlinearSolve
using InvertedIndices: Not
using Random


include("run_nonlinear_goop_exp.jl")

end