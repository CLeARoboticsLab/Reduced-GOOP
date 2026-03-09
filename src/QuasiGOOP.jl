module QuasiGOOP

using TrajectoryGamesBase: to_blockvector
using SymbolicTracingUtils: SymbolicTracingUtils
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes, blockedrange
using SparseArrays: SparseArrays
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, I, norm, eigvals, cond, svdvals
using LinearSolve: LinearSolve, LinearProblem, init, solve!
using SciMLBase: SciMLBase
using SymbolicTracingUtils: SymbolicTracingUtils
using JLD2: JLD2
using ParametricMCPs
using BenchmarkTools

include("goop_kkt_system.jl")
include("goop.jl")
include("solver.jl")

include("parametric_optimization_problem.jl")

end # module QuasiGOOP
