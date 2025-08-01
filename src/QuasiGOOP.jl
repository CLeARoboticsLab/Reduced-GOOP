module QuasiGOOP

using TrajectoryGamesBase: to_blockvector
using SymbolicTracingUtils: SymbolicTracingUtils
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes, blockedrange
using SparseArrays: SparseArrays
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, I, norm, eigvals
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase
using SymbolicTracingUtils: SymbolicTracingUtils
using JLD2: JLD2
using ParametricMCPs

include("goop_kkt_system.jl")
include("goop.jl")

include("parametric_optimization_problem.jl")

end # module QuasiGOOP
