module QuasiGOOP

using LinearAlgebra

using ParametricMCPs: ParametricMCPs, ParametricMCP
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes
using SparseArrays: SparseArrays
using LinearAlgebra: LinearAlgebra, I, norm, eigvals
using LinearSolve: LinearProblem, init, solve!, KrylovJL_GMRES, UMFPACKFactorization
using SymbolicTracingUtils: SymbolicTracingUtils
using JLD2, BenchmarkTools

include("parametric_ordered_preferences_MPCC_game.jl")
export ParametricOrderedPreferencesMPCCGame, solve_relaxed_pop_game, total_dim

include("quasi_goop.jl")
include("primal_dual.jl")
include("solver.jl")
export PrimalDualSys, ordered_preferences, ParametricQuasiGOOP, solve

end # module QuasiGOOP
