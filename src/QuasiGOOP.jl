module QuasiGOOP

using LinearAlgebra

using ParametricMCPs: ParametricMCPs, ParametricMCP
using MixedComplementarityProblems:
    MixedComplementarityProblems, PrimalDualMCP, InteriorPoint
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes
using SparseArrays: SparseArrays
using LinearAlgebra: LinearAlgebra, I, norm, eigvals
using LinearSolve: LinearProblem, init, solve!, KrylovJL_GMRES, UMFPACKFactorization
using SciMLBase: SciMLBase
using SymbolicTracingUtils: SymbolicTracingUtils
using JLD2, BenchmarkTools

include("parametric_ordered_preferences_MPCC_game.jl") # TODO: change file name to reflect quasi goop
export ParametricOrderedPreferencesMPCCGame, solve_relaxed_pop_game, total_dim

# include("primal_dual.jl")
# include("quasi_goop.jl")
# include("solver.jl")
# export PrimalDualSys, ordered_preferences, ParametricQuasiGOOP, solve

end # module QuasiGOOP
