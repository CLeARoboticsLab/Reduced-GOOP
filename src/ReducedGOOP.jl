module ReducedGOOP

using SymbolicTracingUtils: SymbolicTracingUtils
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blockedrange
using SparseArrays: SparseArrays
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, norm, ldiv!
using KLU: KLU
using Random: Random
using TimerOutputs: TimerOutput, @timeit

const TO = TimerOutput()

include("goop_kkt_system.jl")
include("goop.jl")
include("solver.jl")
include("parametric_optimization_problem.jl")

# Scholtes relaxation (opt-in via `complementarity = :scholtes`), ported from
# ScholtesReducedGOOP.jl by Jingqi Li: the KKT blocks, the residual and sign box,
# the linear algebra, the projected bound rule, the ρ-homotopy solver, and the
# certificate and multistart layers.
include("scholtes_kkt.jl")
include("scholtes_residual.jl")
include("scholtes_linsolve.jl")
include("scholtes_step.jl")
include("scholtes_solver.jl")
include("scholtes_certify.jl")
include("scholtes_multistart.jl")

end # module ReducedGOOP
