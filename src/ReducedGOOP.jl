module ReducedGOOP

using SymbolicTracingUtils: SymbolicTracingUtils
using Symbolics: Symbolics, SymbolicUtils
using BlockArrays: BlockArrays, BlockArray, Block, blockedrange
using SparseArrays: SparseArrays
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, norm, ldiv!
using KLU: KLU
using Random: Random
using TimerOutputs: TimerOutput, @timeit
using PrecompileTools: @setup_workload, @compile_workload

const TO = TimerOutput()

include("goop_kkt_system.jl")
include("goop.jl")
include("solver.jl")

# The Scholtes engine (`Scholtes()`), ported from ScholtesReducedGOOP.jl by Jingqi Li: the KKT
# blocks, the residual and sign box, the linear algebra, the projected bound rule, the ρ-homotopy
# solver, and the certificate and multistart layers. Kept with its own tests while it remains; the
# Scholtes system is built by `generate_slacked_scholtes_reduced_kkt_system` and solved by `ProjectedLM`.
include("scholtes_kkt.jl")
include("scholtes_residual.jl")
include("scholtes_linsolve.jl")
include("scholtes_step.jl")
include("scholtes_solver.jl")
include("scholtes_complete_kkt.jl")
include("scholtes_certify.jl")
include("scholtes_multistart.jl")

include("precompile.jl")

end # module ReducedGOOP
