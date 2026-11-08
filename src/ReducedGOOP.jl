module ReducedGOOP

using SymbolicTracingUtils: SymbolicTracingUtils
using Symbolics: Symbolics, SymbolicUtils
using BlockArrays: BlockArrays, BlockArray, Block, blockedrange
using SparseArrays: SparseArrays
using InvertedIndices: Not
using LinearAlgebra:
    LinearAlgebra, I, Symmetric, UpperTriangular, cholesky, cholesky!, dot, issuccess, ldiv!, ldlt, mul!,
    norm, pinv, qr, rank, triu
using KLU: KLU
using Random: Random
using TimerOutputs: TimerOutput, @timeit
using PrecompileTools: @setup_workload, @compile_workload

const TO = TimerOutput()

include("goop_kkt_system.jl")
include("goop.jl")
include("solver.jl")
include("multistart.jl")

include("precompile.jl")

end # module ReducedGOOP
