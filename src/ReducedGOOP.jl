module ReducedGOOP

using SymbolicTracingUtils: SymbolicTracingUtils
using Symbolics: Symbolics
using BlockArrays: BlockArrays, BlockArray, Block, blockedrange
using SparseArrays: SparseArrays
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, norm, ldiv!
using KLU: KLU
using TimerOutputs: TimerOutput, @timeit

const TO = TimerOutput()

include("kkt_metadata.jl")
include("goop_kkt_system.jl")
include("goop.jl")
include("warmstart.jl")
include("solver.jl")
include("parametric_optimization_problem.jl")

export PrimalCoordinateSpec,
    EqualityCoordinateSpec,
    GOOPSemanticLayout,
    KKTVariableCoordinate,
    KKTEquationCoordinate,
    GOOPKKTMetadata,
    KKTDualTransportMap,
    kkt_variable_metadata,
    kkt_equation_metadata,
    receding_dual_transport_map,
    transport_receding_duals,
    SELECTIVE_WARMSTART_MODES,
    kkt_variable_blocks,
    build_selective_warmstart,
    shift_receding_trajectories

end # module ReducedGOOP
