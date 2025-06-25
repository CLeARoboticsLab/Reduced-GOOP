module Experiments

using TrajectoryGamesBase:
    AbstractDynamics, unflatten_trajectory, state_dim, control_dim, control_bounds
using TrajectoryGamesExamples: UnicycleDynamics, planar_double_integrator
using BlockArrays, JLD2, ProgressMeter, Distributions, Random, BenchmarkTools
using CairoMakie: CairoMakie
using Logging

using OrderedPreferences

include("Utils.jl")
include("N_player_KKT_Highway.jl")

include("N_player_KKT_Highway_Baseline.jl")
export run

include("N_player_KKT_Highway_GOOP.jl")
export run

include("Generate_Set_Highway.jl")
export generate_samples

include("Benchmark.jl")
export compare_two

end
