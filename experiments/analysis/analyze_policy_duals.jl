# Do the policy multipliers ψ absorb the goal level's turning residual at the
# swerving 3-level equilibrium? Decompose the saved solution vectors into
# named variable blocks via kkt.z_symbolic, print the ψ magnitudes, and
# re-evaluate the KKT residual F(z) with the ψ entries zeroed. If ψ carries
# the goal's unbalanced pull, ‖ψ‖ should be large at the swerving point and
# zeroing it should make the stationarity rows jump by the raw turning
# residual (~0.7); at the direct point both effects should be far smaller.
#
# Requires experiments/Robotic_arm.jl in its 3-LEVEL configuration, plus the
# saved runs and results/cross_warmstart_solution.jld2 (direct-basin point).
using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "experiments"); io = devnull)

using JLD2
using LinearAlgebra
using ReducedGOOP

include(joinpath(REPO_ROOT, "experiments", "Robotic_arm.jl"))

scenario_config = Robotic_arm.demo_scenario_config()
(; problem, flatten_parameters) = Robotic_arm.get_setup(scenario_config)
@assert length(problem.preferences[1]) == 3

instance_states = (;
    initial_state1 = scenario_config.base_initial_state1,
    initial_state2 = scenario_config.base_initial_state2,
    initial_state3 = scenario_config.initial_state3,
)
θ = Robotic_arm.build_instance_parameters(
    flatten_parameters,
    instance_states,
    scenario_config,
).θ

println("building 3-level KKT system...")
kkt = @time ReducedGOOP.generate_slacked_reduced_kkt_system(
    problem;
    backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
    backend_options = (;),
    codegen = :fast_differentiation,
)

# ── Group z coordinates by variable name prefix (e.g. "ψ_1_2", "λ_2_1") ──
prefix(sym) = split(string(sym), "[")[1]
groups = Dict{String,Vector{Int}}()
for (i, sym) in enumerate(kkt.z_symbolic)
    push!(get!(groups, prefix(sym), Int[]), i)
end
println("\nvariable blocks in z (name => length):")
for name in sort(collect(keys(groups)))
    println("  $name => $(length(groups[name]))")
end
ψ_indices = sort(
    reduce(vcat, [idx for (name, idx) in groups if startswith(name, "ψ")]; init = Int[]),
)
println("total ψ coordinates: $(length(ψ_indices)) of $(kkt.variable_dimension)")

residual(z) = (
    val = zeros(kkt.kkt_dimension);
    kkt.F!(val, z; θ = collect(θ), ϵ = 0.1, η = 0.0);
    val
)

runs_root = joinpath(REPO_ROOT, "data", "robotic_arm_open_loop", "runs")
points = [
    "swerving 3-level equilibrium" => collect(
        JLD2.load_object(
            joinpath(
                runs_root,
                "Robotic_arm_single_robot_agent_3_levels",
                "data",
                "problem",
                "solution",
                "solution_dict_instance_1_eps0.1.jld2",
            ),
        )["z"],
    ),
    "direct-basin 3-level point  " => collect(
        JLD2.load_object(joinpath(@__DIR__, "results", "cross_warmstart_solution.jld2"))["z"],
    ),
]

fmt(v) = round(v; sigdigits = 4)
for (label, z) in points
    @assert length(z) == kkt.variable_dimension
    F = residual(z)
    zψ0 = copy(z);
    zψ0[ψ_indices] .= 0.0
    F0 = residual(zψ0)
    ΔF = F0 .- F
    println("\n══ $label ══")
    println(
        "  ‖F(z)‖₂ = $(fmt(norm(F)))  (solver kkt_error metric),  ‖F(z)‖∞ = $(fmt(norm(F, Inf)))",
    )
    for name in sort(collect(keys(groups)))
        startswith(name, "ψ") || continue
        v = z[groups[name]]
        println(
            "  $name: max|ψ| = $(fmt(maximum(abs, v))), ‖ψ‖₂ = $(fmt(norm(v))), mean|ψ| = $(fmt(sum(abs, v) / length(v)))",
        )
    end
    for name in sort(collect(keys(groups)))
        startswith(name, "λ") || continue
        v = z[groups[name]]
        println("  $name: max|λ| = $(fmt(maximum(abs, v)))")
    end
    println("  after zeroing ψ:  ‖F(z_ψ=0)‖∞ = $(fmt(norm(F0, Inf)))")
    println(
        "  ψ contribution to residual rows: max|ΔF| = $(fmt(norm(ΔF, Inf))), #rows with |ΔF| > 0.01: $(count(>(0.01), abs.(ΔF)))",
    )
end
