"""
The manuscript's Fig 3 (§6.2): the local convergence of Algorithm 5.1, as a focused study around one solution.

For one instance of each objective class of the random GOOP study (`MonteCarlo_RandomGOOP.jl`, the same sampler
and seeds, so the instance is one of Table 2's), each system (the reduced (5.2) and the ρ-perturbed complete
(6.1)) and each ρ in `rhos` (1 and 0.1, as Fig 3 has them), with the study's settings:

 1. the center y*_ρ: the system's solution at ρ, from the study's cold start at z = 0 down the stages in `rhos`,
    each to ‖K_ρ‖₂ < `tol_inner` (`centers`);
 2. `num_starts` starts drawn uniformly from the ball of `radius` around y*_ρ (`ReducedGOOP.ball_starts`), in
    each of `modes`: `:primal_only` (the primal coordinates move; the duals and slacks stay at y*_ρ's values) or
    `:whole_iterate` (every coordinate moves; draws that leave a slack or dual at or below 0 are redrawn). In
    `:primal_only` the reduced and the complete system get the same primal offsets;
 3. Algorithm 5.1 held at ρ from every start (`ReducedGOOP.solve_multi` with `PDIP`; `tol = 0`, so a run stops
    only at ‖K_ρ‖₂ < `tol_inner`, at a line-search failure or after `max_inner_iters` steps), recording ‖K_ρ‖₂ at
    every iterate.

Writes, to a new directory under `output_dir`: traces.csv (one row per iterate: mode, class, K, system, ρ, start,
step, ‖K_ρ‖₂, the step size taken from it; accepted runs only), starts.csv (one row per start: its distance
from y*_ρ, whether its run was accepted and, if so, the run's outcome), centers.csv (per class, system and ρ: ‖K_ρ‖₂ at y*_ρ and, in `:whole_iterate`, the fraction of the
ball's draws that keep the sign box positive), results.jls and run_info.txt; then the figure, by
`RandomGOOP_plotting.py --local-convergence`.

    julia --project=experiments -e 'include("experiments/RandomGOOP_local_convergence.jl");
        RandomGOOP_local_convergence.run()'
"""
module RandomGOOP_local_convergence

using LinearAlgebra: norm
using Random: Xoshiro, randn, rand
using Printf: @printf
using Serialization: serialize
import ReducedGOOP
include(joinpath(@__DIR__, "MonteCarlo_RandomGOOP.jl"))
const MRG = MonteCarlo_RandomGOOP
const MCI = MRG.MCI

"""
Default settings; replace entries with `run(overrides = (; num_starts = 5))`. The instances are the study's
(`study_overrides` changes the study's settings they are sampled and solved with; by default none).
"""
default_settings() = (;
    modes=[:primal_only, :whole_iterate],  # ball_starts' modes, each a figure
    num_starts=20,                         # starts per class, system, ρ and mode
    radius=1.0,                            # the ball's radius, in ‖·‖₂
    rhos=[1.0, 0.1],                       # the stages; the first stages of the study's rho_schedule
    # One instance per class: its K and its trial number in the study.
    instances=(; quadratic=(; num_levels=4, trial=1), nonquadratic=(; num_levels=3, trial=1)),
    seed=1,                                # the starts' random stream (start_rng)
    linear_solver=:klu,                    # PDIPOptions.linear_solver, as the study's
    tol_inner=1e-11,                       # a run stops at ‖K_ρ‖₂ < tol_inner
    max_inner_iters=1000,                  # t_max
    positive_draws=10_000,                 # draws for the :whole_iterate positive fraction (centers.csv)
    study_overrides=(;),
    output_dir=joinpath(dirname(@__DIR__), "output", "random_goop_local_convergence"),
)

function validated_settings(overrides)
    defaults = default_settings()
    unknown = setdiff(keys(overrides), keys(defaults))
    isempty(unknown) || error("Unknown setting(s) in overrides: $(join(unknown, ", "))")
    settings = merge(defaults, overrides)
    all(in((:primal_only, :whole_iterate)), settings.modes) || error("modes must be :primal_only and/or :whole_iterate")
    study = MRG.validated_settings(settings.study_overrides)
    settings.rhos == study.rho_schedule[1:length(settings.rhos)] ||
        error("rhos must be the first stages of the study's rho_schedule, $(study.rho_schedule)")
    (; settings, study)
end

"""
The starts' random stream for one class, ρ and mode: the reduced and the complete system draw from the same
state, so in `:primal_only` (as many primal coordinates) they get the same offsets.
"""
start_rng(settings, class_index, rho_index, mode) =
    Xoshiro(settings.seed + 1000class_index + 10rho_index + findfirst(==(mode), (:primal_only, :whole_iterate)))

"""
The fraction of uniform draws from the ball of `radius` around `center`, in all of z, that keep
`kkt.nonnegative_dims` positive: how many draws `ball_starts` makes per start in `:whole_iterate`. A diagnostic
for centers.csv; the draw is `ball_starts`' own.
"""
function positive_fraction(kkt, center, radius, draws, rng)
    d = kkt.variable_dimension
    count(1:draws) do _
        direction = randn(rng, d)
        z = center .+ direction .* (radius * rand(rng)^(1 / d) / norm(direction))
        all(>(0), view(z, kkt.nonnegative_dims))
    end / draws
end

"PDIPOptions of a run held at ρ: no ‖K₀‖ stop (tol = 0), so it ends at ‖K_ρ‖₂ < tol_inner."
fixed_rho_options(settings, ρ) = ReducedGOOP.PDIPOptions(;
    ρ_schedule=[Float64(ρ)],
    max_inner_iters=settings.max_inner_iters,
    tol=0.0,
    tol_inner=settings.tol_inner,
    linear_solver=settings.linear_solver,
    record_convergence=true,
)

"""
The centers y*_ρ of one system: from the study's cold start (MonteCarlo_RandomGOOP.cold_initialization), one
stage per ρ in `rhos`, each started at the previous one's solution and solved to ‖K_ρ‖₂ < `tol_inner`
(fixed_rho_options). Errors unless every stage gets there.
"""
function centers(formulation, θ, initial_guess, settings, study)
    (; kkt) = formulation
    y = MRG.cold_initialization(formulation, initial_guess, θ, study)
    map(settings.rhos) do ρ
        output = ReducedGOOP.solve(ReducedGOOP.PDIP(), kkt, θ; z₀=y, options=fixed_rho_options(settings, ρ))
        output.termination_reason === :inner_tolerance_reached ||
            error("the stage ρ = $ρ ended with $(output.termination_reason), ‖K_ρ‖₂ = $(output.norm_K_ρ)")
        y = output.z
        copy(y)
    end
end

function run(; overrides=(;))
    (; settings, study) = validated_settings(overrides)
    directory = MCI.run_directory(settings.output_dir)
    println("Local convergence of Algorithm 5.1 (PDIP, $(settings.linear_solver)): $(settings.num_starts) starts in the ball of radius ",
        "$(settings.radius) around y*_ρ, ρ ∈ $(settings.rhos), modes $(settings.modes). Output: $directory")
    traces, starts_rows, center_rows, records = Any[], Any[], Any[], Any[]
    classes = filter(c -> haskey(settings.instances, c.name), MRG.objective_classes(study))
    for (class_index, class) in enumerate(classes)
        (; num_levels, trial) = settings.instances[class.name]
        instance = MRG.sample_instances(MRG.instance_rng(class, num_levels, study), class, num_levels, study)[trial]
        systems = MRG.build_systems(class, num_levels, study)
        isnothing(systems.complete) && error("the study builds no complete system for $(class.name), K = $num_levels")
        (; θ, initial_guess) = MRG.instance_inputs(instance, study)
        for (system, formulation) in MRG.solved_systems(systems)
            (; kkt) = formulation
            ys = centers(formulation, θ, initial_guess, settings, study)
            for (rho_index, (ρ, center)) in enumerate(zip(settings.rhos, ys))
                residual = zeros(kkt.kkt_dimension)
                kkt.K_ρ!(residual, center; θ, ρ)
                fraction = positive_fraction(kkt, center, settings.radius, settings.positive_draws, Xoshiro(settings.seed))
                push!(center_rows, (class.name, num_levels, system, ρ, kkt.variable_dimension, length(kkt.primal_dims), norm(residual), fraction))
                for mode in settings.modes
                    rng = start_rng(settings, class_index, rho_index, mode)
                    starts = ReducedGOOP.ball_starts(kkt, center, settings.num_starts; radius=settings.radius, mode, rng)
                    # A run is accepted when it reaches ‖K_ρ‖₂ < tol_inner; solve_multi keeps those, with their
                    # starts, and merges their primal points.
                    accept(r) = r.termination_reason === :inner_tolerance_reached
                    multi = ReducedGOOP.solve_multi(ReducedGOOP.PDIP(), kkt, θ, starts; options=fixed_rho_options(settings, ρ), accept)
                    push!(records, (; mode, class=class.name, num_levels, trial, system, ρ, center, starts, multi))
                    for (j, z₀) in enumerate(starts)
                        kept = findfirst(==(z₀), multi.starts)
                        distances = (norm(z₀ - center), norm(z₀[kkt.primal_dims] - center[kkt.primal_dims]))
                        if isnothing(kept)
                            push!(starts_rows, (mode, class.name, num_levels, system, ρ, j, distances..., false, "", "", "", ""))
                            continue
                        end
                        output = multi.results[kept]
                        sizes = output.step_size_history
                        for (k, value) in enumerate(output.norm_K_ρ_history)
                            push!(traces, (mode, class.name, num_levels, system, ρ, j, k - 1, value, k <= length(sizes) ? sizes[k] : ""))
                        end
                        push!(starts_rows, (mode, class.name, num_levels, system, ρ, j, distances..., true, output.newton_steps,
                            output.termination_reason, output.norm_K_ρ, norm(output.x - center[kkt.primal_dims])))
                    end
                    @printf("%-12s %-12s K = %d  %-8s ρ = %-4g  %-13s accepted %2d/%d · Newton steps %s · points after the dedupe %d\n",
                        class.name, "trial $trial", num_levels, system, ρ, mode, length(multi.results), length(starts),
                        join(sort([r.newton_steps for r in multi.results]), ","), length(multi.points))
                end
            end
        end
    end
    MRG.write_csv(joinpath(directory, "traces.csv"), ("mode", "class", "K", "system", "rho", "start", "step", "norm_K_ρ", "step_size"), traces)
    MRG.write_csv(joinpath(directory, "starts.csv"), ("mode", "class", "K", "system", "rho", "start", "distance", "primal_distance",
        "accepted", "newton_steps", "termination_reason", "norm_K_ρ", "final_primal_distance"), starts_rows)
    MRG.write_csv(joinpath(directory, "centers.csv"), ("class", "K", "system", "rho", "variables", "primal_variables", "norm_K_ρ",
        "positive_fraction"), center_rows)
    serialize(joinpath(directory, "results.jls"), (; settings, study, records, julia=string(VERSION)))
    write(joinpath(directory, "run_info.txt"), "Local convergence study, $(Base.Libc.strftime("%Y-%m-%d %H:%M", time())), Julia $VERSION.\n",
        "Code: $(readchomp(Cmd(`git rev-parse --short HEAD`; dir=@__DIR__)))$(isempty(readchomp(Cmd(`git status --porcelain src experiments`; dir=@__DIR__))) ? "" : " plus uncommitted changes in src/ or experiments/").\n",
        "Settings: $(settings)\n")
    println("Saved traces.csv, starts.csv, centers.csv, results.jls and run_info.txt: $directory")
    plot_results(directory)
    directory
end

"The figures of a run directory, by RandomGOOP_plotting.py --local-convergence (as MonteCarlo_RandomGOOP.plot_results)."
function plot_results(directory)
    script = joinpath(@__DIR__, "RandomGOOP_plotting.py")
    command = `uv run --quiet --no-project --isolated --with matplotlib --with numpy python -I $script --local-convergence $directory`
    try
        Base.run(command)
        println("Saved the figures: $(joinpath(directory, "figures"))")
    catch err
        @warn "Plotting failed; the CSV files are saved. Rerun plot_results(\"$directory\")." exception = err
    end
    directory
end

end
