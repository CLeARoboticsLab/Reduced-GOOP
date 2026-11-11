"""
The manuscript's Fig 3 (§6.2): the local convergence of the study's solvers, as a focused study around one
solution.

For one instance of each objective class of the random GOOP study (`MonteCarlo_RandomGOOP.jl`, the same sampler
and seeds, so the instance is one of Table 2's), each system (the study's reduced system, by default in
Scholtes form, and the ρ-perturbed complete (6.1)) and each ρ in `rhos` (1 and 0.1, as Fig 3 has them), with
the study's settings, among them each system's solver and linear solver (by default ProjectedLM with :normal
for both; `study_overrides = (; reduced_solver = :pdip, reduced_linear_solver = :klu, …)` gives Algorithm 5.1
on the reduced (5.2)):

 1. the center y*_ρ: the system's solution at ρ, from the study's cold start at z = 0 down the stages in `rhos`,
    each to ‖K_ρ‖₂ < `tol_inner` (`centers`);
 2. `num_starts` starts around y*_ρ, in each of `modes`:
      `:primal_only`    drawn uniformly from the ball of `radius` in the primal coordinates
                        (`ReducedGOOP.ball_starts`); the duals and slacks stay at y*_ρ's values. The reduced and
                        the complete system get the same primal offsets;
      `:whole_iterate`  every coordinate moves, by an offset δ drawn uniformly from the ball of `radius` in all
                        of z: additively, z_k + δ_k, except the sign box (`nonnegative_dims`), which moves
                        relatively, z_k·exp(δ_k), so every start stays positive (whole_iterate_starts);
 3. the system's solver held at ρ from every start (`ReducedGOOP.solve_multi`; `tol = 0`, so a run stops
    only at ‖K_ρ‖₂ < `tol_inner`, at a line-search failure or after `max_inner_iters` steps), recording ‖K_ρ‖₂ at
    every iterate.

Writes, to a new directory under `output_dir`: traces.csv (one row per iterate: mode, class, K, system, ρ, start,
step, ‖K_ρ‖₂, the step size taken from it; accepted runs only), starts.csv (one row per start: its distance
‖z₀ − y*_ρ‖₂ from y*_ρ, whether its run was accepted and, if so, the run's outcome), centers.csv (per class,
system and ρ: ‖K_ρ‖₂ at y*_ρ), results.jls and run_info.txt; then the figure, by
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
(`study_overrides` changes the study's settings they are sampled and solved with, each system's solver and
linear solver among them; by default none).
This script sets no solver of its own; it takes the solver from the settings in MonteCarlo_RandomGOOP.jl.
"""
default_settings() = (;
    modes=[:primal_only, :whole_iterate],  # the start modes, each a figure
    num_starts=20,                         # starts per class, system, ρ and mode
    radius=1.0,                            # the ball's radius, in ‖·‖₂ of the offset δ
    rhos=[1.0, 0.1],                       # the stages; the first stages of the study's rho_schedule
    # One instance per class: its K and its trial number in the study.
    instances=(; quadratic=(; num_levels=4, trial=1), nonquadratic=(; num_levels=3, trial=1)),
    seed=1,                                # the starts' random stream (start_rng)
    tol_inner=1e-11,                       # a run stops at ‖K_ρ‖₂ < tol_inner
    max_inner_iters=1000,                  # t_max
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
`m` starts around `center`, a full iterate positive on `kkt.nonnegative_dims`, that move every coordinate: an
offset δ drawn uniformly from the ball of `radius` in all of z (a Gaussian direction scaled to `radius · U^(1/d)`,
as `ReducedGOOP.ball_starts` draws), applied additively, z_k + δ_k, except on the sign box, where it is applied
relatively, z_k·exp(δ_k). Every start is then positive on the sign box with no redraw, also where `center` is
close to 0 there (as at a Scholtes solution, whose exact products σ̂ ⊙ ϕ = 0 leave one factor at the boundary),
and the two systems get the same rule.
"""
function whole_iterate_starts(kkt, center, m; radius, rng)
    all(>(0), view(center, kkt.nonnegative_dims)) || throw(ArgumentError("center must be positive on nonnegative_dims"))
    d = kkt.variable_dimension
    signed = falses(d)
    signed[kkt.nonnegative_dims] .= true
    map(1:m) do _
        direction = randn(rng, d)
        δ = direction .* (radius * rand(rng)^(1 / d) / norm(direction))
        [signed[k] ? center[k] * exp(δ[k]) : center[k] + δ[k] for k in 1:d]
    end
end

"The starts of `mode` around `center`: `ReducedGOOP.ball_starts` for :primal_only, whole_iterate_starts for :whole_iterate."
function starts_around(kkt, center, mode, settings, rng)
    mode === :primal_only && return ReducedGOOP.ball_starts(kkt, center, settings.num_starts; settings.radius, mode, rng)
    whole_iterate_starts(kkt, center, settings.num_starts; settings.radius, rng)
end

"""
Options of a run of `formulation`'s solver held at ρ: the study's stage options (MonteCarlo_RandomGOOP.stage_options:
its linear solver and η) with no ‖K₀‖ stop (tol = 0), so the run ends at ‖K_ρ‖₂ < tol_inner.
"""
function fixed_rho_options(formulation, settings, study, ρ)
    options = MRG.stage_options(formulation, study, ρ)
    fields = (; (name => getfield(options, name) for name in fieldnames(typeof(options)))...)
    typeof(options)(; fields..., tol=0.0, tol_inner=settings.tol_inner, max_inner_iters=settings.max_inner_iters)
end

"""
The centers y*_ρ of one system: from the study's cold start (MonteCarlo_RandomGOOP.cold_initialization), one
stage per ρ in `rhos`, each started at the previous one's solution and solved to ‖K_ρ‖₂ < `tol_inner`
(fixed_rho_options). Errors unless every stage gets there.
"""
function centers(formulation, θ, initial_guess, settings, study)
    (; kkt) = formulation
    y = MRG.cold_initialization(formulation, initial_guess, θ, study)
    map(settings.rhos) do ρ
        options = fixed_rho_options(formulation, settings, study, ρ)
        output = ReducedGOOP.solve(MRG.solver_method(formulation), kkt, θ; z₀=y, options)
        output.termination_reason === :inner_tolerance_reached ||
            error("the stage ρ = $ρ ended with $(output.termination_reason), ‖K_ρ‖₂ = $(output.norm_K_ρ)")
        y = output.z
        copy(y)
    end
end

function run(; overrides=(;))
    (; settings, study) = validated_settings(overrides)
    directory = MCI.run_directory(settings.output_dir)
    println("Local convergence, $(MRG.system_label(MRG.reduced_system(study))) by $(MRG.solver_label(MRG.solver_settings(study, :reduced)...)) and ",
        "$(MRG.system_label(:perturbed)) by $(MRG.solver_label(MRG.solver_settings(study, :complete)...)): $(settings.num_starts) starts in the ball of radius ",
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
                push!(center_rows, (class.name, num_levels, system, ρ, kkt.variable_dimension, length(kkt.primal_dims), norm(residual)))
                for mode in settings.modes
                    rng = start_rng(settings, class_index, rho_index, mode)
                    starts = starts_around(kkt, center, mode, settings, rng)
                    # A run is accepted when it reaches ‖K_ρ‖₂ < tol_inner; solve_multi keeps those, with their
                    # starts, and merges their primal points.
                    accept(r) = r.termination_reason === :inner_tolerance_reached
                    options = fixed_rho_options(formulation, settings, study, ρ)
                    multi = ReducedGOOP.solve_multi(MRG.solver_method(formulation), kkt, θ, starts; options, accept)
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
                    @printf("%-12s %-12s K = %d  %-8s ρ = %-4g  %-13s accepted %2d/%d · Newton steps %s · points after the dedupe (1 : the same primal reached) %d\n",
                        class.name, "trial $trial", num_levels, system, ρ, mode, length(multi.results), length(starts),
                        join(sort([r.newton_steps for r in multi.results]), ","), length(multi.points))
                end
            end
        end
    end
    MRG.write_csv(joinpath(directory, "traces.csv"), ("mode", "class", "K", "system", "rho", "start", "step", "norm_K_ρ", "step_size"), traces)
    MRG.write_csv(joinpath(directory, "starts.csv"), ("mode", "class", "K", "system", "rho", "start", "distance", "primal_distance",
        "accepted", "newton_steps", "termination_reason", "norm_K_ρ", "final_primal_distance"), starts_rows)
    MRG.write_csv(joinpath(directory, "centers.csv"), ("class", "K", "system", "rho", "variables", "primal_variables", "norm_K_ρ"),
        center_rows)
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
