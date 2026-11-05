module SolverTests

# The two solvers of src/solver.jl, solve(PDIP(), …) (Algorithm 5.1) and solve(ProjectedLM(), …) (the
# projected Levenberg–Marquardt method), and the pieces they share (_newton_step!, _gain_ratio,
# _scheduled_η), on the reduced (5.2), complete (:exact, :perturbed) and Scholtes systems. The Scholtes
# system's own tests, solved by ProjectedLM, are in test/scholtes.jl; the complete systems' in
# test/complete_kkt.jl.

using Test, ReducedGOOP, BlockArrays, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "benchmark_problems.jl"))
const RG = ReducedGOOP
const SOLVERS = ((RG.PDIP(), RG.PDIPOptions), (RG.ProjectedLM(), RG.ProjectedLMOptions))

"A one-player, one-level problem without constraints: min `objective(x)` over x ∈ ℝ."
function free_problem(objective)
    RG.ParametricGOOP(BlockArray([0.0], [1]), BlockArray(Float64[], [0]);
        preferences = [Function[(x, θ) -> objective(x[1])]], is_prioritized_constraint = [[false]],
        equality_constraints = [nothing], inequality_constraints = [nothing])
end

"""
Two players, three levels: ‖x‖² on top, (b_i − 0.3)²/2, and the innermost `innermost`(a_i − 1), with the
coupled constraint a_i + 0.1 a_j ≤ 0.5, whose normal lies in the innermost objective's column space. The
equilibrium is unique, x* = (5/11, 0.3, 5/11, 0.3), and the constraint is active with γ_i = 6/11 > 0. The
lifted multipliers are not, so J is rank-deficient. (The benchmark problems violate the column-space rule,
and Algorithm 5.1 jams on them.)
"""
function coupled_problem(innermost)
    RG.ParametricGOOP(BlockArray(zeros(4), [2, 2]), BlockArray(Float64[], [0]);
        preferences = [Function[(x, t) -> sum(abs2, x), (x, t) -> (x[2i] - 0.3)^2 / 2, (x, t) -> innermost(x[2i - 1] - 1)]
                       for i in 1:2],
        is_prioritized_constraint = [[false, false, false] for _ in 1:2],
        equality_constraints = [nothing, nothing],
        inequality_constraints = [(x, t) -> [0.5 - x[1] - 0.1x[3]], (x, t) -> [0.5 - x[3] - 0.1x[1]]])
end
const COUPLED_EXPECTED = [5 / 11, 0.3, 5 / 11, 0.3]

"The Scholtes system's start at the primal `x` and the level ρ (initialize_scholtes!'s cold start)."
scholtes_start(k, θ, x, ρ) = RG.initialize_scholtes!(zeros(k.variable_dimension), k, θ, x, ρ; warm = false)

"""
`PDIP` down a ρ chain from the primal start x = 0, one solve per stage, as experiments/MonteCarlo_RandomGOOP.jl
does: every stage runs to its inner tolerance (tol = 0). Returns the stages' outputs.
"""
function pdip_chain(k; kwargs...)
    θ, z, scale = zeros(k.parameter_dimension), zeros(length(k.primal_dims)), sqrt(k.kkt_dimension)
    outputs = []
    for ρ in 10.0 .^ (0:-1:-10)
        options = RG.PDIPOptions(; ρ_schedule = [ρ], max_inner_iters = 1000, tol = 0.0,
            tol_inner = min(ρ, 1e-2) * scale, record_convergence = true, kwargs...)
        push!(outputs, RG.solve(RG.PDIP(), k, θ; z₀ = z, options))
        z = last(outputs).z
    end
    outputs
end

@testset "The Newton step of every linear solver" begin
    # δz with J δz = −F: `:svd`, and `:sparseQR_Jᵀ` when J has full row rank, take the minimum-norm step −J⁺F;
    # `:klu`, `:ldl` and `:normal` the Levenberg–Marquardt step −Jᵀ(JJᵀ + η²I)⁻¹F = −V diag(s / (s² + η²)) UᵀF
    # (J = U diag(s) Vᵀ); `:sparseQR_J` a least-squares step. At a point of each kind of system.
    case = build_benchmark_problem(; num_players = 1, levels = 3, kind = :quadratic)
    systems = (RG.generate_slacked_reduced_kkt_system(case.problem),
        RG.generate_slacked_complete_kkt_system(case.problem; complementarity = :perturbed),
        RG.generate_slacked_scholtes_reduced_kkt_system(case.problem))
    for k in systems
        θ, ρ, η = zeros(k.parameter_dimension), 1e-2, 1e-3
        z = k.kind === :scholtes ? RG.initialize_scholtes!(zeros(k.variable_dimension), k, θ, case.expected, ρ) :
            0.5 .+ 0.1 .* sin.(1:(k.variable_dimension))
        F = zeros(k.kkt_dimension)
        k.F!(F, z; θ, ρ)
        J = copy(k.∇F_z!.result_buffer)
        k.∇F_z!(J, z; θ, ρ)
        U, s, V = svd(Matrix(J))
        minimum_norm = -pinv(Matrix(J)) * F
        levenberg_marquardt = -V * (s ./ (s .^ 2 .+ η^2) .* (U' * F))
        δz = zeros(k.variable_dimension)
        @test RG._newton_step!(δz, J, F, :svd, η) == (; fallback = 0, η)
        @test δz == minimum_norm
        for linear_solver in (:klu, :ldl, :normal)
            @test RG._newton_step!(δz, J, F, linear_solver, η) == (; fallback = 0, η)
            @test δz ≈ levenberg_marquardt rtol = 1e-6
        end
        step = RG._newton_step!(δz, J, F, :sparseQR_Jᵀ, η)
        if rank(Matrix(J)) == k.kkt_dimension
            @test step.fallback == 0
            @test δz ≈ minimum_norm rtol = 1e-8
        else
            @test step.fallback == 1 # J without full row rank: the :svd step
            @test δz == minimum_norm
        end
        @test RG._newton_step!(δz, J, F, :sparseQR_J, η) == (; fallback = 0, η)
        @test norm(J' * (J * δz + F)) <= 1e-8 * opnorm(Matrix(J))^2 * norm(F) # the normal equations
        @test_throws ArgumentError RG._newton_step!(δz, J, F, :lu, η)
    end
    # :normal retries with 100 η when CHOLMOD cannot factorize JJᵀ + η²I. Below η ≈ 2e-162, η² underflows to 0
    # and the singular JJᵀ is factorized as it is: from η = 1e-170, the last of the six tries, at 1e-160,
    # succeeds. A zero J fails every try and takes the :svd step, reporting the η it was given.
    δz = zeros(2)
    step = RG._newton_step!(δz, sparse([1.0 0.0; 0.0 0.0]), [1.0, 1.0], :normal, 1e-170)
    @test step.fallback == 0
    @test step.η ≈ 1e-160
    @test δz == [-1.0, 0.0]
    δz = zeros(1)
    @test RG._newton_step!(δz, spzeros(1, 1), [1.0], :normal, 1e-200) == (; fallback = 1, η = 1e-200)
    @test δz == [0.0]
end

@testset "The gain ratio and the η schedule" begin
    rates = (; gain_low = 0.25, gain_high = 0.75, η_min = 1e-8, η_max = 1e2, η_tightening_rate = 1.2,
        η_loosening_rate = 3.0)
    up, down = 1 + exp(-3.0), 1 - exp(-1.2)
    next_η(η, gain_ratio, α) = RG._scheduled_η(η, gain_ratio, α; rates...)
    @test next_η(1.0, 0.25, 1.0) ≈ up # a poor step
    @test next_η(1.0, 0.75, 1.0) == 1.0
    @test next_η(1.0, 0.8, 1.0) ≈ down # a good one
    @test next_η(1.0, 1.0, 0.5) ≈ up # a shortened one
    @test next_η(1.0, 1.0, 0.99) ≈ down
    @test next_η(1.0, -Inf, 0.0) ≈ up # a failed line search
    @test next_η(1e2, -1.0, 1.0) == 1e2 # within [η_min, η_max]
    @test next_η(1e-8, 1.0, 1.0) == 1e-8
    # The actual over the predicted decrease of ½‖F‖²: 1 for a step that lands where the linear model F + JΔz
    # says, −Inf when the model predicts no decrease.
    @test RG._gain_ratio([1.0], [-1.0], 1.0, 0.0) == 1.0
    @test RG._gain_ratio([1.0], [-1.0], 1.0, 0.5) == 0.75
    @test RG._gain_ratio([1.0], [0.0], 1.0, 1.0) == -Inf
end

@testset "PDIP: Algorithm 5.1 on the perturbed reduced and complete systems" begin
    for innermost in (d -> d^2 / 2, d -> cosh(d) - 1)
        goop = coupled_problem(innermost)
        for k in (RG.generate_slacked_reduced_kkt_system(goop),
                  RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed))
            θ = zeros(k.parameter_dimension)
            # J is rank-deficient at the start (s, σ and γ at 1, x = 0), and :svd still takes −J⁺F.
            start = RG.solve(RG.PDIP(), k, θ; z₀ = zeros(4), options = RG.PDIPOptions(; max_inner_iters = 0)).z
            J = copy(k.∇F_z!.result_buffer)
            k.∇F_z!(J, start; θ, ρ = 1.0)
            @test rank(Matrix(J)) < k.kkt_dimension
            for linear_solver in (:svd, :klu, :ldl)
                outputs = pdip_chain(k; linear_solver)
                z = last(outputs).z
                @test all(o -> o.termination_reason === :inner_tolerance_reached, outputs)
                @test all(o -> o.svd_fallbacks == 0, outputs)
                # Every accepted step lowers ‖F(z; ρ)‖ strictly.
                @test all(o -> all(<(0), diff(o.relaxed_residual_history)), outputs)
                @test norm(z[k.primal_dims] - COUPLED_EXPECTED) < 1e-8
                @test minimum(z[k.nonnegative_dims]) > 0
            end
            # The ‖K₀‖ < tol stop, checked before every Newton step: with tol far above ‖K₀‖, a solve stops at
            # its first residual evaluation.
            for linear_solver in (:svd, :klu)
                options = RG.PDIPOptions(; ρ_schedule = [1.0], tol = 1e10, linear_solver)
                output = RG.solve(RG.PDIP(), k, θ; z₀ = zeros(4), options)
                @test (output.residual_evaluations, output.newton_steps) == (1, 0)
                @test output.termination_reason === :tolerance_reached
            end
        end
    end
end

@testset "PDIP: the fraction-to-the-boundary line search and the sparse QR steps" begin
    # On the reduced system (wide) and the complete one (square, and singular along the solve), whose J is
    # rank-deficient: :sparseQR_Jᵀ then takes the :svd step (a fallback), :sparseQR_J a least-squares step.
    goop = coupled_problem(d -> cosh(d) - 1)
    for k in (RG.generate_slacked_reduced_kkt_system(goop),
              RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed)),
        (options, fallbacks) in (((; linesearch = :fraction_to_boundary), false),
                                 ((; linear_solver = :sparseQR_J), false),
                                 ((; linear_solver = :sparseQR_Jᵀ), true))
        outputs = pdip_chain(k; options...)
        z = last(outputs).z
        @test all(o -> o.termination_reason === :inner_tolerance_reached, outputs)
        @test any(o -> o.svd_fallbacks > 0, outputs) == fallbacks
        @test norm(z[k.primal_dims] - COUPLED_EXPECTED) < 1e-8
        @test minimum(z[k.nonnegative_dims]) > 0
    end
end

@testset "ProjectedLM down a ρ chain on the perturbed reduced system (5.2)" begin
    # One solve per stage from z = 0, as the intersection study's reduced solves. The 3-player nonlinear case
    # levels off at ‖K₀‖/√m ≈ 1e-8 with a primal error of ≈ 4e-6; a wiring error would miss by O(1).
    for (levels, players, kind) in ((2, 1, :quadratic), (3, 3, :nonlinear))
        case = build_benchmark_problem(; num_players = players, levels, kind)
        k = RG.generate_slacked_reduced_kkt_system(case.problem)
        θ, z, scale = zeros(k.parameter_dimension), zeros(length(case.expected)), sqrt(k.kkt_dimension)
        for ρ in (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
            options = RG.ProjectedLMOptions(; ρ_schedule = [ρ], max_inner_iters = 1000, tol = 0.0,
                tol_inner = min(ρ, 1e-2) * scale, proj_eps = 1e-8, η = 1e-2, η_min = 1e-8, η_max = 1e4)
            z = RG.solve(RG.ProjectedLM(), k, θ; z₀ = z, options).z
        end
        @test norm(z[k.primal_dims] - case.expected) < 1e-5
        @test minimum(z[k.nonnegative_dims]) > 0
    end
end

@testset "Both solvers: starts and option checks" begin
    case = build_benchmark_problem(; num_players = 1, levels = 2, kind = :quadratic)
    k = RG.generate_slacked_reduced_kkt_system(case.problem)
    θ = zeros(k.parameter_dimension)
    for (solver, Options) in SOLVERS
        # A primal start: s, σ and γ at 1, every other variable at 0.
        z = RG.solve(solver, k, θ; z₀ = case.expected, options = Options(; max_inner_iters = 0)).z
        @test z[k.primal_dims] == case.expected
        ones_dims = [k.preference_slack_dims; k.interior_point_slack_dims; k.inequality_constraint_dual_dims]
        @test all(==(1.0), z[ones_dims])
        @test all(iszero, z[setdiff(1:(k.variable_dimension), [k.primal_dims; ones_dims])])
        # A start of neither length, or not positive on nonnegative_dims (σ and γ), is rejected.
        @test_throws ArgumentError RG.solve(solver, k, θ; z₀ = zeros(k.variable_dimension + 1))
        bad = copy(z)
        bad[first(k.nonnegative_dims)] = 0.0
        @test_throws ArgumentError RG.solve(solver, k, θ; z₀ = bad)
        @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; ρ_schedule = Float64[]))
        @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; β = 1.0))
        @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; min_step = 0.0))
        @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; η = 0.0))
        @test_throws ArgumentError RG.solve(solver, k, θ; options = Options(; linear_solver = :lu))
    end
    for options in (RG.PDIPOptions(; linesearch = :exact), RG.PDIPOptions(; linear_solver = :normal))
        @test_throws ArgumentError RG.solve(RG.PDIP(), k, θ; options)
    end
    for options in (RG.ProjectedLMOptions(; linear_solver = :svd), RG.ProjectedLMOptions(; η = Inf),
                    RG.ProjectedLMOptions(; η_min = 0.0), RG.ProjectedLMOptions(; η_min = 2.0, η_max = 1.0),
                    RG.ProjectedLMOptions(; gain_low = 0.8, gain_high = 0.5),
                    RG.ProjectedLMOptions(; gain_low = -0.1), RG.ProjectedLMOptions(; τ = 0.0),
                    RG.ProjectedLMOptions(; τ = 1.0))
        @test_throws ArgumentError RG.solve(RG.ProjectedLM(), k, θ; options)
    end
    # Algorithm 5.1 has no rule for the Scholtes system's exact products σ̂ ⊙ ϕ = 0.
    scholtes = RG.generate_slacked_scholtes_reduced_kkt_system(case.problem)
    @test_throws ArgumentError RG.solve(RG.PDIP(), scholtes, θ; z₀ = scholtes_start(scholtes, θ, case.expected, 1.0))
end

@testset "Both solvers: termination reasons" begin
    quadratic = RG.generate_slacked_reduced_kkt_system(free_problem(x -> x^2))
    # F = atan(x): from x = 2 the full Newton step overshoots, to |atan(x)| larger than at x = 2.
    overshoot = RG.generate_slacked_reduced_kkt_system(free_problem(x -> x * atan(x) - log(1 + x^2) / 2))
    for (solver, Options) in SOLVERS
        solve_from(k, z₀; kwargs...) =
            RG.solve(solver, k, Float64[]; z₀, options = Options(; ρ_schedule = [1e-3], kwargs...))
        r = solve_from(quadratic, [1.0]; tol = Inf)
        @test (r.status, r.termination_reason, r.residual_evaluations) == (:solved, :tolerance_reached, 1)
        r = solve_from(quadratic, [1.0]; tol = 0.0, tol_inner = Inf)
        @test (r.status, r.termination_reason, r.residual_evaluations) == (:failed, :inner_tolerance_reached, 1)
        r = solve_from(quadratic, [1.0]; max_inner_iters = 0, tol = 0.0, record_convergence = true)
        @test (r.termination_reason, r.residual_evaluations, r.newton_steps) == (:max_inner_iters, 0, 0)
        @test isempty(r.relaxed_residual_history)
        r = solve_from(quadratic, [1.0]; max_inner_iters = 1, tol = 0.0, tol_inner = 0.0, record_convergence = true)
        @test (r.termination_reason, r.newton_steps) == (:max_inner_iters, 1)
        @test r.relaxed_residual < only(r.relaxed_residual_history)
        # The returned iterate is judged again: a last step can meet tol after the last test before a step.
        r = solve_from(quadratic, [1.0]; max_inner_iters = 1, tol = 1e-6, tol_inner = 0.0)
        @test (r.status, r.termination_reason) == (:solved, :tolerance_reached)
        r = solve_from(quadratic, [NaN])
        @test (r.status, r.termination_reason) == (:failed, :nonfinite_residual)
        @test isnan(r.relaxed_residual)
        # With min_step = 0.9 the line search tries α = 1 only, and the overshooting step fails it.
        r = solve_from(overshoot, [2.0]; min_step = 0.9, tol = 0.0, tol_inner = 0.0)
        @test (r.status, r.termination_reason, r.newton_steps) == (:failed, :line_search_failed, 0)
        @test r.z == [2.0]
    end
end

@testset "Both solvers: the reported residuals are those of the returned point" begin
    goop = coupled_problem(d -> cosh(d) - 1)
    reduced = RG.generate_slacked_reduced_kkt_system(goop)
    complete = RG.generate_slacked_complete_kkt_system(goop; complementarity = :perturbed)
    scholtes = RG.generate_slacked_scholtes_reduced_kkt_system(goop)
    θ = Float64[]
    runs = [(RG.PDIP(), RG.PDIPOptions, reduced, zeros(4)), (RG.PDIP(), RG.PDIPOptions, complete, zeros(4)),
            (RG.ProjectedLM(), RG.ProjectedLMOptions, reduced, zeros(4)),
            (RG.ProjectedLM(), RG.ProjectedLMOptions, complete, zeros(4)),
            (RG.ProjectedLM(), RG.ProjectedLMOptions, scholtes, scholtes_start(scholtes, θ, zeros(4), 1.0))]
    for (solver, Options, k, z₀) in runs, tol in (1e-6, 0.0)
        r = RG.solve(solver, k, θ; z₀, options = Options(; ρ_schedule = [1.0, 1e-2, 1e-4], max_inner_iters = 30, tol))
        F = zeros(k.kkt_dimension)
        k.K₀!(F, r.z; θ)
        @test r.residual == norm(F)
        k.F!(F, r.z; θ, r.ρ)
        @test r.relaxed_residual == norm(F)
        @test r.x == r.z[k.primal_dims]
        @test r.status === (r.residual < tol ? :solved : :failed)
        @test minimum(r.z[k.nonnegative_dims]) > 0
    end
end

@testset "ProjectedLM: the η schedule along a solve" begin
    k = RG.generate_slacked_reduced_kkt_system(free_problem(x -> x^2))
    down = 1 - exp(-1.2)
    for linear_solver in (:normal, :klu, :ldl)
        options = RG.ProjectedLMOptions(; ρ_schedule = [1.0, 0.1], max_inner_iters = 1, tol = 0.0, tol_inner = 0.0,
            linear_solver, η = 0.1, record_convergence = true)
        r = RG.solve(RG.ProjectedLM(), k, Float64[]; z₀ = [1.0], options)
        # F = 2x is linear: every full step lands where the model says (gain ratio 1), and η shrinks.
        @test r.step_size_history == [1.0, 1.0]
        @test r.η_history ≈ [0.1, 0.1down]
        @test r.η ≈ 0.1down^2
        # A solve continues from the returned η.
        options = RG.ProjectedLMOptions(; ρ_schedule = [0.01], max_inner_iters = 1, tol = 0.0, tol_inner = 0.0,
            linear_solver, η = r.η, record_convergence = true)
        @test only(RG.solve(RG.ProjectedLM(), k, Float64[]; z₀ = r.z, options).η_history) == r.η
    end
end

end
