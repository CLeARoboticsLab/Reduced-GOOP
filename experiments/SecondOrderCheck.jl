"""
SecondOrderCheck — reusable second-order optimality diagnostics for candidate
equilibria of equality-constrained GOOPs solved via the reduced (or quasi)
formulation.

For each player i, the reduced reformulation's top level is the optimization
problem

	min_{w}  f_top^i(x, θ)   s.t.  c(w) = [ fᵢ(x, θ); πᵢ(x, dᵢ) ] = 0,

with decision variables w = (xᵢ, dᵢ), where dᵢ are that player's lower-level
duals (λ of levels 2..K and ψ of levels 2..K−1), fᵢ the private equality
constraints, and πᵢ the stacked lower-level stationarity conditions. The
top-level Lagrangian is L₁ = f_top − ψᵢ₁ᵀπᵢ − λᵢ₁ᵀfᵢ with the top-level
multipliers taken from the candidate solution. Two diagnostics are evaluated
at a candidate KKT point z*:

  * Full-space test (quick, sufficient only): eigenvalues of H = ∇²_ww L₁.
	If H is PSD the second-order condition holds on the critical cone. For
	players whose lower levels contain constraints that are nonlinear in x
	this test is structurally indefinite, hence always inconclusive.
  * Projected test (decisive): eigenvalues of ZᵀHZ, where Z is an orthonormal
	basis of null(∂c/∂w) — the tangent space of the top-level feasible set,
	which equals the critical cone because the problem is purely
	equality-constrained.

# Usage

Generic (any scenario whose `get_setup` returns an equality-only
`ParametricGOOP`, e.g. `Robotic_arm`, `Intersection`):

```julia
include("experiments/SecondOrderCheck.jl")

checker = SecondOrderCheck.build_checker(problem)   # one-time symbolic build
report  = SecondOrderCheck.check(checker, z, θ)     # fast; reusable across candidates
SecondOrderCheck.print_report(report)
```

Robotic arm (rebuilds the exact `Robotic_arm.demo()` problem):

```julia
SecondOrderCheck.check_robotic_arm_demo(
	"data/robotic_arm_open_loop/runs/<run_id>/data/problem/solution/solution_dict_instance_1_eps0.1.jld2",
)
```

Intersection follows the generic pattern:

```julia
(; problem, flatten_parameters) = Intersection.get_setup(2; planning_horizon = ...)
θ = ...  # via flatten_parameters, as in Intersection.demo
SecondOrderCheck.check_solution(problem, solution_dict["z"], θ)
```

# Assumptions

  * The candidate `z` uses the variable layout of
	`generate_slacked_reduced_kkt_system` / `generate_slacked_quasi_kkt_system`
	(z = [x; Λ; Ψ]); solutions of the `complete` formulation are not supported.
  * Purely equality-constrained: no private or shared inequality constraints
	(prioritized-constraint preference levels entering as smooth penalties are
	fine — they are part of the objectives). With no inequalities the critical
	cone is the tangent space null(∂c/∂w).
  * Cost-type preference levels are scalar-valued.
  * z* is an approximate critical point; curvature smaller than ‖F(z*)‖ cannot
	be certified either way, so ‖F(z*)‖ is the default curvature tolerance floor.
"""
module SecondOrderCheck

using LinearAlgebra: LinearAlgebra, Symmetric, eigvals, hermitianpart!, norm, svd
using SparseArrays: SparseArrays, SparseMatrixCSC, findnz, nnz
using BlockArrays: Block
using JLD2: JLD2
using ReducedGOOP: ReducedGOOP

const STU = ReducedGOOP.SymbolicTracingUtils
const Symbolics = ReducedGOOP.Symbolics

# ── Checker types ──────────────────────────────────────────────────────────────

struct PlayerChecker{TG,TC}
    player::Int
    num_levels::Int
    primal_dim::Int
    dual_dim::Int
    constraint_dim::Int
    "In-place `(buffer, z, θ)`: fills ∂[∇L₁; f; π]/∂w of size (primal_dim + constraint_dim) × (primal_dim + dual_dim)."
    G!::TG
    G_buffer::SparseMatrixCSC{Float64,Int}
    "In-place `(buffer, z, θ)`: fills C = ∇²_dd L₁; `nothing` when structurally zero."
    C!::TC
    C_buffer::Union{SparseMatrixCSC{Float64,Int},Nothing}
    "Preallocated dense scratch buffers."
    G_dense::Matrix{Float64}
    H_dense::Matrix{Float64}
end

struct Checker{TP,TF}
    problem::TP
    players::Vector{PlayerChecker}
    "In-place `(buffer, z, θ)`: fills the full KKT residual at η = 0 (row order permuted w.r.t. the solver's, same norm)."
    residual!::TF
    residual_buffer::Vector{Float64}
    variable_dimension::Int
end

# ── Symbolic construction ──────────────────────────────────────────────────────

"Rebuild player `player`'s Lagrangian hierarchy of the reduced reformulation (equality-only path)."
function _build_player_hierarchy(problem, x_blocked, θ_blocked, backend, player)
    preferences = problem.preferences[player]
    is_constraint = problem.is_prioritized_constraint[player]
    num_levels = length(preferences)
    symbolic_type = eltype(collect(x_blocked))

    f =
        isnothing(problem.equality_constraints[player]) ? symbolic_type[] :
        problem.equality_constraints[player](x_blocked, θ_blocked)
    λs = [
        STU.make_variables(backend, Symbol("λ_$(player)_$(k)"), length(f)) for
        k in 1:num_levels
    ]
    x = collect(x_blocked[Block(player)])

    preference_term = function (level)
        h = preferences[level](x_blocked, θ_blocked)
        if is_constraint[level]
            sum(ReducedGOOP.smooth_piecewise_preference_objective.(h, level))
        else
            h isa AbstractVector ? only(h) : h
        end
    end
    dual_term(k) = isempty(f) ? 0 : λs[k]' * f

    # Innermost (base) level, then unwind outward to the top level.
    L = preference_term(num_levels) - dual_term(num_levels)
    π = num_levels > 1 ? Symbolics.gradient(L, x) : symbolic_type[]
    ψs = Vector{Vector{symbolic_type}}(undef, max(num_levels - 1, 0))
    for k in (num_levels - 1):-1:1
        ψs[k] = STU.make_variables(backend, Symbol("ψ_$(player)_$(k)"), length(π))
        L = preference_term(k) - ψs[k]' * π - dual_term(k)
        k > 1 && (π = vcat(Symbolics.gradient(L, x), π))
    end
    ∇L_top = Symbolics.gradient(L, x)

    (; L_top = L, ∇L_top, λs, ψs, π, f, x, num_levels)
end

"Compile a sparse symbolic matrix into an in-place function plus a matching numeric buffer."
function _compile_sparse(M_symbolic, z_symbolic, θ_symbolic)
    fn! = STU.build_function(M_symbolic, z_symbolic, θ_symbolic; in_place = true)
    buffer = SparseMatrixCSC(
        size(M_symbolic)...,
        copy(M_symbolic.colptr),
        copy(M_symbolic.rowval),
        zeros(nnz(M_symbolic)),
    )
    fn!, buffer
end

"""
	build_checker(problem::ReducedGOOP.ParametricGOOP; backend = SymbolicsBackend())

One-time symbolic construction of the second-order checker for an equality-only
GOOP. The returned `Checker` holds compiled in-place derivative functions and
preallocated buffers, so `check` can be called repeatedly (e.g. per ϵ-stage or
per receding-horizon step) at purely numeric cost.
"""
function build_checker(
    problem::ReducedGOOP.ParametricGOOP;
    backend = STU.SymbolicsBackend(),
)
    all(iszero, problem.inequality_dims) || error(
        "SecondOrderCheck supports purely equality-constrained GOOPs; found private inequality constraints. " *
        "Inequalities introduce interior-point slacks/duals (σ, γ) whose layout and active-set handling are not implemented.",
    )
    problem.shared_equality_dims == 0 && problem.shared_inequality_dims == 0 || error(
        "SecondOrderCheck supports purely equality-constrained GOOPs; found shared constraints.",
    )

    x_blocked =
        STU.make_variables(backend, :x, sum(problem.primal_dims)) |>
        ReducedGOOP.to_blockvector(problem.primal_dims)
    θ_blocked =
        STU.make_variables(backend, :θ, sum(problem.parameter_dims)) |>
        ReducedGOOP.to_blockvector(problem.parameter_dims)
    symbolic_type = eltype(collect(x_blocked))

    hierarchies = [
        _build_player_hierarchy(problem, x_blocked, θ_blocked, backend, player) for
        player in 1:problem.num_players
    ]

    # Variable layout of generate_slacked_reduced_kkt_system with no slack /
    # inequality-dual / shared blocks: z = [x; Λ; Ψ], Λ grouped per player with
    # levels ascending, Ψ grouped per player with levels descending (creation order).
    z_symbolic = vcat(
        collect(x_blocked),
        reduce(
            vcat,
            [h.λs[k] for h in hierarchies for k in eachindex(h.λs)];
            init = symbolic_type[],
        ),
        reduce(
            vcat,
            [h.ψs[k] for h in hierarchies for k in reverse(eachindex(h.ψs))];
            init = symbolic_type[],
        ),
    )
    θ_symbolic = collect(θ_blocked)

    # Residual for candidate validation (row order is irrelevant for the norm).
    F_symbolic = reduce(vcat, [vcat(h.∇L_top, h.π, h.f) for h in hierarchies])
    residual! = STU.build_function(F_symbolic, z_symbolic, θ_symbolic; in_place = true)

    players = map(enumerate(hierarchies)) do (player, h)
        duals = reduce(vcat, [h.λs[2:end]; h.ψs[2:end]]; init = symbolic_type[])
        w = vcat(h.x, duals)
        primal_dim, dual_dim = length(h.x), length(duals)
        constraint_dim = length(h.f) + length(h.π)

        # Rows 1:primal_dim give [∇²_xx L₁ | ∇²_x,dual L₁]; the rest is ∂c/∂w.
        G_symbolic = STU.sparse_jacobian(vcat(h.∇L_top, h.f, h.π), w)
        G!, G_buffer = _compile_sparse(G_symbolic, z_symbolic, θ_symbolic)

        # Dual–dual Hessian block: differentiation w.r.t. duals only. (Never
        # differentiate the scalar L₁ w.r.t. x symbolically — expression swell.)
        C!, C_buffer = if dual_dim > 0
            C_symbolic = STU.sparse_jacobian(Symbolics.gradient(h.L_top, duals), duals)
            nnz(C_symbolic) > 0 ?
            _compile_sparse(C_symbolic, z_symbolic, θ_symbolic) :
            (nothing, nothing)
        else
            (nothing, nothing)
        end

        decision_dim = primal_dim + dual_dim
        PlayerChecker(
            player,
            h.num_levels,
            primal_dim,
            dual_dim,
            constraint_dim,
            G!,
            G_buffer,
            C!,
            C_buffer,
            zeros(primal_dim + constraint_dim, decision_dim),
            zeros(decision_dim, decision_dim),
        )
    end

    Checker(problem, players, residual!, zeros(length(F_symbolic)), length(z_symbolic))
end

# ── Numeric evaluation ─────────────────────────────────────────────────────────

"""
	check(checker, z, θ; rank_rtol = 1e-8, curvature_rtol = 1e-8, curvature_floor = nothing)

Evaluate both second-order diagnostics at candidate solution `z` (layout of the
reduced KKT system) with parameters `θ`. Returns a report (NamedTuple); render
it with [`print_report`](@ref).

Tolerances: constraint-Jacobian rank is decided at `rank_rtol * σ_max`;
eigenvalues are called negative below `-max(curvature_rtol * |λ|_max,
curvature_floor)`, where `curvature_floor` defaults to the KKT residual norm
‖F(z)‖ — curvature smaller than the residual of the approximate critical point
cannot be certified either way.
"""
function check(
    checker::Checker,
    z::AbstractVector{<:Real},
    θ::AbstractVector{<:Real};
    rank_rtol = 1e-8,
    curvature_rtol = 1e-8,
    curvature_floor = nothing,
)
    length(z) == checker.variable_dimension || error(
        "candidate has $(length(z)) variables, expected $(checker.variable_dimension); " *
        "is it a solution of the reduced/quasi KKT system of this problem?",
    )
    checker.residual!(checker.residual_buffer, z, θ)
    residual_norm = norm(checker.residual_buffer)
    floor_value = something(curvature_floor, residual_norm)

    players = map(checker.players) do pc
        n_x, n_d = pc.primal_dim, pc.dual_dim
        n_w = n_x + n_d

        pc.G!(pc.G_buffer, z, θ)
        G = pc.G_dense
        copyto!(G, pc.G_buffer)

        # Assemble H = [H_xx B; Bᵀ C].
        H = pc.H_dense
        fill!(H, 0.0)
        @views H[1:n_x, :] .= G[1:n_x, :]
        @views H[(n_x + 1):n_w, 1:n_x] .= G[1:n_x, (n_x + 1):n_w]'
        if pc.C! !== nothing
            pc.C!(pc.C_buffer, z, θ)
            @views H[(n_x + 1):n_w, (n_x + 1):n_w] .+= pc.C_buffer
        end
        block_norms = (;
            H_xx = norm(@view H[1:n_x, 1:n_x]),
            H_x_dual = norm(@view H[1:n_x, (n_x + 1):n_w]),
            H_dual_dual = norm(@view H[(n_x + 1):n_w, (n_x + 1):n_w]),
        )
        asymmetry = norm(H - H') / max(norm(H), eps())
        H_sym = hermitianpart!(H)

        λ = eigvals(H_sym)                      # ascending
        λ_scale = max(abs(λ[1]), abs(λ[end]), eps())
        tol_full = max(curvature_rtol * λ_scale, floor_value)
        full = (;
            λ_min = λ[1],
            λ_max = λ[end],
            n_negative = count(<(-tol_full), λ),
            tol = tol_full,
            pass = λ[1] ≥ -tol_full,
        )

        # Tangent space of c(w) = [f; π] = 0 and projected Hessian.
        A = @view G[(n_x + 1):end, :]
        decomposition = svd(Matrix(A); full = true)
        σ = decomposition.S
        σ_max = isempty(σ) ? 0.0 : σ[1]
        rank_tol = rank_rtol * σ_max
        rank = count(>(rank_tol), σ)
        null_dim = n_w - rank
        jacobian = (;
            σ_max,
            smallest_σ = sort(σ)[1:min(8, length(σ))],
            rank,
            rank_tol,
            null_dim,
            licq = rank == pc.constraint_dim,
        )

        projected = if null_dim == 0
            nothing
        else
            Z = decomposition.V[:, (rank + 1):n_w]
            λp = eigvals(Symmetric(Z' * H_sym * Z))
            λp_scale = max(abs(λp[1]), abs(λp[end]), eps())
            tol_projected = max(curvature_rtol * λ_scale, floor_value)
            (;
                λ_min = λp[1],
                λ_max = λp[end],
                n_negative = count(<(-tol_projected), λp),
                tol = tol_projected,
                pass = λp[1] ≥ -tol_projected,
                strict = λp[1] > tol_projected,
                flat = λp_scale ≤ tol_projected,
                smallest_λ = λp[1:min(8, length(λp))],
            )
        end

        (;
            player = pc.player,
            num_levels = pc.num_levels,
            primal_dim = n_x,
            dual_dim = n_d,
            decision_dim = n_w,
            constraint_dim = pc.constraint_dim,
            asymmetry,
            block_norms,
            full,
            jacobian,
            projected,
            pass = isnothing(projected) ? true : projected.pass,
        )
    end

    (;
        residual_norm,
        curvature_floor = floor_value,
        rank_rtol,
        curvature_rtol,
        variable_dimension = checker.variable_dimension,
        players,
        pass = all(p -> p.pass, players),
    )
end

"""
	check_solution(problem, z, θ; kwargs...)

One-shot convenience: build the checker, run [`check`](@ref), print the report.
Returns `(; report, checker)` so the checker can be reused for further candidates.
"""
function check_solution(problem::ReducedGOOP.ParametricGOOP, z, θ; kwargs...)
    checker = build_checker(problem)
    report = check(checker, z, θ; kwargs...)
    print_report(report)
    (; report, checker)
end

# ── Reporting ──────────────────────────────────────────────────────────────────

_round(x; digits = 4) = round(x; sigdigits = digits)

"Render a `check` report to `io` (default `stdout`)."
function print_report(report; io = stdout)
    println(
        io,
        "════ Second-order optimality check (equality-constrained reduced GOOP) ════",
    )
    println(
        io,
        "KKT residual ‖F(z)‖₂ = ",
        report.residual_norm,
        "  → curvature tolerance floor = ",
        _round(report.curvature_floor),
    )
    println(
        io,
        "tolerances: rank_rtol = ",
        report.rank_rtol,
        ", curvature_rtol = ",
        report.curvature_rtol,
    )

    for p in report.players
        println(
            io,
            "\n── Player ",
            p.player,
            " (",
            p.num_levels,
            " preference levels) ──",
        )
        println(
            io,
            "decision variables w = (x, lower duals): ",
            p.decision_dim,
            " (x: ",
            p.primal_dim,
            ", duals: ",
            p.dual_dim,
            "),  constraints c = [f; π]: ",
            p.constraint_dim,
        )
        println(
            io,
            "H = ∇²_ww L₁ block norms: ‖H_xx‖ = ",
            _round(p.block_norms.H_xx),
            ", ‖B‖ = ",
            _round(p.block_norms.H_x_dual),
            ", ‖C‖ = ",
            _round(p.block_norms.H_dual_dual),
            "  (asymmetry ",
            _round(p.asymmetry; digits = 2),
            ")",
        )

        f = p.full
        println(
            io,
            "[full-space test]  λ ∈ [",
            _round(f.λ_min),
            ", ",
            _round(f.λ_max),
            "],  ",
            f.n_negative,
            " negative below tol ",
            _round(f.tol),
            "  → ",
            f.pass ? "PASS (PSD ⇒ SOC hold on critical cone)" :
            "not PSD → inconclusive, see projected test",
        )

        j = p.jacobian
        println(
            io,
            "[constraint Jacobian]  rank ",
            j.rank,
            " of ",
            size_str(p),
            " at tol ",
            _round(j.rank_tol),
            ",  tangent-space dim = ",
            j.null_dim,
            j.licq ? "" :
            "  (rank-deficient: LICQ fails, $(p.constraint_dim - j.rank) dependent rows)",
        )
        println(
            io,
            "                       smallest σ: ",
            _round.(j.smallest_σ; digits = 3),
        )

        if isnothing(p.projected)
            println(
                io,
                "[projected test]   tangent space is trivial → second-order condition holds vacuously",
            )
        else
            q = p.projected
            println(
                io,
                "[projected test]   λ(ZᵀHZ) ∈ [",
                _round(q.λ_min),
                ", ",
                _round(q.λ_max),
                "],  ",
                q.n_negative,
                " negative below tol ",
                _round(q.tol),
            )
            verdict =
                q.pass ?
                (
                    q.strict ?
                    "PASS (strict: positive definite on tangent space ⇒ strict local optimum)" :
                    q.flat ?
                    "PASS, DEGENERATE (≈ zero curvature on the whole tangent space — no strict optimum certificate)" :
                    "PASS (second-order necessary conditions hold within tolerance)"
                ) : "FAIL (negative curvature on the tangent space exceeds tolerance)"
            println(io, "                   → ", verdict)
        end
    end
    println(
        io,
        "\noverall: ",
        report.pass ? "PASS" : "FAIL",
        "  (curvature below the KKT residual level is not certifiable either way)",
    )
    nothing
end

size_str(p) = string(p.constraint_dim, "×", p.decision_dim)

# ── Scenario adapters ──────────────────────────────────────────────────────────

# Referenced only from functions entered via `invokelatest`, so the binding
# access happens in the latest world even right after the `include`.
function _robotic_arm_module()
    Main.Robotic_arm
end

"""
	check_robotic_arm_demo(solution_path; instance_states = nothing, checker = nothing, kwargs...)

Run the second-order check on a saved `solution_dict_*.jld2` produced by
`Robotic_arm.demo()`. Rebuilds the identical problem via
`Robotic_arm.demo_scenario_config()`. For runs with `random_initial_state = true`
pass `instance_states = (; initial_state1, initial_state2, initial_state3)`
from the run's `problem_data_instance_*.jld2`. Pass a previously returned
`checker` to skip the symbolic build. Returns `(; report, checker)`.
"""
function check_robotic_arm_demo(
    solution_path::AbstractString;
    instance_states = nothing,
    checker = nothing,
    kwargs...,
)
    # Robotic_arm.jl may be `include`d here at runtime, so its methods (and the
    # closures captured in `problem`) live in a newer world age; run the whole
    # body in the latest world.
    isdefined(Main, :Robotic_arm) ||
        Base.include(Main, joinpath(@__DIR__, "Robotic_arm.jl"))
    Base.invokelatest(
        _check_robotic_arm_demo,
        solution_path,
        instance_states,
        checker;
        kwargs...,
    )
end

function _check_robotic_arm_demo(solution_path, instance_states, checker; kwargs...)
    RA = _robotic_arm_module()
    scenario_config = RA.demo_scenario_config()
    (; problem, flatten_parameters) = RA.get_setup(scenario_config)
    states = something(
        instance_states,
        (;
            initial_state1 = scenario_config.base_initial_state1,
            initial_state2 = scenario_config.base_initial_state2,
            initial_state3 = scenario_config.initial_state3,
        ),
    )
    θ = RA.build_instance_parameters(flatten_parameters, states, scenario_config).θ
    z = JLD2.load_object(solution_path)["z"]

    checker = something(checker, build_checker(problem))
    report = check(checker, z, θ; kwargs...)
    print_report(report)
    (; report, checker)
end

end # module SecondOrderCheck
