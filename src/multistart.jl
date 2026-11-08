# multistart.jl -- solving one KKT system from many starts: the different points of a GOOP's answer when it
# is a set, or a study of the local convergence around one solution.
#
# After ScholtesReducedGOOP.jl's `src/multistart.jl` (Jingqi Li). A GOOP's answer is often a set, and
# different starts land in different places; merging the primal points that lie within a tolerance of each
# other recovers the set.

"""
    solve_multi(solver, kkt, θ, starts; options, accept = r -> r.status === :solved, dedupe_tol = 1e-4)
        -> (; points, results, starts, assignment, rejected)

`solve(solver, kkt, θ; z₀, options)` from every start, for `solver` `PDIP()` or `ProjectedLM()` (`options` of
its type, by default its defaults). A start is a full iterate (length `kkt.variable_dimension`), used as given,
or a primal point (length of `kkt.primal_dims`): the Scholtes system is then started by `initialize_scholtes!`
(its cold start, at the first ρ of `options.ρ_schedule`), every other system by the solver's own start for
the coordinates a primal point leaves out.

The results that `accept` passes are kept, with the starts they came from (`starts`); `rejected` counts the
others, and the starts that could not be used (an `ArgumentError` from the start or the solve: a Scholtes
start with an inequality not strictly satisfied, or a full start not positive on `nonnegative_dims`). A start
of neither length is an error. `accept` is a keyword because a solve
held at one ρ > 0 never meets the ‖K₀‖ test that sets `status`: there, `r.termination_reason ===
:inner_tolerance_reached` is the test. `points` are the kept results' primal points `x`, each merged into the
first earlier one within `dedupe_tol` in max-norm; `results[j]` was merged into `points[assignment[j]]`.
"""
function solve_multi(
    solver::SolverType,
    kkt::GOOPKKTSystem,
    θ,
    starts::AbstractVector;
    options = _default_options(solver),
    accept = r -> r.status === :solved,
    dedupe_tol = 1e-4,
)
    results, kept, rejected = Any[], Vector{Float64}[], 0
    for start in starts
        start = Float64.(collect(start))
        length(start) in (kkt.variable_dimension, length(kkt.primal_dims)) || throw(
            ArgumentError(
                "a start must have length $(kkt.variable_dimension) (full) or $(length(kkt.primal_dims)) (primal); got $(length(start)).",
            ),
        )
        result = try
            solve(solver, kkt, θ; z₀ = _multistart_z₀(kkt, θ, start, options), options)
        catch err
            err isa ArgumentError || rethrow()
            nothing
        end
        if !isnothing(result) && accept(result)
            push!(results, result)
            push!(kept, start)
        else
            rejected += 1
        end
    end
    points, assignment = _dedupe([r.x for r in results], dedupe_tol)
    (; points, results, starts = kept, assignment, rejected)
end

"The options `solve_multi` uses when none are given: the solver's defaults."
_default_options(::PDIP) = PDIPOptions()
_default_options(::ProjectedLM) = ProjectedLMOptions()

"""
The `z₀` of one start of `solve_multi`: a full iterate as given, a primal point as given (the solver fills the
other coordinates), or for the Scholtes system a primal point's cold start by `initialize_scholtes!`.
"""
function _multistart_z₀(kkt, θ, start, options)
    length(start) == kkt.variable_dimension && return start
    kkt.kind === :scholtes || return start
    initialize_scholtes!(zeros(kkt.variable_dimension), kkt, θ, start, first(options.ρ_schedule); warm = false)
end

"Merge each point into the first kept point within `tol` of it in max-norm: `(points, assignment)`."
function _dedupe(points, tol)
    kept, assignment = Vector{Float64}[], Int[]
    for p in points
        index = findfirst(q -> maximum(abs, q .- p) <= tol, kept)
        isnothing(index) && (push!(kept, copy(p)); index = length(kept))
        push!(assignment, index)
    end
    kept, assignment
end

"""
    ball_starts(kkt, center, m; radius = 1.0, mode = :primal_only, rng = Random.default_rng(),
                max_tries = 1000) -> Vector{Vector{Float64}}

`m` full iterates drawn uniformly from the ball of `radius` around `center`, a full iterate (typically a
solution), for `solve_multi`:

  - `mode = :primal_only`: the ball in the primal coordinates `kkt.primal_dims`; every other coordinate keeps
    `center`'s value;
  - `mode = :whole_iterate`: the ball in all of z; a draw that leaves a coordinate of `kkt.nonnegative_dims` at
    or below 0 is rejected (the solvers need them positive), and after `max_tries · m` draws it is an error.

A draw is `center` plus a Gaussian direction scaled to the length `radius · U^(1/d)`, with U uniform on [0, 1]
and d the ball's dimension: uniform in the ball, so in high dimension most draws lie close to its surface. Two
systems with the same number of primal coordinates get the same `:primal_only` offsets from the same `rng`
state.
"""
function ball_starts(
    kkt::GOOPKKTSystem,
    center,
    m::Int;
    radius = 1.0,
    mode::Symbol = :primal_only,
    rng = Random.default_rng(),
    max_tries = 1000,
)
    mode in (:primal_only, :whole_iterate) ||
        throw(ArgumentError("mode must be :primal_only or :whole_iterate, got $mode"))
    length(center) == kkt.variable_dimension || throw(
        ArgumentError("center must be a full iterate of length $(kkt.variable_dimension); got $(length(center))."),
    )
    radius > 0 || throw(ArgumentError("radius must be positive, got $radius"))
    c = Float64.(collect(center))
    all(>(0), view(c, kkt.nonnegative_dims)) ||
        throw(ArgumentError("center must be positive on nonnegative_dims."))
    dims = mode === :primal_only ? collect(kkt.primal_dims) : collect(1:(kkt.variable_dimension))
    d = length(dims)
    starts = Vector{Vector{Float64}}()
    draws = 0
    while length(starts) < m
        draws += 1
        draws > max_tries * m && throw(
            ArgumentError(
                "could not draw $m starts positive on nonnegative_dims in $(max_tries * m) draws; use a smaller radius or mode = :primal_only.",
            ),
        )
        direction = randn(rng, d)
        z = copy(c)
        z[dims] .+= direction .* (radius * rand(rng)^(1 / d) / norm(direction))
        all(>(0), view(z, kkt.nonnegative_dims)) && push!(starts, z)
    end
    starts
end

"""
    random_starts(goop, θ, m; radius = 1.0, center, rng = Random.default_rng(),
                  max_tries = 100) -> Vector{Vector{Float64}}

`m` strictly feasible primal starts drawn uniformly from the box of half-width
`radius` around `center` (default zeros), rejected unless every `g(z, θ) > 0`.
"""
function random_starts(
    goop::ParametricGOOP,
    θ,
    m::Int;
    radius = 1.0,
    center = zeros(sum(goop.primal_dims)),
    rng = Random.default_rng(),
    max_tries = 100,
)
    n = sum(goop.primal_dims)
    c = Float64.(collect(center))
    out = Vector{Vector{Float64}}()
    tries = 0
    while length(out) < m
        tries += 1
        tries > max_tries * m && throw(
            ArgumentError(
                "could not draw $m strictly feasible starts in $(max_tries * m) tries; is `center` interior?",
            ),
        )
        z = c .+ radius .* (2 .* rand(rng, n) .- 1)
        is_feasible(goop, z, θ).worst > 0 && push!(out, z)
    end
    return out
end

"""
    grid_starts(goop, θ, values...) -> Vector{Vector{Float64}}

The strictly feasible points of the grid `values[1] × … × values[n]`, one range per
primal coordinate.
"""
function grid_starts(goop::ParametricGOOP, θ, values...)
    n = sum(goop.primal_dims)
    length(values) == n ||
        throw(ArgumentError("grid_starts needs one range per primal coordinate ($n)"))
    points = [collect(Float64, t) for t in Iterators.product(values...)]
    return [z for z in vec(points) if is_feasible(goop, z, θ).worst > 0]
end
