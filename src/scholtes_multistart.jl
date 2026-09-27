# scholtes_multistart.jl -- returning different solutions when the answer is not unique.
#
# Ported from ScholtesReducedGOOP.jl `src/multistart.jl`. A GOOP's answer is often a
# set; under the Scholtes relaxation the relaxed system's solution set contains it at
# every ρ, so different starts land in different places and a dedupe recovers the set.

"""
    is_feasible(goop, z, θ; tol = 1e-6) -> (; feasible, worst, per_player)

Worst (most negative) inequality component across all players at primal `z`; a player
with no inequalities reports `Inf`.
"""
function is_feasible(goop::ParametricGOOP, z, θ; tol = 1e-6)
    x = BlockArray(collect(z), goop.primal_dims)
    θ_blocks = BlockArray(collect(θ), goop.parameter_dims)
    per_player = [
        goop.inequality_dims[i] > 0 ? minimum(goop.inequality_constraints[i](x, θ_blocks)) :
        Inf for i in 1:(goop.num_players)
    ]
    worst = minimum(per_player)
    return (feasible = worst >= -tol, worst, per_player)
end

# One start, walked down the `retries` ladder until a solve is accepted (converged,
# inside the sign box, finite). `nothing` if every attempt failed.
function _solve_from(kkt, θ, z₀, tol, retries, options)
    for extra in retries
        r = try
            solve(Scholtes(), kkt, θ; z₀, options = _with_options(options; tol, extra...))
        catch err
            err isa ArgumentError || rethrow()
            continue
        end
        r.residual < tol && r.sign_min > 0 && all(isfinite, r.w) && return r
    end
    return nothing
end

# Fold each result into the first kept point within `tol` of it, in max-norm.
function _dedupe(results, tol)
    points, assignment = Vector{Vector{Float64}}(), Int[]
    for r in results
        idx = findfirst(p -> maximum(abs, p .- r.z) <= tol, points)
        isnothing(idx) && (push!(points, copy(r.z)); idx = length(points))
        push!(assignment, idx)
    end
    return points, assignment
end

"""
    solve_multi(kkt, θ, starts; tol = 1e-7, dedupe_tol = 1e-4,
                retries = [(;), (; rho_init = 1e1), (; rho_init = 1e3)],
                options = ScholtesOptions()) -> NamedTuple

Solve from every start, keep the solves that converge inside the sign box, and
deduplicate the primal points. `retries` are option overrides tried in order. Returns
`(; points, results, starts, assignment, rejected)`.
"""
function solve_multi(
    kkt::ScholtesKKTSystem,
    θ,
    starts::AbstractVector;
    tol = 1e-7,
    dedupe_tol = 1e-4,
    retries = [(;), (; rho_init = 1e1), (; rho_init = 1e3)],
    options::ScholtesOptions = ScholtesOptions(),
)
    results, kept, rejected = Any[], Vector{Float64}[], 0
    for z₀ in starts
        z = Float64.(collect(z₀))
        r = _solve_from(kkt, θ, z, tol, retries, options)
        if isnothing(r)
            rejected += 1
        else
            push!(results, r)
            push!(kept, z)
        end
    end
    points, assignment = _dedupe(results, dedupe_tol)
    return (; points, results, starts = kept, assignment, rejected)
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
