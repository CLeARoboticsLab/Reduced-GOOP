# scholtes_kkt.jl -- the reduced KKT system in Scholtes form.
#
# Ported from ScholtesReducedGOOP.jl (Jingqi Li), `src/kkt.jl` and `src/compile.jl`,
# on top of `ParametricGOOP` and this package's code generator. Only the explicit
# formulation (`phi = true`) is supported: every player keeps its original
# `γ_k ≥ 0` on `g` at every level, and each upper level k adds, for each of the
# K−k levels m below it,
#
#     −φ_{k,m}ᵀ (ρ𝟙 − G_m ⊙ γ_m),        φ_{k,m} ≥ 0,
#
# whose slack σ_{k,m} = ρ𝟙 − G_m ⊙ γ_m and φ_{k,m} satisfy EXACT
# complementarity σ ⊙ φ = 0. The level inequality pairs are Scholtes-relaxed by the
# solver: s ⊙ γ + u = ρ𝟙. The system is returned as three blocks,
#
# A prioritized constraint h_m >= 0 instead introduces e_m, minimizes sum(e_m),
# and adds h_m + e_m >= 0, e_m >= 0. These variables and inequalities persist in
# every upper level k <= m, with a fresh inequality multiplier at each level.
# Let G_k collect g and the preference inequalities available at level k. Then
#
#     F_nc(y)  stationarity w.r.t. own x and the current level's e_k, plus equalities
#     a(y)     G_1 ... G_K, then ρ𝟙 − G_m ⊙ γ_m for every φ pair
#     b(y)     γ_1 ... γ_K, then φ                     (a coordinate projection)
#
# with y = [x; preference slacks; duals]. The slacks s and the relaxation u are solver-side blocks, as
# in the source (src/scholtes_residual.jl). ρ enters the generated code in the slot
# the interior-point system uses for ϵ.

# --- Lagrangian terms, tagged with derivative order ---------------------------------
#
# A level's Lagrangian is carried as a list of additive terms so that quasi
# truncation has somewhere to record how often each piece has been differentiated.
# With no truncation, summing the terms' gradients is the gradient of the sum.

"""
    _LTerm(expr; duals = nothing, order = 0)

One additive piece of a preference level's Lagrangian: a scalar (an objective) or a
vector dotted against `duals` (a constraint, a φ slack, or a chained lower-level
stationarity residual), entering as `-dualsᵀexpr`. `order` counts how many times the
piece has been differentiated, for quasi truncation.
"""
struct _LTerm
    expr::Union{AbstractVector{Symbolics.Num},Symbolics.Num}
    duals::Union{AbstractVector{Symbolics.Num},Nothing}
    order::Int
end
_LTerm(expr; duals = nothing, order = 0) = _LTerm(expr, duals, order)

_sdot(a, b) = isempty(a) ? 0 : sum(a .* b)
_num(x) = x isa Symbolics.Num ? x : Symbolics.Num(x)

# Differentiate one term w.r.t. `x`; a term already differentiated `trunc` times
# returns an exact zero (quasi-GOOP's only approximation) and keeps `order + 1`.
function _diff(t::_LTerm, x::AbstractVector{Symbolics.Num}, trunc::Int)
    t.order >= trunc && return _LTerm(zero.(x), t.duals, t.order + 1)
    scalar = isnothing(t.duals) ? t.expr : -_sdot(t.duals, t.expr)
    return _LTerm(_gradient(scalar, x), t.duals, t.order + 1)
end

const _SU = Symbolics.SymbolicUtils

"""
    _gradient(O, x) -> Vector{Symbolics.Num}

`Symbolics.gradient(O, x)`, expression for expression, without walking all of `O` once
per variable: `executediff` is called only on the terms of a sum that can contain
`x[j]` (found once from each term's variables). Anything that is not a plain sum, or an
`x` with repeated entries, goes to `Symbolics.gradient` itself.
"""
function _gradient(O, x::AbstractVector{Symbolics.Num})
    u = Symbolics.unwrap(O)
    (u isa _SU.BasicSymbolic && _SU.isadd(u)) || return Symbolics.gradient(O, x)
    col = Dict(Symbolics.unwrap(v) => j for (j, v) in enumerate(x))
    length(col) == length(x) || return Symbolics.gradient(O, x)

    terms = _SU.arguments(u)
    candidates = [Int[] for _ in x]
    for (t, term) in enumerate(terms)
        every = false
        for v in Symbolics.get_variables(term)
            j = get(col, v, 0)
            j > 0 ? push!(candidates[j], t) : (every |= !_distinct_leaf(v))
        end
        every && foreach(c -> (isempty(c) || last(c) != t) && push!(c, t), candidates)
    end

    VT = Symbolics.VartypeT
    return map(eachindex(x)) do j
        D = Symbolics.Differential(x[j])
        summed = _SU.ArgsT{VT}()
        for t in candidates[j]
            t2 = Symbolics.executediff(D, terms[t])
            Symbolics._iszero(t2) && continue
            push!(summed, t2)
        end
        !isempty(summed) && return Symbolics.Num(_SU.add_worker(VT, summed))
        any(t -> Symbolics.occursin_info(D.x, terms[t]), candidates[j]) ?
        Symbolics.Num(_SU.add_worker(VT, summed)) : Symbolics.Num(Symbolics.COMMON_ZERO)
    end
end

# A leaf that provably is not any `x[j]` when it is not one of them: a scalar symbol,
# or an element indexed by constants.
function _distinct_leaf(v)
    _SU.issym(v) && return !_SU.is_array_shape(_SU.shape(v))
    _SU.iscall(v) && _SU.operation(v) === getindex || return false
    return all(i -> i isa Integer || _SU.isconst(i), @view _SU.arguments(v)[2:end])
end

# Differentiate every term, sum into one stationarity row, and return the
# differentiated terms: they are the ψ-seed the next (outer) level chains.
function _diff_sum(terms::Vector{_LTerm}, x::AbstractVector{Symbolics.Num}, trunc::Int)
    next = map(t -> _diff(t, x, trunc), terms)
    total = reduce((a, t) -> a .+ t.expr, next; init = zero.(x))
    return total, next
end

# Level k carries ψ_kᵀπ_{k+1}, π_{k+1} the stacked stationarity rows of levels
# k+1..K (`groups`, innermost last). Partition ψ across them and re-pair each lower
# term with its own slice.
function _couple_policy_terms(groups::Vector{Vector{_LTerm}}, ψ::AbstractVector{Symbolics.Num})
    coupled = _LTerm[]
    offset = 0
    for group in groups
        isempty(group) && continue
        block = view(ψ, (offset + 1):(offset + length(group[1].expr)))
        append!(coupled, (_LTerm(t.expr; duals = block, order = t.order) for t in group))
        offset += length(group[1].expr)
    end
    @assert offset == length(ψ) "policy-multiplier length mismatch"
    return coupled
end

# A prioritized level minimizes the sum of its own explicit preference slacks.
# Its vector h enters h + preference_slack >= 0 below, not a smooth penalty.
function _scholtes_level_objective(goop::ParametricGOOP, i, k, value, preference_slack)
    goop.is_prioritized_constraint[i][k] && return _num(sum(preference_slack))
    return _num(value isa AbstractVector ? only(value) : value)
end

# One player's rows, duals and complementarity factors (source: `_player_system`).
function _scholtes_player_system(
    goop::ParametricGOOP,
    i::Int,
    x,
    θ,
    xi::Vector{Symbolics.Num},
    trunc::Int,
    ρ,
)
    K = length(goop.preferences[i])
    nE, nI = goop.equality_dims[i], goop.inequality_dims[i]
    h_val =
        nE > 0 ? collect(Symbolics.Num, goop.equality_constraints[i](x, θ)) :
        Symbolics.Num[]
    g_val =
        nI > 0 ? collect(Symbolics.Num, goop.inequality_constraints[i](x, θ)) :
        Symbolics.Num[]
    sym(base, args...) = Symbolics.variable(Symbol(base, "_p$(i)", args...))

    values = [preference(x, θ) for preference in goop.preferences[i]]
    prioritized = goop.is_prioritized_constraint[i]
    as_vector(value) = collect(Symbolics.Num, value isa AbstractVector ? value : [value])
    preference_values = [prioritized[k] ? as_vector(values[k]) : Symbolics.Num[] for k in 1:K]
    # Each source level owns ONE vector. All upper levels reference those same
    # variables and carry both h_m + e_m >= 0 and e_m >= 0 with fresh duals.
    preference_slacks = [
        Symbolics.Num[sym("preference_slack", "_k$(k)_$j") for j in eachindex(preference_values[k])]
        for k in 1:K
    ]

    function level_data(k)
        lam = Symbolics.Num[sym("lam", k == K ? "_K$j" : "_k$(k)_$j") for j in 1:nE]
        gam = Symbolics.Num[sym("gam", k == K ? "_K$j" : "_k$(k)_$j") for j in 1:nI]
        G = copy(g_val)
        labels = NamedTuple[
            (; kind = :gamma, player = i, level = k, lower_level = 0, inequality = j)
            for j in 1:nI
        ]
        for m in k:K
            prioritized[m] || continue
            e = preference_slacks[m]
            append!(G, preference_values[m] .+ e)
            append!(G, e)
            append!(gam, Symbolics.Num[sym("gam_preference", "_k$(k)_m$(m)_$j") for j in eachindex(e)])
            append!(gam, Symbolics.Num[sym("mu_preference", "_k$(k)_m$(m)_$j") for j in eachindex(e)])
            for kind in (:preference, :preference_slack), j in eachindex(e)
                push!(labels, (; kind, player = i, level = k, lower_level = m, inequality = j))
            end
        end
        # Match goop.jl's commented vars = vcat(x[Block(player)], preference_slack):
        # differentiate only the current level's slack, holding inherited slacks
        # and lower-level multipliers fixed in upper-level stationarity.
        vars = vcat(xi, preference_slacks[k])
        objective = _scholtes_level_objective(goop, i, k, values[k], preference_slacks[k])
        terms = _LTerm[_LTerm(objective)]
        nE > 0 && push!(terms, _LTerm(h_val; duals = lam))
        !isempty(G) && push!(terms, _LTerm(G; duals = gam))
        return (; vars, G, gam, lam, terms, labels)
    end
    levels = [level_data(k) for k in 1:K]

    # Innermost level k = K: no policy chain, no φ. Preference slack gradients
    # include 1 - gam_preference - mu_preference = 0 at their source level.
    inner = levels[K]
    gL_K, chain_K = _diff_sum(inner.terms, inner.vars, trunc)
    stat = Vector{Symbolics.Num}[gL_K]
    chains = Vector{_LTerm}[chain_K]
    duals = Symbolics.Num[inner.lam; inner.gam]
    phi_pairs = NamedTuple[]

    # Upper levels k = K-1, …, 1.
    for k in (K - 1):-1:1
        # Lower stationarity columns can have different lengths because each
        # level includes its own newly introduced preference slacks.
        n_policy = sum(length(group[1].expr) for group in chains)
        ψk = Symbolics.Num[sym("psi", "_k$(k)_$j") for j in 1:n_policy]
        φk = Vector{Symbolics.Num}[]
        current = levels[k]
        terms = copy(current.terms)
        append!(terms, _couple_policy_terms(chains, ψk))
        for m in K:-1:(k + 1)
            lower = levels[m]
            product = lower.G .* lower.gam
            φ = Symbolics.Num[sym("phi_upper", "_k$(k)_m$(m)_$j") for j in eachindex(lower.G)]
            slack = ρ .- product
            push!(terms, _LTerm(slack; duals = φ))   # −φᵀ(ρ − G_m⊙γ_m)
            push!(φk, φ)
            for j in eachindex(lower.G)
                push!(
                    phi_pairs,
                    (;
                        a = slack[j],
                        b = φ[j],
                        player = i,
                        level = k,
                        lower_level = m,
                        inequality = j,
                        kind = :phi_upper,
                    ),
                )
            end
        end

        gL_k, chain_k = _diff_sum(terms, current.vars, trunc)
        push!(stat, gL_k)
        pushfirst!(chains, chain_k)
        append!(duals, ψk)
        foreach(block -> append!(duals, block), φk)
        append!(duals, current.lam)
        append!(duals, current.gam)
    end

    reverse!(stat)
    # One h_m + e_m row at the source level initializes each e_m. The upper
    # copies refer to the same e_m but need no additional primal coordinate.
    preference_slack_rows = Int[]
    row_offset = 0
    for k in 1:K
        append!(preference_slack_rows, row_offset .+ nI .+ eachindex(preference_slacks[k]))
        row_offset += length(levels[k].G)
    end
    return (
        F_nc = [vcat(stat...); h_val],
        duals,
        preference_slacks = reduce(vcat, preference_slacks; init = Symbolics.Num[]),
        preference_slack_rows,
        comp_a = reduce(vcat, (level.G for level in levels); init = Symbolics.Num[]),
        comp_b = reduce(vcat, (level.gam for level in levels); init = Symbolics.Num[]),
        comp_labels = reduce(vcat, (level.labels for level in levels); init = NamedTuple[]),
        phi_pairs,
    )
end

abstract type AbstractScholtesKKTSystem end

_direct_sign_dims(::AbstractScholtesKKTSystem) = ()
_initialize_direct_signs!(::AbstractScholtesKKTSystem, y) = y

"""
Numeric reduced KKT system in Scholtes form, built by
`generate_slacked_reduced_kkt_system(goop; complementarity = :scholtes)`.

Holds in-place evaluators `F_nc!`, `a!`, `b!` and sparse-Jacobian fillers `J_nc!`, `J_a!`,
`J_b!` (all `(out, y, θ, ρ)`), the Jacobian buffers and their structural patterns, and
the evaluation state the solver sets: the parameters `θ`, the relaxation level `ρ`, and a
Jacobian memo (`Jy`, `Jvalid`), since one Newton step asks for the Jacobian several times
at the same iterate.
"""
struct ScholtesKKTSystem{T1,T2,T3,T4,T5,T6} <: AbstractScholtesKKTSystem
    "Length of y = [x; preference slacks; duals]"
    n::Int
    "Number of non-complementarity rows (stationarity and equalities)"
    n_nc::Int
    "Number of complementarity pairs (original, then φ)"
    n_comp::Int
    "Coordinates of y holding the primal x"
    primal_dims::UnitRange{Int}
    "Coordinates of y holding explicit prioritized-constraint preference slacks"
    preference_slack_dims::UnitRange{Int}
    "One source-level h + preference_slack row of a per preference slack, in coordinate order"
    preference_slack_rows::Vector{Int}
    "Length of θ"
    parameter_dimension::Int
    F_nc!::T1
    a!::T2
    b!::T3
    J_nc!::T4
    J_a!::T5
    J_b!::T6
    Jnc_buf::SparseArrays.SparseMatrixCSC{Float64,Int}
    Ja_buf::SparseArrays.SparseMatrixCSC{Float64,Int}
    Jb_buf::SparseArrays.SparseMatrixCSC{Float64,Int}
    "(rows, cols) of each Jacobian's structural nonzeros, in `nonzeros` order"
    S_nc::Tuple{Vector{Int},Vector{Int}}
    S_a::Tuple{Vector{Int},Vector{Int}}
    S_b::Tuple{Vector{Int},Vector{Int}}
    "Compact u index per complementarity row; 0 marks an exact φ row"
    u_index::Vector{Int}
    "Number of relaxed G/γ pairs, including preference inequalities; remaining rows are exact φ pairs"
    original_nc::Int
    "Kind, player, level, lower level and inequality of each complementarity row"
    labels::Vector{NamedTuple}
    "Symbolic y, for looking up columns by name"
    vars::Vector{Symbolics.Num}
    θ::Vector{Float64}
    ρ::Base.RefValue{Float64}
    Jy::Vector{Float64}
    Jvalid::Base.RefValue{Bool}
end

# Compile `exprs(y; θ, ρ)` to `f!(out, y, θ, ρ)`, and its sparse Jacobian w.r.t. y to
# `J!(buffer, y, θ, ρ)` writing the buffer's `nonzeros` in CSC order.
function _compile_scholtes_block(exprs, y, θ, ρ, η; codegen, fd_codegen_chunk_size, backend_options)
    n = length(y)
    if isempty(exprs)
        buffer = SparseArrays.spzeros(0, n)
        return (out, y, θ, ρ) -> out, (J, y, θ, ρ) -> J, buffer, (Int[], Int[])
    end
    J_symbolic = _build_symbolics_sparse_jacobian(exprs, y)
    rows, cols, values = SparseArrays.findnz(J_symbolic)
    buffer = SparseArrays.sparse(rows, cols, zeros(length(rows)), length(exprs), n)
    compile(e) =
        if codegen === :fast_differentiation
            _build_fd_codegen_function(e, y, θ, ρ, η; chunk_size = fd_codegen_chunk_size)
        else
            SymbolicTracingUtils.build_function(e, y, θ, ρ, η; in_place = true, backend_options)
        end
    _f! = compile(exprs)
    _v! = isempty(values) ? ((out, args...) -> out) : compile(values)
    f! = (out, y, θ, ρ) -> (_f!(out, y, θ, ρ, 0.0); out)
    J! = (J, y, θ, ρ) -> (_v!(SparseArrays.nonzeros(J), y, θ, ρ, 0.0); J)
    return f!, J!, buffer, (rows, cols)
end

"""
Default `fd_codegen_chunk_size` of the Scholtes path. The FastDifferentiation code generator
otherwise emits each block as one function, whose compile time grows faster than linearly
with its size (robotic arm: first solve 44 s unchunked, about 18 s at 32; see
_port_logs/step7/startup_time.md). `nothing` restores a single function per block.
"""
const SCHOLTES_FD_CHUNK_SIZE = 32

"""
    generate_scholtes_reduced_kkt_system(goop; quasi = false, quasi_order = 2, phi = true,
        codegen = :fast_differentiation, fd_codegen_chunk_size = SCHOLTES_FD_CHUNK_SIZE,
        backend_options = (;))

Build and compile the reduced KKT system of `goop` in Scholtes form (see the top of
src/scholtes_kkt.jl). `quasi = true` drops derivative terms of order above `quasi_order`
without changing the system's shape. Only `phi = true` is supported.
"""
function generate_scholtes_reduced_kkt_system(
    goop::ParametricGOOP;
    quasi::Bool = false,
    quasi_order::Int = 2,
    phi::Bool = true,
    codegen = :fast_differentiation,
    fd_codegen_chunk_size = SCHOLTES_FD_CHUNK_SIZE,
    backend_options = (;),
)
    phi || throw(
        ArgumentError(
            "complementarity = :scholtes supports only phi = true (explicit φ ≥ 0 rows).",
        ),
    )
    quasi_order >= 1 || throw(ArgumentError("quasi_order must be >= 1, got $quasi_order"))
    codegen in (:native, :fast_differentiation) ||
        throw(ArgumentError("Unknown codegen option: $(codegen)."))
    trunc = quasi ? quasi_order : typemax(Int)
    backend = SymbolicTracingUtils.SymbolicsBackend()

    local F_nc, a, b, y, θ_symbolic, ρ, η, labels, exact, original_nc, preference_slack_dims, preference_slack_rows
    @timeit TO "Scholtes symbolic construction" begin
        x = SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims))
        θ_symbolic = SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims))
        x_blocks = to_blockvector(goop.primal_dims)(x)
        θ_blocks = to_blockvector(goop.parameter_dims)(θ_symbolic)
        ρ = Symbolics.variable(:scholtes_rho)
        η = Symbolics.variable(:scholtes_unused_eta)
        stops = cumsum(goop.primal_dims)
        players = [
            _scholtes_player_system(
                goop,
                i,
                x_blocks,
                θ_blocks,
                x[(stops[i] - goop.primal_dims[i] + 1):stops[i]],
                trunc,
                ρ,
            ) for i in 1:(goop.num_players)
        ]

        preference_slack_rows = Int[]
        row_offset = 0
        for p in players
            append!(preference_slack_rows, row_offset .+ p.preference_slack_rows)
            row_offset += length(p.comp_a)
        end
        preference_slacks = reduce(vcat, (p.preference_slacks for p in players); init = Symbolics.Num[])
        preference_slack_dims = (length(x) + 1):(length(x) + length(preference_slacks))
        y = Symbolics.Num[x; preference_slacks; reduce(vcat, (p.duals for p in players); init = Symbolics.Num[])]
        F_nc = reduce(vcat, (p.F_nc for p in players); init = Symbolics.Num[])
        a = reduce(vcat, (p.comp_a for p in players); init = Symbolics.Num[])
        b = reduce(vcat, (p.comp_b for p in players); init = Symbolics.Num[])
        original_nc = length(a)
        labels = reduce(vcat, (p.comp_labels for p in players); init = NamedTuple[])
        exact = falses(original_nc)
        for p in players, pair in p.phi_pairs
            push!(a, pair.a)
            push!(b, pair.b)
            push!(exact, true)
            push!(
                labels,
                (;
                    pair.kind,
                    pair.player,
                    pair.level,
                    pair.lower_level,
                    pair.inequality,
                ),
            )
        end
    end

    blocks = @timeit TO "Scholtes KKT codegen" map((F_nc, a, b)) do exprs
        _compile_scholtes_block(
            exprs,
            y,
            θ_symbolic,
            ρ,
            η;
            codegen,
            fd_codegen_chunk_size,
            backend_options,
        )
    end
    (F_nc!, J_nc!, Jnc_buf, S_nc), (a!, J_a!, Ja_buf, S_a), (b!, J_b!, Jb_buf, S_b) = blocks

    n_comp = length(a)
    u_index = zeros(Int, n_comp)
    relaxed_rows = findall(!, exact)
    u_index[relaxed_rows] .= 1:length(relaxed_rows)

    ScholtesKKTSystem(
        length(y),
        length(F_nc),
        n_comp,
        1:sum(goop.primal_dims),
        preference_slack_dims,
        preference_slack_rows,
        sum(goop.parameter_dims),
        F_nc!,
        a!,
        b!,
        J_nc!,
        J_a!,
        J_b!,
        Jnc_buf,
        Ja_buf,
        Jb_buf,
        S_nc,
        S_a,
        S_b,
        u_index,
        original_nc,
        labels,
        y,
        zeros(sum(goop.parameter_dims)),
        Ref(NaN),
        zeros(length(y)),
        Ref(false),
    )
end

# --- evaluation state -----------------------------------------------------------------

function _set_rho!(kkt::AbstractScholtesKKTSystem, ρ)
    if kkt.ρ[] != ρ
        kkt.ρ[] = ρ
        kkt.Jvalid[] = false
    end
    kkt
end

function _set_theta!(kkt::AbstractScholtesKKTSystem, θ)
    length(θ) == kkt.parameter_dimension || throw(
        ArgumentError("θ has length $(length(θ)), expected $(kkt.parameter_dimension)"),
    )
    if kkt.θ != θ
        copyto!(kkt.θ, θ)
        kkt.Jvalid[] = false
    end
    kkt
end

_F_nc!(kkt::AbstractScholtesKKTSystem, out, y) = kkt.F_nc!(out, y, kkt.θ, kkt.ρ[])
_a!(kkt::AbstractScholtesKKTSystem, out, y) = kkt.a!(out, y, kkt.θ, kkt.ρ[])
_b!(kkt::AbstractScholtesKKTSystem, out, y) = kkt.b!(out, y, kkt.θ, kkt.ρ[])
_F_nc(kkt::AbstractScholtesKKTSystem, y) = _F_nc!(kkt, zeros(kkt.n_nc), y)
_a(kkt::AbstractScholtesKKTSystem, y) = _a!(kkt, zeros(kkt.n_comp), y)
_b(kkt::AbstractScholtesKKTSystem, y) = _b!(kkt, zeros(kkt.n_comp), y)

_n_u(kkt::AbstractScholtesKKTSystem) = maximum(kkt.u_index; init = 0)
@inline _u_col(kkt::AbstractScholtesKKTSystem, j::Int) =
    kkt.u_index[j] == 0 ? 0 : kkt.n + kkt.n_comp + kkt.u_index[j]
