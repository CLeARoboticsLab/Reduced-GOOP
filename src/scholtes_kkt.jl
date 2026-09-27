# scholtes_kkt.jl -- the reduced KKT system in Scholtes form.
#
# Ported from ScholtesReducedGOOP.jl (Jingqi Li), `src/kkt.jl` and `src/compile.jl`,
# on top of `ParametricGOOP` and this package's code generator. Only the explicit
# formulation (`phi = true`) is supported: every player keeps its original
# `γ_k ≥ 0` on `g` at every level, and each upper level k adds, for each of the
# K−k levels m below it,
#
#     −φ_{k,m}ᵀ (ρ𝟙 − g ⊙ γ_m),        φ_{k,m} ≥ 0,
#
# whose lifted slack σ_{k,m} = ρ𝟙 − g ⊙ γ_m and φ_{k,m} satisfy EXACT
# complementarity σ ⊙ φ = 0. The original pairs are Scholtes-relaxed by the
# solver: s ⊙ γ + u = ρ𝟙. The system is returned as three blocks,
#
#     F_nc(y)  stationarity of every level (w.r.t. the player's own x only) + equalities
#     a(y)     g repeated Kⁱ times, then ρ𝟙 − g ⊙ γ_m for every φ pair
#     b(y)     γ_1 … γ_K, then φ                       (a coordinate projection)
#
# with y = [x; duals]. The slacks s and the relaxation u are solver-side blocks, as
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

# The scalar objective of player `i` at level `k` (1 = outermost). A prioritized
# constraint level uses the interior-point builder's smooth violation penalty.
function _scholtes_level_objective(goop::ParametricGOOP, i, k, x, θ)
    value = goop.preferences[i][k](x, θ)
    if goop.is_prioritized_constraint[i][k]
        return _num(sum(smooth_piecewise_preference_objective.(value, k)))
    end
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

    function base_terms(k, lam, gam)
        terms = _LTerm[_LTerm(_scholtes_level_objective(goop, i, k, x, θ))]
        nE > 0 && push!(terms, _LTerm(h_val; duals = lam))
        nI > 0 && push!(terms, _LTerm(g_val; duals = gam))
        return terms
    end

    # Innermost level k = K: no policy chain, no φ.
    lamK = [sym("lam", "_K$j") for j in 1:nE]
    gamK = [sym("gam", "_K$j") for j in 1:nI]
    gL_K, chain_K = _diff_sum(base_terms(K, lamK, gamK), xi, trunc)

    stat = Vector{Symbolics.Num}[gL_K]
    gams = Vector{Symbolics.Num}[gamK]
    chains = Vector{_LTerm}[chain_K]
    duals = Symbolics.Num[lamK; gamK]
    phi_pairs = NamedTuple[]

    # Upper levels k = K-1, …, 1.
    for k in (K - 1):-1:1
        n_lower = K - k
        ψk = [sym("psi", "_k$(k)_$j") for j in 1:(n_lower * length(xi))]
        φk = Vector{Symbolics.Num}[]
        lamk = [sym("lam", "_k$(k)_$j") for j in 1:nE]
        gamk = [sym("gam", "_k$(k)_$j") for j in 1:nI]

        terms = base_terms(k, lamk, gamk)
        append!(terms, _couple_policy_terms(chains, ψk))
        for ℓ in 1:n_lower
            m = K - ℓ + 1
            product = g_val .* gams[ℓ]
            φ = [sym("phi_upper", "_k$(k)_m$(m)_$j") for j in 1:nI]
            slack = ρ .- product
            push!(terms, _LTerm(slack; duals = φ))   # −φᵀ(ρ − g⊙γ_m)
            push!(φk, φ)
            for j in 1:nI
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

        gL_k, chain_k = _diff_sum(terms, xi, trunc)
        push!(stat, gL_k)
        push!(gams, gamk)
        pushfirst!(chains, chain_k)
        append!(duals, ψk)
        foreach(block -> append!(duals, block), φk)
        append!(duals, lamk)
        append!(duals, gamk)
    end

    reverse!(stat)
    reverse!(gams)
    return (
        F = [vcat(stat...); h_val; repeat(g_val, K) .* vcat(gams...)],
        duals,
        comp_offset = K * length(xi) + nE,
        comp_a = repeat(g_val, K),
        comp_b = vcat(gams...),
        phi_pairs,
    )
end

"""
Numeric reduced KKT system in Scholtes form, built by
`generate_slacked_reduced_kkt_system(goop; complementarity = :scholtes)`.

Holds in-place evaluators `F_nc!`, `a!`, `b!` and sparse-Jacobian fillers `J_nc!`, `J_a!`,
`J_b!` (all `(out, y, θ, ρ)`), the Jacobian buffers and their structural patterns, and
the evaluation state the solver sets: the parameters `θ`, the relaxation level `ρ`, and a
Jacobian memo (`Jy`, `Jvalid`), since one Newton step asks for the Jacobian several times
at the same iterate.
"""
struct ScholtesKKTSystem{T1,T2,T3,T4,T5,T6}
    "Length of y = [x; duals]"
    n::Int
    "Number of non-complementarity rows (stationarity and equalities)"
    n_nc::Int
    "Number of complementarity pairs (original, then φ)"
    n_comp::Int
    "Coordinates of y holding the primal x"
    primal_dims::UnitRange{Int}
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
    "Number of original g/γ pairs; rows past it are φ pairs"
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
    generate_scholtes_reduced_kkt_system(goop; quasi = false, quasi_order = 2, phi = true,
        codegen = :fast_differentiation, fd_codegen_chunk_size = nothing, backend_options = (;))

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
    fd_codegen_chunk_size = nothing,
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

    local F_nc, a, b, y, θ_symbolic, ρ, η, labels, exact, original_nc
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

        comp_idx = Int[]
        row_base = 0
        for (i, p) in enumerate(players)
            K = length(goop.preferences[i])
            start = row_base + p.comp_offset + 1
            append!(comp_idx, start:(start + K * goop.inequality_dims[i] - 1))
            row_base += length(p.F)
        end

        y = Symbolics.Num[x; reduce(vcat, (p.duals for p in players); init = Symbolics.Num[])]
        F = reduce(vcat, (p.F for p in players); init = Symbolics.Num[])
        a = reduce(vcat, (p.comp_a for p in players); init = Symbolics.Num[])
        b = reduce(vcat, (p.comp_b for p in players); init = Symbolics.Num[])
        original_nc = length(a)
        labels = NamedTuple[
            (; kind = :gamma, player = i, level = k, lower_level = 0, inequality = j) for
            i in 1:(goop.num_players) for k in 1:length(goop.preferences[i]) for
            j in 1:goop.inequality_dims[i]
        ]
        exact = falses(original_nc)
        for p in players, pair in p.phi_pairs
            push!(F, pair.a * pair.b)
            push!(comp_idx, length(F))
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
        F_nc = F[setdiff(1:length(F), comp_idx)]
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

function _set_rho!(kkt::ScholtesKKTSystem, ρ)
    if kkt.ρ[] != ρ
        kkt.ρ[] = ρ
        kkt.Jvalid[] = false
    end
    kkt
end

function _set_theta!(kkt::ScholtesKKTSystem, θ)
    length(θ) == kkt.parameter_dimension || throw(
        ArgumentError("θ has length $(length(θ)), expected $(kkt.parameter_dimension)"),
    )
    if kkt.θ != θ
        copyto!(kkt.θ, θ)
        kkt.Jvalid[] = false
    end
    kkt
end

_F_nc!(kkt::ScholtesKKTSystem, out, y) = kkt.F_nc!(out, y, kkt.θ, kkt.ρ[])
_a!(kkt::ScholtesKKTSystem, out, y) = kkt.a!(out, y, kkt.θ, kkt.ρ[])
_b!(kkt::ScholtesKKTSystem, out, y) = kkt.b!(out, y, kkt.θ, kkt.ρ[])
_F_nc(kkt::ScholtesKKTSystem, y) = _F_nc!(kkt, zeros(kkt.n_nc), y)
_a(kkt::ScholtesKKTSystem, y) = _a!(kkt, zeros(kkt.n_comp), y)
_b(kkt::ScholtesKKTSystem, y) = _b!(kkt, zeros(kkt.n_comp), y)

_n_u(kkt::ScholtesKKTSystem) = maximum(kkt.u_index; init = 0)
@inline _u_col(kkt::ScholtesKKTSystem, j::Int) =
    kkt.u_index[j] == 0 ? 0 : kkt.n + kkt.n_comp + kkt.u_index[j]
