""" The KKT system of a GOOP,

                    K_ρ(z; θ) = 0,      z_k > 0 for k in `nonnegative_dims`,

as the generators return it: the reduced interior-point system (5.2)
(`generate_slacked_reduced_kkt_system`), the complete systems (2.12)–(2.17) and (6.1)
(`generate_slacked_complete_kkt_system`) and the reduced system in Scholtes form
(`generate_slacked_scholtes_reduced_kkt_system`). θ are the parameters and ρ ≥ 0 relaxes the
complementarity rows.

The fields, by group:
  - compiled: the residual `K_ρ!(out, z; θ, ρ)`, its sparse Jacobian `∇K_ρ!(J, z; θ, ρ)`, for
    complete and Scholtes systems `slack_targets!(out, z; θ, ρ)`, the expressions that the slacks
    σ stand for (rows expression − σ = 0), and `K₀!(out, z; θ)`, the unrelaxed residual;
  - sizes: `kkt_dimension` (rows of K_ρ), `variable_dimension` (length of z),
    `parameter_dimension` (length of θ);
  - index sets of z: x, s, σ, γ, λ, ψ, and `nonnegative_dims`, the coordinates the solver keeps
    positive: σ and γ (and for the Scholtes system ϕ, σ̂ and u). The preference slacks s are not
    among them: s ≥ 0 is a row of K_ρ, s − σ = 0, like every other inequality;
  - symbolic: `K_ρ_symbolic`, `z_symbolic`, for complete and Scholtes systems also
    `slack_expressions`, for complete systems `G_symbolic` and `levels`;
  - the tags: `kind`, the system (`:reduced`, `:complete` or `:scholtes`), and `complementarity`,
    its complementarity rows (`:interior_point`, `:exact`, `:perturbed` or `:scholtes`);
  - for the Scholtes system only: `complementarity_pairs` and `preference_slack_rows`, which its
    start (`initialize_scholtes!`) and K₀ read.
The evaluation state (θ, ρ) belongs to the solver, not to the container.

# The reduced system (`generate_slacked_reduced_kkt_system`)

Player i has levels k = 1 (outermost) … Kⁱ (innermost) with objectives hₖ, its own
equalities f(x, θ) = 0 and inequalities g(x, θ) ≥ 0. A level whose
`is_prioritized_constraint` flag is set treats hₖ as a constraint hₖ ≥ 0 with preference
slacks sₖ: it minimizes cₖ = 1ᵀsₖ subject to hₖ + sₖ ≥ 0 and sₖ ≥ 0 (otherwise cₖ = hₖ).
Level k's inequalities Gₖ are g and, for every prioritized m ≥ k, hₘ + sₘ and sₘ. Each
level k carries its own copies λₖ, γₖ, σₖ (γₖ and σₖ over all of Gₖ) and has the Lagrangian

    Lₖ = cₖ − λₖᵀf − γₖᵀGₖ − ψₖᵀπₖ₊₁ − Σ_{m>k} φₖ,ₘᵀ(Gₘ ⊙ γₘ),

where πₖ₊₁ = [∇Lₖ₊₁; πₖ₊₂; …] stacks the stationarity residuals of every level below
k (the lower levels' policy, enforced through the multipliers ψₖ), and ∇ is the gradient
w.r.t. player i's own xⁱ and, at a prioritized level, sₖ. The last two terms are absent at
the innermost level. K_ρ stacks, for every player and level,

    ∇Lₖ = 0                             stationarity
    f = 0                               innermost level only
    Gₖ − σₖ = 0                         interior-point slack
    σₖ ⊙ γₖ − ρ𝟙 = 0                    relaxed complementarity

over z = [x; s; σ; λ; γ; ψ; φ]. The solver keeps σ and γ positive, drives ρ → 0,
and recovers x from the primal block. φ enters only through the stationarity rows. With
`drop_higher_order_terms = true` (quasi), every term of ∇Lₖ that would be differentiated
more than `quasi_order` times (default 2) is set to zero.

For the Scholtes form of the same hierarchy (exact φ rows, ρ homotopy) see
`generate_slacked_scholtes_reduced_kkt_system`.
"""

struct GOOPKKTSystem{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14}
    "A callable function that computes K_ρ!(out, z; θ, ρ) in-place for 'out'"
    K_ρ!::T1
    "A `SparseFunction` that computes ∇K_ρ!(J, z; θ, ρ) in-place for 'J'; it carries the Jacobian's buffer (`result_buffer`) and pattern (`rows`, `cols`)"
    ∇K_ρ!::T2
    "A callable function that computes slack_targets!(out, z; θ, ρ), the slack expressions, in-place for 'out' (complete and Scholtes systems; a no-op otherwise)"
    slack_targets!::T3
    "A callable function that computes K₀!(out, z; θ), the unrelaxed residual: K_ρ at ρ = 0, and for the Scholtes system with u = 0"
    K₀!::T4
    "KKT dimension (rows of K_ρ)"
    kkt_dimension::Int
    "Length of z vector"
    variable_dimension::Int
    "Length of θ vector"
    parameter_dimension::Int
    "Coordinates of z associated to the primal x"
    primal_dims::T5
    "Coordinates of z associated to preference slacks"
    preference_slack_dims::T5
    "Coordinates of z associated to interior point slacks (in the order of `slack_expressions` when it is given)"
    interior_point_slack_dims::T6
    "Coordinates of z associated to inequality constraint duals"
    inequality_constraint_dual_dims::T7
    "Coordinates of z associated to equality constraint duals"
    equality_constraint_dual_dims::T8
    "Coordinates of z associated to stationarity duals"
    stationarity_dual_dims::T9
    "Coordinates of z associated to equality or stationarity duals"
    all_equality_stationarity_dual_dims::T10
    "Coordinates of stationarity duals associated with innermost preferences (reduced system only; empty otherwise)"
    innermost_stationarity_dual_dims::T11
    "Coordinates of z the solver keeps positive (the sign box): interior point slacks and inequality constraint duals, and for the Scholtes system ϕ, σ̂ and u"
    nonnegative_dims::T12
    "K_ρ in symbolics"
    K_ρ_symbolic::T13
    "z in symbolics"
    z_symbolic::T13
    "The expressions the interior point slacks stand for, in symbolics (rows expression − σ = 0 of K_ρ; complete systems; empty otherwise)"
    slack_expressions::T13
    "Every level's inequalities in symbolics (complete systems; empty otherwise)"
    G_symbolic::T13
    "Per player and level, the symbolic data of the complete construction (complete systems; empty otherwise)"
    levels::T14
    "The system: :reduced, :complete or :scholtes"
    kind::Symbol
    "Its complementarity rows: :interior_point (reduced), :exact or :perturbed (complete), :scholtes"
    complementarity::Symbol
    # Required for the Scholtes system only (empty otherwise): its start, `initialize_scholtes!`, and K₀ read them.
    "Per complementarity product, (; slack, multiplier, u_relaxation, label): the coordinates of σ (σ̂), γ (ϕ) and u (0 on an exact product), and the kind, player, level, lower level and inequality; the relaxed γ pairs, then the exact ϕ pairs (Scholtes only)"
    complementarity_pairs::Vector{NamedTuple}
    "Per preference slack, the index of its h + s row among `slack_expressions` (Scholtes only)"
    preference_slack_rows::Vector{Int}
end

"""
Recursively convert a Symbolics expression to FastDifferentiation nodes.

`varmap` maps every unwrapped Symbolics variable to its FD counterpart; `memo`
caches converted sub-expressions so shared subtrees map to the *same* FD node,
which is what lets FD's DAG-based code generator collapse repeated work.
"""
function _symbolics_to_fd(x, varmap, memo)
    x = x isa Symbolics.Num ? Symbolics.unwrap(x) : x
    x isa Number && return x
    # SymbolicUtils v3 wraps literal constants as symbolic nodes.
    if x isa Symbolics.SymbolicUtils.BasicSymbolic && Symbolics.SymbolicUtils.isconst(x)
        return Symbolics.SymbolicUtils.unwrap_const(x)
    end
    haskey(memo, x) && return memo[x]
    # The varmap lookup must precede the call check: scalarized array variables
    # such as `z[5]` are themselves `getindex` call expressions.
    result = if haskey(varmap, x)
        varmap[x]
    elseif Symbolics.SymbolicUtils.iscall(x)
        op = Symbolics.SymbolicUtils.operation(x)
        # `sorted_arguments`, not `arguments`: for a sum or product, `arguments` follows the
        # iteration order of SymbolicUtils' internal Dict, which can differ between Julia
        # processes. FD then adds the same terms in a different order, the last bits of a few
        # generated rows change, and solves can take different paths. The canonical order, also
        # used by SymbolicUtils' own code generator, makes the generated code identical in every
        # process.
        args = map(
            arg -> _symbolics_to_fd(arg, varmap, memo),
            Symbolics.SymbolicUtils.sorted_arguments(x),
        )
        if op === ifelse || op === Core.ifelse
            # FD supports if_else for evaluation/codegen (differentiation has
            # already happened on the Symbolics side).
            SymbolicTracingUtils.FD.if_else(args...)
        else
            op(args...)
        end
    else
        error(
            "Cannot convert Symbolics node `$x` of type $(typeof(x)) to FastDifferentiation.",
        )
    end
    memo[x] = result
    result
end

"""
Compile Symbolics expressions to in-place callables through FastDifferentiation's
code generator. Differentiation is NOT redone: the expressions (typically already
containing symbolic derivatives) are converted node-by-node, so the emitted code
computes exactly the same quantities as Symbolics' own code generator. FD's
hash-consed DAG shares common subexpressions across all outputs, which shrinks
the generated code by orders of magnitude and avoids the pathological LLVM
compile times of the Symbolics/RuntimeGeneratedFunctions path on large systems.

Returns an in-place function `f!(result, z, θ, scalars...)`, one runtime value per symbolic
scalar parameter (every caller passes one, the relaxation ρ).
"""
function _build_fd_codegen_function(
    exprs::AbstractVector{<:Symbolics.Num},
    z_symbolic,
    θ_symbolic,
    scalar_symbolics...;
    chunk_size = nothing,
)
    isnothing(chunk_size) ||
        chunk_size > 0 ||
        throw(ArgumentError("FastDifferentiation codegen chunk_size must be positive."))
    fd_z = SymbolicTracingUtils.FD.make_variables(:z, length(z_symbolic))
    fd_θ = SymbolicTracingUtils.FD.make_variables(:θ, length(θ_symbolic))
    fd_scalars = [
        SymbolicTracingUtils.FD.make_variables(Symbol(:scalar, i), 1) for
        i in eachindex(scalar_symbolics)
    ]

    varmap = Dict{Any,Any}()
    for (s, f) in zip(
        Iterators.flatten((z_symbolic, θ_symbolic, scalar_symbolics)),
        Iterators.flatten((fd_z, fd_θ, fd_scalars...)),
    )
        varmap[Symbolics.unwrap(s)] = f
    end

    memo = IdDict{Any,Any}()
    fd_exprs = SymbolicTracingUtils.FD.Node[
        SymbolicTracingUtils.FD.Node(_symbolics_to_fd(expr, varmap, memo)) for
        expr in exprs
    ]

    # A single concatenated input vector keeps the generated function's arity
    # in sync with the runtime call below, which flattens its arguments the
    # same way.
    fd_inputs = vcat(fd_z, fd_θ, fd_scalars...)
    if isnothing(chunk_size) || length(fd_exprs) <= chunk_size
        _f! = SymbolicTracingUtils.FD.make_function(fd_exprs, fd_inputs; in_place = true)
        return (result, z, θ, scalars...) -> _f!(result, vcat(z, θ, scalars...))
    end

    chunk_ranges = [
        start:min(start + chunk_size - 1, length(fd_exprs)) for
        start in 1:chunk_size:length(fd_exprs)
    ]
    # Keep these abstractly typed on purpose. Dynamic dispatch provides a
    # function barrier between chunks, preventing Julia inference from merging
    # all generated ASTs back into one enormous compiler unit.
    chunk_functions = Function[
        SymbolicTracingUtils.FD.make_function(
            collect(@view fd_exprs[range]),
            fd_inputs;
            in_place = true,
        ) for range in chunk_ranges
    ]
    function (result, z, θ, scalars...)
        inputs = vcat(z, θ, scalars...)
        for (range, f!) in zip(chunk_ranges, chunk_functions)
            f!(@view(result[range]), inputs)
        end
        nothing
    end
end

"""
Construct a Symbolics sparse Jacobian while timing sparsity discovery,
derivative expansion, and sparse assembly separately.

`Symbolics.sparsejacobian` first discovers the sparsity pattern and then
expands every structural derivative in a serial loop. Keeping those phases
explicit here provides actionable profiling without changing Symbolics'
derivative semantics. When the residual contains no deferred Differential
operators—as is true for generated GOOP KKT systems—calling `executediff`
directly avoids recursively checking the already-expanded residual again for
every structural nonzero.
"""
function _build_symbolics_sparse_jacobian(
    ops::AbstractVector{<:Symbolics.Num},
    vars::AbstractVector{<:Symbolics.Num},
)
    sp = @timeit TO "Symbolics Jacobian sparsity detection" Symbolics.jacobian_sparsity(
        ops,
        vars,
    )
    rows, cols, _ = SparseArrays.findnz(sp)
    values = @timeit TO "Symbolics derivative expansion" begin
        unwrapped_ops = Symbolics.unwrap.(ops)
        unwrapped_vars = Symbolics.unwrap.(vars)
        if any(Symbolics.hasderiv, unwrapped_ops)
            # Preserve the general Symbolics path if a caller supplies residuals
            # containing deferred Differential operators.
            Symbolics.sparsejacobian_vals(copy(ops), copy(vars), rows, cols)
        else
            differentials = Symbolics.Differential.(unwrapped_vars)
            map(rows, cols) do row, col
                Symbolics.Num(Symbolics.executediff(differentials[col], unwrapped_ops[row]))
            end
        end
    end

    @timeit TO "Symbolics sparse Jacobian assembly" SparseArrays.sparse(
        rows,
        cols,
        values,
        length(ops),
        length(vars),
    )
end

"""
Default `fd_codegen_chunk_size` of every generator. The FastDifferentiation
code generator otherwise emits each block as one function, whose compile time grows faster
than linearly with its size. `nothing` restores a single function per block.
"""
const FD_CODEGEN_CHUNK_SIZE = 32

"Compile the rows `exprs(z; θ, ρ)` to `f!(out, z, θ, ρ)`, which returns `out`."
function _compile_kkt_rows(exprs, z, θ, ρ; codegen, fd_codegen_chunk_size, backend_options)
    isempty(exprs) && return (out, z, θ, ρ) -> out
    _f! = if codegen === :fast_differentiation
        _build_fd_codegen_function(exprs, z, θ, ρ; chunk_size = fd_codegen_chunk_size)
    else
        SymbolicTracingUtils.build_function(exprs, z, θ, ρ; in_place = true, backend_options)
    end
    (out, z, θ, ρ) -> (_f!(out, z, θ, ρ); out)
end

"""
Compile the rows `exprs(z; θ, ρ)` to `f!(out, z, θ, ρ)`, and their sparse Jacobian w.r.t. z to
`J!(buffer, z, θ, ρ)`, which writes the buffer's `nonzeros` in CSC order. Returns
`(f!, J!, buffer, (rows, cols))`, the last being the Jacobian's structural nonzeros.
"""
function _compile_kkt_block(exprs, z, θ, ρ; codegen, fd_codegen_chunk_size, backend_options)
    n = length(z)
    if isempty(exprs)
        buffer = SparseArrays.spzeros(0, n)
        return (out, z, θ, ρ) -> out, (J, z, θ, ρ) -> J, buffer, (Int[], Int[])
    end
    J_symbolic = _build_symbolics_sparse_jacobian(exprs, z)
    rows, cols, values = SparseArrays.findnz(J_symbolic)
    buffer = SparseArrays.sparse(rows, cols, zeros(length(rows)), length(exprs), n)
    compile(e) = _compile_kkt_rows(e, z, θ, ρ; codegen, fd_codegen_chunk_size, backend_options)
    f! = compile(exprs)
    v! = compile(values)
    J! = (J, z, θ, ρ) -> (v!(SparseArrays.nonzeros(J), z, θ, ρ); J)
    return f!, J!, buffer, (rows, cols)
end

"""
Compile the symbolic rows `K_ρ_symbolic(z; θ, ρ)` of a GOOP KKT system into a `GOOPKKTSystem`: the
residual, its sparse Jacobian w.r.t. z and the slack targets (`slack_expressions`, complete systems
only), through one compile path, `_compile_kkt_block` (FastDifferentiation by default, as every study
uses).
`complementarity = :exact` has no relaxation: its functions evaluate at ρ = 0 whatever ρ they are
given, as the complete systems' container always did.
"""
function BuildGOOPKKTSystem(
    K_ρ_symbolic::Vector{T},
    z_symbolic::Vector{T},
    θ_symbolic::Vector{T},
    ρ_symbolic::T,
    primal_dims,
    preference_slack_dims,
    interior_point_slack_dims,
    inequality_constraint_dual_dims;
    equality_constraint_dual_dims = Int[],
    stationarity_dual_dims = Int[],
    all_equality_stationarity_dual_dims = Int[],
    innermost_stationarity_dual_dims = Int[],
    nonnegative_dims,
    slack_expressions = T[],
    G_symbolic = T[],
    levels = Any[],
    kind,
    complementarity,
    complementarity_pairs = NamedTuple[],
    preference_slack_rows = Int[],
    backend_options = (;),
    codegen = :fast_differentiation,
    fd_codegen_chunk_size = FD_CODEGEN_CHUNK_SIZE,
) where {T<:SymbolicTracingUtils.Symbolics.Num}
    codegen in (:native, :fast_differentiation) ||
        error("Unknown codegen option: $(codegen). Use :native or :fast_differentiation.")
    (kind, complementarity) in
    ((:reduced, :interior_point), (:complete, :exact), (:complete, :perturbed), (:scholtes, :scholtes)) ||
        error("Unknown system: kind = $kind, complementarity = $complementarity.")
    isempty(slack_expressions) || length(slack_expressions) == length(interior_point_slack_dims) ||
        error("one slack expression per interior point slack is expected")
    exact = complementarity === :exact
    @timeit TO "KKT executable construction" begin
        f!, J!, J_buffer, (rows, cols) = _compile_kkt_block(
            K_ρ_symbolic,
            z_symbolic,
            θ_symbolic,
            ρ_symbolic;
            codegen,
            fd_codegen_chunk_size,
            backend_options,
        )
        # The slack targets need no Jacobian.
        g! = _compile_kkt_rows(
            slack_expressions,
            z_symbolic,
            θ_symbolic,
            ρ_symbolic;
            codegen,
            fd_codegen_chunk_size,
            backend_options,
        )
        K_ρ! = (out, z; θ, ρ) -> f!(out, z, θ, exact ? 0.0 : ρ)
        ∇K_ρ! = SymbolicTracingUtils.SparseFunction(
            (J, z; θ, ρ) -> J!(J, z, θ, exact ? 0.0 : ρ),
            rows,
            cols,
            size(J_buffer),
        )
        slack_targets! = (out, z; θ, ρ) -> g!(out, z, θ, exact ? 0.0 : ρ)
        # K₀ is K_ρ at ρ = 0. The Scholtes system writes σ ⊙ γ ≤ ρ𝟙 as σ ⊙ γ + u − ρ𝟙 = 0 with u ≥ 0,
        # so its K₀ is K_ρ at ρ = 0 with u = 0: the row is then σ ⊙ γ. The u's coordinates in z come from
        # complementarity_pairs (Scholtes only; empty otherwise, and then K₀ is just K_ρ at ρ = 0): a
        # pair's `u_relaxation` is its u in σ ⊙ γ + u − ρ = 0, and 0 for an exact pair σ̂ ⊙ ϕ = 0.
        u_dims = [pair.u_relaxation for pair in complementarity_pairs if pair.u_relaxation != 0]
        K₀! = if isempty(u_dims)
            (out, z; θ) -> K_ρ!(out, z; θ, ρ = 0.0)
        else
            z_u_zero = zeros(length(z_symbolic)) # z with u = 0
            (out, z; θ) -> (z_u_zero .= z; z_u_zero[u_dims] .= 0.0; K_ρ!(out, z_u_zero; θ, ρ = 0.0))
        end

        GOOPKKTSystem(
            K_ρ!,
            ∇K_ρ!,
            slack_targets!,
            K₀!,
            length(K_ρ_symbolic),
            length(z_symbolic),
            length(θ_symbolic),
            primal_dims,
            preference_slack_dims,
            interior_point_slack_dims,
            inequality_constraint_dual_dims,
            equality_constraint_dual_dims,
            stationarity_dual_dims,
            all_equality_stationarity_dual_dims,
            innermost_stationarity_dual_dims,
            nonnegative_dims,
            K_ρ_symbolic,
            z_symbolic,
            slack_expressions,
            G_symbolic,
            levels,
            kind,
            complementarity,
            complementarity_pairs,
            preference_slack_rows,
        )
    end
end

"""
Compile the Scholtes engine's reduced KKT system (`_generate_scholtes_reduced_kkt_system`) and pack
it into the container the engine reads: a `ScholtesKKTSystem`
with the rows F and the complementarity blocks a and b, whose first `original_nc` pairs are
relaxed by the solver and the rest (the φ pairs) exact. `generate_slacked_scholtes_reduced_kkt_system`
builds the same system as a `GOOPKKTSystem`, without a and b.
"""
function BuildScholtesKKTSystem(
    F_symbolic::Vector{T},
    a_symbolic::Vector{T},
    b_symbolic::Vector{T},
    z_symbolic::Vector{T},
    θ_symbolic::Vector{T},
    ρ_symbolic::T,
    primal_dims,
    preference_slack_dims;
    preference_slack_rows,
    original_nc,
    labels,
    backend_options = (;),
    codegen = :fast_differentiation,
    fd_codegen_chunk_size = FD_CODEGEN_CHUNK_SIZE,
) where {T<:Symbolics.Num}
    (f!, J!, J_buffer, J_pattern), (a!, J_a!, Ja_buffer, Ja_pattern), (b!, J_b!, Jb_buffer, Jb_pattern) =
        map((F_symbolic, a_symbolic, b_symbolic)) do exprs
            _compile_kkt_block(
                exprs,
                z_symbolic,
                θ_symbolic,
                ρ_symbolic;
                codegen,
                fd_codegen_chunk_size,
                backend_options,
            )
        end

    n, m, p = length(z_symbolic), length(F_symbolic), length(θ_symbolic)
    # Each relaxed pair has its own u; 0 marks an exact φ pair.
    u_index = [1:original_nc; zeros(Int, length(a_symbolic) - original_nc)]
    # The evaluation state (θ, ρ, Jacobian memo) is set by the solver.
    ScholtesKKTSystem(
        n, m, length(a_symbolic), primal_dims, preference_slack_dims, preference_slack_rows, p,
        f!, a!, b!, J!, J_a!, J_b!, J_buffer, Ja_buffer, Jb_buffer,
        J_pattern, Ja_pattern, Jb_pattern, u_index, original_nc, labels, z_symbolic,
        zeros(p), Ref(NaN), zeros(n), Ref(false),
    )
end
