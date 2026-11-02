""" Maintain callable functions to compute the KKT error for a GOOP problem
and its Jacobian with respect to all the (primal and dual) decision variables.
Also keeps track of several types of variables (preference slacks, interior
point slacks, and inequality constraint duals) that must satisfy nonnegativity
constraints that are not explicitly included in the KKT system.

The KKT system is parameterized by a vector θ and a scalar ϵ > 0 which
controls the complementarity relaxation of the interior point scheme.

# The reduced system (`generate_slacked_reduced_kkt_system`)

Player i has levels k = 1 (outermost) … Kⁱ (innermost) with objectives hₖ, its own
equalities f(x, θ) = 0 and inequalities g(x, θ) ≥ 0. A level whose
`is_prioritized_constraint` flag is set uses `smooth_piecewise_preference_objective`
(max(−hₖ, 0)^(k+2)) in place of hₖ. Each level k carries its own copies λₖ, γₖ, σₖ
and has the Lagrangian

    Lₖ = hₖ − λₖᵀf − γₖᵀg − ψₖᵀπₖ₊₁ − Σ_{m>k} φₖ,ₘᵀ(g ⊙ γₘ),

where πₖ₊₁ = [∇ₓLₖ₊₁; πₖ₊₂; …] stacks the stationarity residuals of every level below
k (the lower levels' policy, enforced through the multipliers ψₖ), and ∇ₓ is the gradient
w.r.t. player i's own xⁱ only. The last two terms are absent at the innermost level.
F stacks, for every player and level,

    ∇ₓLₖ + η xⁱ = 0                     stationarity (η: Newton regularization)
    f = 0                               innermost level only
    g − σₖ = 0                          interior-point slack
    σₖ ⊙ γₖ − ϵ𝟙 = 0                    relaxed complementarity

over z = [x; s; σ; λ; γ; ψ; φ]. The solver keeps σ, γ (and s) positive, drives ϵ → 0,
and recovers x from the primal block. φ enters only through the stationarity rows. The
reduced path creates no preference slacks s. With `drop_higher_order_terms = true`
(quasi), every term of ∇ₓLₖ that would be differentiated more than twice is set to zero.

For the Scholtes form of the same hierarchy (exact φ rows, ρ homotopy) see
`ScholtesKKTSystem` in src/scholtes_kkt.jl.
"""

struct GOOPKKTSystem{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11}
    "A callable function that computes F!(val, z; θ, ϵ) in-place for 'val'"
    F!::T1
    "A callable function that computes ∇F!(val, z; θ, ϵ) in-place for 'val'"
    ∇F_z!::T2
    "Length of primal x vector"
    primal_dims::T3
    "Coordinates of z associated to preference slacks"
    preference_slack_dims::T3
    "Coordinates of z associated to interior point slacks"
    interior_point_slack_dims::T4
    "Coordinates of z associated to inequality constraint duals"
    inequality_constraint_dual_dims::T5
    "Coordinates of z associated to equality constraint duals"
    equality_constraint_dual_dims::T6
    "Coordinates of z associated to stationarity duals"
    stationarity_dual_dims::T7
    "Coordinates of z associated to equality or stationarity duals"
    all_equality_stationarity_dual_dims::T8
    "Coordinates of stationarity duals associated with innermost preferences"
    innermost_stationarity_dual_dims::T9
    "KKT dimension"
    kkt_dimension::T10
    "Length of z vector"
    variable_dimension::T10
    "F in symbolics"
    F_symbolic::T11
    "z in symbolics"
    z_symbolic::T11
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
        # generated rows change, and Scholtes solves took different paths in about 1 process
        # in 8 (2026-10-24). The canonical order, also used by SymbolicUtils' own code
        # generator, makes the generated code identical in every process (checked in 48).
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
scalar parameter: the relaxation, and the regularization η where a construction still has one.
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
Default `fd_codegen_chunk_size` of the complete and Scholtes systems. The FastDifferentiation
code generator otherwise emits each block as one function, whose compile time grows faster
than linearly with its size (robotic arm: first solve 44 s unchunked, about 18 s at 32).
`nothing` restores a single function per block.
"""
const FD_CODEGEN_CHUNK_SIZE = 32

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
    compile(e) =
        if codegen === :fast_differentiation
            _build_fd_codegen_function(e, z, θ, ρ; chunk_size = fd_codegen_chunk_size)
        else
            SymbolicTracingUtils.build_function(e, z, θ, ρ; in_place = true, backend_options)
        end
    _f! = compile(exprs)
    _v! = isempty(values) ? ((out, _...) -> out) : compile(values)
    f! = (out, z, θ, ρ) -> (_f!(out, z, θ, ρ); out)
    J! = (J, z, θ, ρ) -> (_v!(SparseArrays.nonzeros(J), z, θ, ρ); J)
    return f!, J!, buffer, (rows, cols)
end

function BuildGOOPKKTSystem(
    F_symbolic::Vector{T},
    z_symbolic::Vector{T},
    θ_symbolic::Vector{T},
    ϵ_symbolic::T,
    η_symbolic::T,
    primal_dims,
    preference_slack_dims,
    interior_point_slack_dims,
    inequality_constraint_dual_dims;
    equality_constraint_dual_dims = Int[],
    stationarity_dual_dims = Int[],
    all_equality_stationarity_dual_dims = Int[],
    innermost_stationarity_dual_dims = Int[],
    backend_options = (;),
    codegen = :native,
    fd_codegen_chunk_size = nothing,
) where {T<:Union{SymbolicTracingUtils.FD.Node,SymbolicTracingUtils.Symbolics.Num}}
    codegen in (:native, :fast_differentiation) ||
        error("Unknown codegen option: $(codegen). Use :native or :fast_differentiation.")
    # The FD tracing backend already generates code through FD.
    use_fd_codegen =
        codegen === :fast_differentiation && T === SymbolicTracingUtils.Symbolics.Num
    isnothing(fd_codegen_chunk_size) ||
        use_fd_codegen ||
        error(
            "fd_codegen_chunk_size requires SymbolicsBackend with " *
            "codegen = :fast_differentiation.",
        )
    @timeit TO "KKT executable construction" begin
        if T == SymbolicTracingUtils.FD.Node
            # FD.make_function only accepts vector-valued symbolic inputs, so the
            # scalar ϵ and η are wrapped in 1-element vectors here. The generated
            # function flattens its runtime arguments, so callers still pass scalars.
            ϵ_arg = [ϵ_symbolic]
            η_arg = [η_symbolic]
        else
            @assert T === SymbolicTracingUtils.Symbolics.Num
            ϵ_arg = ϵ_symbolic
            η_arg = η_symbolic
        end

        F! = @timeit TO "residual function build" let
            _F! = if use_fd_codegen
                _build_fd_codegen_function(
                    F_symbolic,
                    z_symbolic,
                    θ_symbolic,
                    ϵ_symbolic,
                    η_symbolic,
                    chunk_size = fd_codegen_chunk_size,
                )
            else
                SymbolicTracingUtils.build_function(
                    F_symbolic,
                    z_symbolic,
                    θ_symbolic,
                    ϵ_arg,
                    η_arg;
                    in_place = true,
                    backend_options,
                )
            end

            (result, z; θ, ϵ, η) -> _F!(result, z, θ, ϵ, η)
        end

        ∇F_z! = @timeit TO "Jacobian function construction" let
            ∇F_symbolic = @timeit TO "Jacobian symbolic construction" begin
                if T === SymbolicTracingUtils.Symbolics.Num
                    _build_symbolics_sparse_jacobian(F_symbolic, z_symbolic)
                else
                    SymbolicTracingUtils.sparse_jacobian(F_symbolic, z_symbolic)
                end
            end
            _∇F! = @timeit TO "Jacobian function build" if use_fd_codegen
                # Generate code for the structural nonzeros only; the wrapper
                # writes them straight into the sparse buffer, whose nzval
                # ordering matches findnz order for a CSC matrix.
                _Jvals! = _build_fd_codegen_function(
                    SparseArrays.findnz(∇F_symbolic)[3],
                    z_symbolic,
                    θ_symbolic,
                    ϵ_symbolic,
                    η_symbolic,
                    chunk_size = fd_codegen_chunk_size,
                )
                (result, z, θ, ϵ, η) ->
                    _Jvals!(SparseArrays.nonzeros(result), z, θ, ϵ, η)
            else
                SymbolicTracingUtils.build_function(
                    ∇F_symbolic,
                    z_symbolic,
                    θ_symbolic,
                    ϵ_arg,
                    η_arg;
                    in_place = true,
                    backend_options,
                )
            end

            @timeit TO "Jacobian sparsity metadata" begin
                rows, cols, _ = SparseArrays.findnz(∇F_symbolic)
                constant_entries =
                    SymbolicTracingUtils.get_constant_entries(∇F_symbolic, z_symbolic)
                SymbolicTracingUtils.SparseFunction(
                    (result, z; θ, ϵ, η) -> _∇F!(result, z, θ, ϵ, η),
                    rows,
                    cols,
                    size(∇F_symbolic),
                    constant_entries,
                )
            end
        end

        GOOPKKTSystem(
            F!,
            ∇F_z!,
            primal_dims,
            preference_slack_dims,
            interior_point_slack_dims,
            inequality_constraint_dual_dims,
            equality_constraint_dual_dims,
            stationarity_dual_dims,
            all_equality_stationarity_dual_dims,
            innermost_stationarity_dual_dims,
            length(F_symbolic),
            length(z_symbolic),
            F_symbolic,
            z_symbolic,
        )
    end
end

"""
Compile a complete KKT system (`generate_slacked_complete_kkt_system`) and pack it into the
containers the Scholtes engine reads: a `ScholtesKKTSystem` without a complementarity block
(a, b), wrapped in a `CompleteKKTSystem` with the symbolic data. Until one `GOOPKKTSystem`
holds every system. The exact system has no relaxation: it is evaluated at ρ = 0 whatever ρ
it is given.
"""
function BuildCompleteKKTSystem(
    F_symbolic::Vector{T},
    z_symbolic::Vector{T},
    θ_symbolic::Vector{T},
    ρ_symbolic::T,
    primal_dims,
    preference_slack_dims,
    interior_point_slack_dims,
    inequality_constraint_dual_dims;
    equality_constraint_dual_dims,
    complementarity,
    G_symbolic,
    slack_expressions,
    levels,
    backend_options = (;),
    codegen = :fast_differentiation,
    fd_codegen_chunk_size = FD_CODEGEN_CHUNK_SIZE,
) where {T<:Symbolics.Num}
    exact = complementarity === :exact
    f!, J!, J_buffer, J_pattern = _compile_kkt_block(
        F_symbolic,
        z_symbolic,
        θ_symbolic,
        ρ_symbolic;
        codegen,
        fd_codegen_chunk_size,
        backend_options,
    )
    F! = (out, z, θ, ρ) -> f!(out, z, θ, exact ? 0.0 : ρ)
    ∇F! = (out, z, θ, ρ) -> J!(out, z, θ, exact ? 0.0 : ρ)
    no_f!, no_J!, no_buffer, no_pattern = _compile_kkt_block(
        T[],
        z_symbolic,
        θ_symbolic,
        ρ_symbolic;
        codegen,
        fd_codegen_chunk_size,
        backend_options,
    )

    n, m, p = length(z_symbolic), length(F_symbolic), length(θ_symbolic)
    # The evaluation state (θ, ρ, Jacobian memo) starts at ρ = 0 (exact) or 1e-9 (perturbed).
    system = ScholtesKKTSystem(
        n, m, 0, primal_dims, preference_slack_dims, Int[], p,
        F!, no_f!, no_f!, ∇F!, no_J!, no_J!, J_buffer, no_buffer, copy(no_buffer),
        J_pattern, no_pattern, no_pattern, Int[], 0, NamedTuple[], z_symbolic,
        zeros(p), Ref(exact ? 0.0 : 1e-9), zeros(n), Ref(false),
    )
    nonnegative_dims = [
        collect(preference_slack_dims)
        interior_point_slack_dims
        inequality_constraint_dual_dims
    ]
    CompleteKKTSystem(
        system,
        (;
            formulation = complementarity,
            complete_residual! = f!,
            F_symbolic,
            G_symbolic,
            z_symbolic,
            levels,
            slack_expressions,
            nonnegative_dims,
            interior_point_slack_dims,
            inequality_constraint_dual_dims,
            equality_constraint_dual_dims,
            kkt_dimension = m,
            variable_dimension = n,
        ),
    )
end
