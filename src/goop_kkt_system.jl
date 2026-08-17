""" Maintain callable functions to compute the KKT error for a GOOP problem
and its Jacobian with respect to all the (primal and dual) decision variables.
Also keeps track of several types of variables (preference slacks, interior
point slacks, and inequality constraint duals) that must satisfy nonnegativity
constraints that are not explicitly included in the KKT system.

The KKT system is parameterized by a vector θ and a scalar ϵ > 0 which
controls the complementarity relaxation of the interior point scheme.

TODO (@Jingqi/@DongHo): Please flesh this comment out with more of the math,
or with a pointer to a LaTeX derivation somewhere in this repository.
"""

struct GOOPKKTSystem{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13}
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
    "In-place evaluator for the innermost prioritized-preference inequalities"
    innermost_preference!::T12
    "Sparse Jacobian evaluator for the innermost preferences with respect to z"
    ∇innermost_preference_z!::T13
    "Number of innermost prioritized-preference inequalities"
    innermost_preference_dimension::T10
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
        args = map(
            arg -> _symbolics_to_fd(arg, varmap, memo),
            Symbolics.SymbolicUtils.arguments(x),
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

Returns an in-place function `f!(result, z, θ, ϵ, η)`.
"""
function _build_fd_codegen_function(
    exprs::AbstractVector{<:Symbolics.Num},
    z_symbolic,
    θ_symbolic,
    ϵ_symbolic,
    η_symbolic,
    ;
    chunk_size = nothing,
)
    isnothing(chunk_size) ||
        chunk_size > 0 ||
        throw(ArgumentError("FastDifferentiation codegen chunk_size must be positive."))
    fd_z = SymbolicTracingUtils.FD.make_variables(:z, length(z_symbolic))
    fd_θ = SymbolicTracingUtils.FD.make_variables(:θ, length(θ_symbolic))
    fd_ϵ = SymbolicTracingUtils.FD.make_variables(:ϵ, 1)
    fd_η = SymbolicTracingUtils.FD.make_variables(:η, 1)

    varmap = Dict{Any,Any}()
    for (s, f) in zip(
        Iterators.flatten((z_symbolic, θ_symbolic, (ϵ_symbolic, η_symbolic))),
        Iterators.flatten((fd_z, fd_θ, fd_ϵ, fd_η)),
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
    fd_inputs = vcat(fd_z, fd_θ, fd_ϵ, fd_η)
    if isnothing(chunk_size) || length(fd_exprs) <= chunk_size
        _f! = SymbolicTracingUtils.FD.make_function(fd_exprs, fd_inputs; in_place = true)
        return (result, z, θ, ϵ, η) -> _f!(result, vcat(z, θ, ϵ, η))
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
    function (result, z, θ, ϵ, η)
        inputs = vcat(z, θ, ϵ, η)
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
    innermost_preference_symbolic = nothing,
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

        innermost_preference!, ∇innermost_preference_z!, innermost_preference_dimension =
            if isnothing(innermost_preference_symbolic)
                nothing, nothing, 0
            else
                preference! = @timeit TO "innermost preference function build" let
                    _preference! = if use_fd_codegen
                        _build_fd_codegen_function(
                            innermost_preference_symbolic,
                            z_symbolic,
                            θ_symbolic,
                            ϵ_symbolic,
                            η_symbolic,
                            chunk_size = fd_codegen_chunk_size,
                        )
                    else
                        SymbolicTracingUtils.build_function(
                            innermost_preference_symbolic,
                            z_symbolic,
                            θ_symbolic,
                            ϵ_arg,
                            η_arg;
                            in_place = true,
                            backend_options,
                        )
                    end
                    (result, z; θ, ϵ, η) -> _preference!(result, z, θ, ϵ, η)
                end

                preference_jacobian! =
                    @timeit TO "innermost preference Jacobian function construction" let
                        preference_jacobian_symbolic =
                            @timeit TO "innermost preference Jacobian symbolic construction" begin
                                if T === SymbolicTracingUtils.Symbolics.Num
                                    _build_symbolics_sparse_jacobian(
                                        innermost_preference_symbolic,
                                        z_symbolic,
                                    )
                                else
                                    SymbolicTracingUtils.sparse_jacobian(
                                        innermost_preference_symbolic,
                                        z_symbolic,
                                    )
                                end
                            end
                        _preference_jacobian! =
                            @timeit TO "innermost preference Jacobian function build" if use_fd_codegen
                                _Jvals! = _build_fd_codegen_function(
                                    SparseArrays.findnz(preference_jacobian_symbolic)[3],
                                    z_symbolic,
                                    θ_symbolic,
                                    ϵ_symbolic,
                                    η_symbolic,
                                    chunk_size = fd_codegen_chunk_size,
                                )
                                (result, z, θ, ϵ, η) -> _Jvals!(
                                    SparseArrays.nonzeros(result),
                                    z,
                                    θ,
                                    ϵ,
                                    η,
                                )
                            else
                                SymbolicTracingUtils.build_function(
                                    preference_jacobian_symbolic,
                                    z_symbolic,
                                    θ_symbolic,
                                    ϵ_arg,
                                    η_arg;
                                    in_place = true,
                                    backend_options,
                                )
                            end

                        rows, cols, _ = SparseArrays.findnz(preference_jacobian_symbolic)
                        constant_entries = SymbolicTracingUtils.get_constant_entries(
                            preference_jacobian_symbolic,
                            z_symbolic,
                        )
                        SymbolicTracingUtils.SparseFunction(
                            (result, z; θ, ϵ, η) ->
                                _preference_jacobian!(result, z, θ, ϵ, η),
                            rows,
                            cols,
                            size(preference_jacobian_symbolic),
                            constant_entries,
                        )
                    end
                preference!, preference_jacobian!, length(innermost_preference_symbolic)
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
            innermost_preference!,
            ∇innermost_preference_z!,
            innermost_preference_dimension,
        )
    end
end
