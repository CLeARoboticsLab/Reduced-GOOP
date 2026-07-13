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

struct GOOPKKTSystem{T1, T2, T3, T4, T5, T6, T7}
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
	"KKT dimension"
	kkt_dimension::T6
	"Length of z vector"
	variable_dimension::T6
	"F in symbolics"
	F_symbolic::T7
	"z in symbolics"
	z_symbolic::T7
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
)
	fd_z = SymbolicTracingUtils.FD.make_variables(:z, length(z_symbolic))
	fd_θ = SymbolicTracingUtils.FD.make_variables(:θ, length(θ_symbolic))
	fd_ϵ = SymbolicTracingUtils.FD.make_variables(:ϵ, 1)
	fd_η = SymbolicTracingUtils.FD.make_variables(:η, 1)

	varmap = Dict{Any, Any}()
	for (s, f) in zip(
		Iterators.flatten((z_symbolic, θ_symbolic, (ϵ_symbolic, η_symbolic))),
		Iterators.flatten((fd_z, fd_θ, fd_ϵ, fd_η)),
	)
		varmap[Symbolics.unwrap(s)] = f
	end

	memo = IdDict{Any, Any}()
	fd_exprs = SymbolicTracingUtils.FD.Node[
		SymbolicTracingUtils.FD.Node(_symbolics_to_fd(expr, varmap, memo)) for
		expr in exprs
	]

	# A single concatenated input vector keeps the generated function's arity
	# in sync with the runtime call below, which flattens its arguments the
	# same way.
	_f! = SymbolicTracingUtils.FD.make_function(
		fd_exprs,
		vcat(fd_z, fd_θ, fd_ϵ, fd_η);
		in_place = true,
	)

	(result, z, θ, ϵ, η) -> _f!(result, vcat(z, θ, ϵ, η))
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
	backend_options = (;),
	codegen = :native,
) where {T <: Union{SymbolicTracingUtils.FD.Node, SymbolicTracingUtils.Symbolics.Num}}
	codegen in (:native, :fast_differentiation) ||
		error("Unknown codegen option: $(codegen). Use :native or :fast_differentiation.")
	# The FD tracing backend already generates code through FD.
	use_fd_codegen =
		codegen === :fast_differentiation && T === SymbolicTracingUtils.Symbolics.Num

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
			∇F_symbolic = @timeit TO "Jacobian symbolic construction" SymbolicTracingUtils.sparse_jacobian(F_symbolic, z_symbolic)
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
			length(F_symbolic),
			length(z_symbolic),
			F_symbolic,
			z_symbolic,
		)
	end
end
