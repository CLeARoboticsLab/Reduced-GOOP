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
) where {T <: Union{SymbolicTracingUtils.FD.Node, SymbolicTracingUtils.Symbolics.Num}}
	if T == SymbolicTracingUtils.FD.Node
		backend = SymbolicTracingUtils.FastDifferentiationBackend()
	else
		@assert T === SymbolicTracingUtils.Symbolics.Num
		backend = SymbolicTracingUtils.SymbolicsBackend()
	end

	F! = let
		_F! = SymbolicTracingUtils.build_function(
			F_symbolic,
			z_symbolic,
			θ_symbolic,
			ϵ_symbolic,
			η_symbolic;
			in_place = true,
			backend_options,
		)

		(result, z; θ, ϵ, η) -> _F!(result, z, θ, ϵ, η)
	end

	∇F_z! = let
		∇F_symbolic = SymbolicTracingUtils.sparse_jacobian(F_symbolic, z_symbolic)
		_∇F! = SymbolicTracingUtils.build_function(
			∇F_symbolic,
			z_symbolic,
			θ_symbolic,
			ϵ_symbolic,
			η_symbolic;
			in_place = true,
			backend_options,
		)

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
