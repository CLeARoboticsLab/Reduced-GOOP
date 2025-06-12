""" Store key elements of the primal-dual KKT system for a MCP composed of
functions G(.) and H(.) such that
                             0 = G(x, y; θ)
                             0 ≤ H(x, y; θ) ⟂ y ≥ 0.

The primal-dual system arises when we introduce slack variable `s` and set
                             G(x, y; θ)     = 0
                             H(x, y; θ) - s = 0
                             s ⦿ y - ϵ      = 0
for some ϵ > 0. Define the function `F(x, y, s; θ, ϵ)` to return the left
hand side of this system of equations.
"""

struct PrimalDualSys{T1, T2}
	"A callable function that computes F!(val, x, λ; θ, μ) in-place for 'val'"
	F!::T1
	"A callable function that computes ∇F!(val, x, λ; θ, μ) in-place for 'val'"
	∇F_z!::T2
	"Dimension of unconstrained variable."
	unconstrained_dimension::Int
	"Dimension of constrained variable."
	constrained_dimension::Int
	"Game dimensions"
	dims::NamedTuple
end

"Helper function to create a 'PrimalDualSys' object from K_symbolic, z_symbolic and bounds"
function PrimalDualSys(
	K_symbolic::Vector{Symbolics.Num},
	z_symbolic::Vector{Symbolics.Num},
	θ_symbolic::Vector{Symbolics.Num},
	lower_bounds::Vector,
	upper_bounds::Vector,
    dims::NamedTuple;
	backend_options = (;),
)
	# All upper bounds are Inf, and lower bounds are either -Inf or 0.
	@assert all(isinf.(upper_bounds)) && all(isinf.(lower_bounds) .|| lower_bounds .== 0)

	unconstrained_indices = findall(isinf, lower_bounds)
	constrained_indices = findall(!isinf, lower_bounds)

	G_symbolic = K_symbolic[unconstrained_indices]
	H_symbolic = K_symbolic[constrained_indices]
	x_symbolic = z_symbolic[unconstrained_indices]
	y_symbolic = z_symbolic[constrained_indices]

	PrimalDualSys(
		G_symbolic,
		H_symbolic,
		x_symbolic,
		y_symbolic,
		θ_symbolic,
		dims;
		backend_options
	)
end


function PrimalDualSys(
	G_symbolic::Vector{Symbolics.Num},
	H_symbolic::Vector{Symbolics.Num},
	x_symbolic::Vector{Symbolics.Num},
	y_symbolic::Vector{Symbolics.Num},
	θ_symbolic::Vector{Symbolics.Num},
	dims::NamedTuple;
	backend_options = (;),
)
	# F!(val, x, y, s; θ) computes the primal-dual system equation in-place.
	# ∇F!(val, x, y, s; θ) computes the Jacobian of the primal-dual system equation in-place.

	backend = SymbolicTracingUtils.SymbolicsBackend()

	s_symbolic = SymbolicTracingUtils.make_variables(backend, :s, length(y_symbolic))
	ϵ_symbolic = only(SymbolicTracingUtils.make_variables(backend, :ϵ, 1))
	z_symbolic = [x_symbolic; y_symbolic; s_symbolic]

	F_symbolic = [
		G_symbolic
		H_symbolic - s_symbolic
		s_symbolic .* y_symbolic .- ϵ_symbolic
	]

	F! = let
		_F! = SymbolicTracingUtils.build_function(
			F_symbolic,
			x_symbolic,
			y_symbolic,
			s_symbolic,
			θ_symbolic,
			ϵ_symbolic;
			in_place = true,
			backend_options,
		)

		(result, x, y, s; θ, ϵ) -> _F!(result, x, y, s, θ, ϵ)
	end

	∇F_z! = let
		∇F_symbolic = SymbolicTracingUtils.sparse_jacobian(F_symbolic, z_symbolic)
		_∇F! = SymbolicTracingUtils.build_function(
			∇F_symbolic,
			x_symbolic,
			y_symbolic,
			s_symbolic,
			θ_symbolic,
			ϵ_symbolic;
			in_place = true,
			backend_options,
		)

		rows, cols, _ = SparseArrays.findnz(∇F_symbolic)
		constant_entries =
			SymbolicTracingUtils.get_constant_entries(∇F_symbolic, z_symbolic)
		SymbolicTracingUtils.SparseFunction(
			(result, x, y, s; θ, ϵ) -> _∇F!(result, x, y, s, θ, ϵ),
			rows,
			cols,
			size(∇F_symbolic),
			constant_entries,
		)
	end

	PrimalDualSys(F!, ∇F_z!, length(x_symbolic), length(y_symbolic), dims) # return struct 
end
