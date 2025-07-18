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

struct GOOPKKTSystem{T1,T2,T3,T4,T5}
    "A callable function that computes F!(val, z; θ, ϵ) in-place for 'val'"
    F!::T1
    "A callable function that computes ∇F!(val, z; θ, ϵ) in-place for 'val'"
    ∇F_z!::T2
    "Coordinates of z associated to preference slacks"
    preference_slack_dims::T3
    "Coordinates of z associated to interior point slacks"
    interior_point_slack_dims::T4
    "Coordinates of z associated to inequality constraint duals"
    inequality_constraint_dual_dims::T5
end

function GOOPKKTSystem(
    F::Vector{Symbolics.Num},
    z_symbolic::Vector{Symbolics.Num},
    θ_symbolic::Vector{Symbolics.Num},
    preference_slack_dims,
    interior_point_slack_dims,
    inequality_constraint_dual_dims;
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
