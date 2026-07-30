module ProjectionDiagnostic

using LinearAlgebra
using SparseArrays

export project_row_null

raw"""
    project_row_null(Jd, d; kwargs...)

Split a selected dual vector `d` into a component represented by the row
space of the sparse Jacobian block `Jd` and its complement. The implementation
uses SuiteSparse SPQR least-squares solves; it never forms a dense projector,
`Jd * Jd'`, or a full singular-value decomposition.

The exact, unregularized projection solves

```math
u = \arg\min_u \|J_d^\mathsf{T}u-d\|_2,\qquad
d_\mathrm{row}=J_d^\mathsf{T}u.
```

Keyword arguments:

- `scale_mode = :unscaled`: use the ordinary Euclidean coordinate metric.
- `scale_mode = :scale_aware`: use `coordinate_scales`, or
  `max.(1, abs.(d))` when they are omitted. With `S = Diagonal(scales)`, the
  projection is performed on `q = S⁻¹d` with Jacobian `Jd*S`, then mapped back
  to the original coordinates. This matches the relative-unit convention used
  by the selective warm-start sensitivity diagnostics.
- `rank_rtol = 1e-10`, `rank_atol = 0`: SPQR's absolute rank threshold is
  `max(rank_atol, rank_rtol * reference_scale)`, where `reference_scale` is
  the largest 2-norm of a scaled Jacobian column.
- `regularization_rtol = 0`, `regularization_atol = 0`: optional Tikhonov
  regularization is
  `max(regularization_atol, regularization_rtol * reference_scale)`. A
  positive value solves the augmented sparse least-squares problem
  `min_u ||Jd' * u - d||² + regularization² * ||u||²`. In this case the
  returned `null_component` is the complement of a filtered row-space
  component, not an exact algebraic-nullspace vector; `null_action` and
  `idempotence` quantify that distinction.
- `ordering = :colamd`: deterministic fill-reducing SPQR ordering for fixed
  input and SuiteSparse version. `:default`, `:fixed`, `:natural`, `:amd`,
  and `:bestamd` are also supported and recorded in the result.

The returned named tuple contains the original-coordinate row and null
components, numerical rank and nullity, Euclidean and scale-aware energies,
orthogonality, projection idempotence, null-action leakage, thresholds, and
matrix metadata. Inputs are copied before scaling and are never mutated.
"""
function project_row_null(
    Jd::SparseMatrixCSC{<:Real, <:Integer},
    d::AbstractVector{<:Real};
    scale_mode::Symbol = :unscaled,
    coordinate_scales::Union{Nothing, AbstractVector{<:Real}} = nothing,
    rank_rtol::Real = 1e-10,
    rank_atol::Real = 0.0,
    regularization_rtol::Real = 0.0,
    regularization_atol::Real = 0.0,
    ordering::Symbol = :colamd,
)
    residual_rows, selected_columns = size(Jd)
    selected_columns > 0 ||
        throw(ArgumentError("Jd must have at least one selected dual column."))
    residual_rows > 0 ||
        throw(ArgumentError("Jd must have at least one residual row."))
    length(d) == selected_columns || throw(
        DimensionMismatch(
            "d has length $(length(d)); Jd has $(selected_columns) columns.",
        ),
    )

    _validate_threshold("rank_rtol", rank_rtol)
    _validate_threshold("rank_atol", rank_atol)
    _validate_threshold("regularization_rtol", regularization_rtol)
    _validate_threshold("regularization_atol", regularization_atol)

    d_values = Float64.(d)
    all(isfinite, d_values) ||
        throw(ArgumentError("d must contain only finite values."))
    J_values = SparseMatrixCSC{Float64, Int}(Jd)
    all(isfinite, nonzeros(J_values)) ||
        throw(ArgumentError("Jd must contain only finite stored values."))

    normalized_scale_mode, scales = _coordinate_scales(
        scale_mode,
        coordinate_scales,
        d_values,
    )
    scaled_jacobian = _scale_sparse_columns(J_values, scales)
    metric_vector = d_values ./ scales

    reference_scale = _maximum_column_norm(scaled_jacobian)
    rank_threshold = max(
        Float64(rank_atol),
        Float64(rank_rtol) * reference_scale,
    )
    regularization = max(
        Float64(regularization_atol),
        Float64(regularization_rtol) * reference_scale,
    )
    ordering_code = _spqr_ordering(ordering)

    # A's range is the row space of the scaled Jacobian.
    A = sparse(transpose(scaled_jacobian))
    rank_factor = qr(A; tol = rank_threshold, ordering = ordering_code)
    numerical_rank = rank(rank_factor)

    apply_projection = if numerical_rank == 0
        vector -> zeros(Float64, selected_columns)
    elseif regularization == 0.0
        vector -> Vector{Float64}(A * (rank_factor \ vector))
    else
        regularizer = spdiagm(
            0 => fill(regularization, residual_rows),
        )
        augmented = vcat(A, regularizer)
        regularized_factor =
            qr(augmented; tol = 0.0, ordering = ordering_code)
        zero_tail = zeros(Float64, residual_rows)
        vector -> Vector{Float64}(
            A * (regularized_factor \ vcat(vector, zero_tail)),
        )
    end

    metric_row = apply_projection(metric_vector)
    metric_null = metric_vector .- metric_row
    row_component = scales .* metric_row
    null_component = d_values .- row_component

    projected_again = apply_projection(metric_row)
    metric_idempotence_absolute = norm(projected_again .- metric_row)
    metric_idempotence_relative = _stable_ratio(
        metric_idempotence_absolute,
        norm(metric_row),
        norm(metric_vector),
    )
    euclidean_projected_again = scales .* projected_again
    euclidean_idempotence_absolute =
        norm(euclidean_projected_again .- row_component)
    euclidean_idempotence_relative = _stable_ratio(
        euclidean_idempotence_absolute,
        norm(row_component),
        norm(d_values),
    )

    input_action = norm(J_values * d_values)
    null_action_absolute = norm(J_values * null_component)
    null_action_relative = _stable_ratio(
        null_action_absolute,
        input_action,
        reference_scale * norm(metric_vector),
    )

    euclidean_energy =
        _energy_diagnostics(d_values, row_component, null_component)
    metric_energy =
        _energy_diagnostics(metric_vector, metric_row, metric_null)

    return (
        row_component = row_component,
        null_component = null_component,
        rank = numerical_rank,
        nullity = selected_columns - numerical_rank,
        regularized = regularization > 0.0,
        dimensions = (
            residual_rows = residual_rows,
            selected_columns = selected_columns,
            structural_nonzeros = nnz(J_values),
        ),
        scale_mode = normalized_scale_mode,
        coordinate_scales = scales,
        ordering = ordering,
        thresholds = (
            reference_scale = reference_scale,
            rank_rtol = Float64(rank_rtol),
            rank_atol = Float64(rank_atol),
            rank_threshold = rank_threshold,
            regularization_rtol = Float64(regularization_rtol),
            regularization_atol = Float64(regularization_atol),
            regularization = regularization,
        ),
        energies = (
            euclidean = euclidean_energy,
            metric = metric_energy,
        ),
        orthogonality = (
            euclidean_absolute =
                abs(dot(row_component, null_component)),
            euclidean_cosine =
                _orthogonality_cosine(row_component, null_component),
            metric_absolute = abs(dot(metric_row, metric_null)),
            metric_cosine =
                _orthogonality_cosine(metric_row, metric_null),
        ),
        idempotence = (
            euclidean_absolute = euclidean_idempotence_absolute,
            euclidean_relative = euclidean_idempotence_relative,
            metric_absolute = metric_idempotence_absolute,
            metric_relative = metric_idempotence_relative,
        ),
        null_action = (
            absolute = null_action_absolute,
            relative_to_input = null_action_relative,
        ),
    )
end

function _validate_threshold(name, value)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("$(name) must be finite and nonnegative; got $(value)."))
    nothing
end

function _coordinate_scales(scale_mode, coordinate_scales, d)
    normalized_mode =
        scale_mode === :relative ? :scale_aware : scale_mode
    normalized_mode in (:unscaled, :scale_aware) || throw(
        ArgumentError(
            "scale_mode must be :unscaled or :scale_aware; got $(scale_mode).",
        ),
    )

    if normalized_mode === :unscaled
        isnothing(coordinate_scales) || throw(
            ArgumentError(
                "coordinate_scales are only valid with scale_mode=:scale_aware.",
            ),
        )
        return normalized_mode, ones(Float64, length(d))
    end

    scales =
        isnothing(coordinate_scales) ?
        max.(1.0, abs.(d)) :
        Float64.(coordinate_scales)
    length(scales) == length(d) || throw(
        DimensionMismatch(
            "coordinate_scales has length $(length(scales)); expected $(length(d)).",
        ),
    )
    all(value -> isfinite(value) && value > 0.0, scales) || throw(
        ArgumentError(
            "coordinate_scales must contain only finite, strictly positive values.",
        ),
    )
    normalized_mode, scales
end

function _scale_sparse_columns(J, scales)
    scaled = copy(J)
    values = nonzeros(scaled)
    for column in axes(scaled, 2)
        scale = scales[column]
        for index in nzrange(scaled, column)
            values[index] *= scale
        end
    end
    scaled
end

function _maximum_column_norm(J)
    maximum(
        (
            sqrt(sum(abs2, view(nonzeros(J), nzrange(J, column)))) for
            column in axes(J, 2)
        );
        init = 0.0,
    )
end

function _spqr_ordering(ordering)
    ordering === :default && return SparseArrays.SPQR.ORDERING_DEFAULT
    ordering === :fixed && return SparseArrays.SPQR.ORDERING_FIXED
    ordering === :natural && return SparseArrays.SPQR.ORDERING_NATURAL
    ordering === :amd && return SparseArrays.SPQR.ORDERING_AMD
    ordering === :colamd && return SparseArrays.SPQR.ORDERING_COLAMD
    ordering === :bestamd && return SparseArrays.SPQR.ORDERING_BESTAMD
    throw(
        ArgumentError(
            "Unsupported SPQR ordering $(ordering); use :default, :fixed, " *
            ":natural, :amd, :colamd, or :bestamd.",
        ),
    )
end

function _energy_diagnostics(total, row, null)
    total_energy = sum(abs2, total)
    row_energy = sum(abs2, row)
    null_energy = sum(abs2, null)
    cross_energy = dot(row, null)
    closure_absolute =
        abs(total_energy - row_energy - null_energy - 2 * cross_energy)
    denominator = max(total_energy, eps(Float64))
    (
        total = total_energy,
        row = row_energy,
        null = null_energy,
        cross = cross_energy,
        row_fraction = row_energy / denominator,
        null_fraction = null_energy / denominator,
        closure_relative = closure_absolute / denominator,
    )
end

function _orthogonality_cosine(a, b)
    denominator = norm(a) * norm(b)
    denominator == 0.0 && return 0.0
    abs(dot(a, b)) / denominator
end

function _stable_ratio(numerator, denominator, reference)
    scale = max(
        abs(Float64(denominator)),
        eps(Float64) * max(1.0, abs(Float64(reference))),
    )
    Float64(numerator) / scale
end

end
