module ScalingBenchmark

using BlockArrays: Block, BlockArray
using JLD2: JLD2
using LinearAlgebra: norm, opnorm, svd
using Printf: @sprintf
using ReducedGOOP

export run_scaling_benchmark, validate_scaling_benchmark

const SCHEMA_VERSION = 1
const FIXTURE_ID = "affine_two_level_semantic_transport_t20_dt0p1_v1"
const PLANNING_HORIZON = 20
const DELTA_T = 0.1
const BARRIER_EPSILON = 0.1
const STRICT_REFERENCE_TOL = 1e-8
const VALIDATION_TOL = 1e-10
const RANK_RTOL = 1e-10
const POSITIVE_EPSILONS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
const EPSILONS = (0.0, POSITIVE_EPSILONS...)
const FORMULATIONS = (:reduced, :quasi)
const TRANSPORTS = (
    :identity_copy,
    :stage_shift_zero_tail,
    :stage_shift_hold_tail,
    :diagnostic_oracle_terminal_completion,
)
const SEMANTIC_TRANSPORTS = (
    :stage_shift_zero_tail,
    :stage_shift_hold_tail,
    :diagnostic_oracle_terminal_completion,
)
const RESIDUAL_REGIONS = (:interior, :initial, :terminal, :global)
const SLOPE_METRICS = (
    "total_residual_norm2",
    "interior_residual_norm2",
    "initial_residual_norm2",
    "terminal_residual_norm2",
    "global_residual_norm2",
    "distance_to_solution_manifold",
    "terminal_completion_error_norm2",
)

_trajectory_coordinate(variable::Symbol, stage::Integer) =
    2 * (stage - 1) + (variable === :control ? 2 : 1)

function _equality_coordinate(equation_type::Symbol, stage)
    equation_type === :initial_state && return 1
    equation_type === :dynamics && return 1 + something(stage)
    throw(
        ArgumentError(
            "Unsupported benchmark equality coordinate $(equation_type), stage $(stage).",
        ),
    )
end

function _trajectory_vector(states, controls)
    length(states) == length(controls) ||
        throw(DimensionMismatch("State and control horizons must match."))
    vec(permutedims(hcat(states, controls)))
end

function _atomic_write(path::AbstractString, text::AbstractString)
    mkpath(dirname(path))
    temporary = path * ".tmp." * string(getpid()) * "." * string(time_ns())
    open(temporary, "w") do io
        write(io, text)
        flush(io)
    end
    mv(temporary, path; force = true)
    path
end

function _atomic_save(path::AbstractString, object)
    mkpath(dirname(path))
    temporary = path * ".tmp." * string(getpid()) * "." * string(time_ns())
    JLD2.save_object(temporary, object)
    mv(temporary, path; force = true)
    path
end

function _csv_escape(value)
    if value === nothing || value === missing
        return ""
    elseif value isa AbstractFloat
        return isfinite(value) ? repr(value) :
               isnan(value) ? "NaN" :
               value > 0 ? "Inf" : "-Inf"
    end
    text = value isa Symbol ? String(value) : string(value)
    text = replace(text, '\r' => "\\r", '\n' => "\\n")
    if occursin(',', text) || occursin('"', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    text
end

function _write_csv(path, rows; columns = nothing)
    rows = collect(rows)
    if isnothing(columns)
        names = Set{String}()
        for row in rows, name in keys(row)
            push!(names, string(name))
        end
        columns = sort!(collect(names))
        if "case_id" in columns
            columns = vcat(
                ["case_id"],
                filter(name -> name != "case_id", columns),
            )
        end
    else
        columns = String.(columns)
    end

    io = IOBuffer()
    println(io, join(columns, ","))
    for row in rows
        println(
            io,
            join((_csv_escape(get(row, name, "")) for name in columns), ","),
        )
    end
    _atomic_write(path, String(take!(io)))
end

function _epsilon_label(epsilon::Real)
    iszero(epsilon) && return "0"
    replace(replace(@sprintf("%.0e", epsilon), "+" => ""), "-" => "m")
end

function _build_dynamics_matrix()
    T = PLANNING_HORIZON
    primal_dimension = 2 * T
    equality_dimension = T
    A = zeros(Float64, equality_dimension, primal_dimension)
    A[1, _trajectory_coordinate(:state, 1)] = 1.0
    for stage in 1:(T-1)
        row = 1 + stage
        A[row, _trajectory_coordinate(:state, stage + 1)] = 1.0
        A[row, _trajectory_coordinate(:state, stage)] = -1.0
        A[row, _trajectory_coordinate(:control, stage)] = -DELTA_T
    end
    A
end

function _semantic_layout()
    T = PLANNING_HORIZON
    primal = ReducedGOOP.PrimalCoordinateSpec[]
    for stage in 1:T
        push!(
            primal,
            ReducedGOOP.PrimalCoordinateSpec(;
                player = 1,
                variable = :state,
                stage,
                component = 1,
                shift_rule = :successor,
                tail_role =
                    stage == T ? :dynamic_completion : :shifted,
            ),
        )
        push!(
            primal,
            ReducedGOOP.PrimalCoordinateSpec(;
                player = 1,
                variable = :control,
                stage,
                component = 1,
                shift_rule = :successor,
                tail_role =
                    stage == T ? :zero_completion : :shifted,
            ),
        )
    end

    equality = ReducedGOOP.EqualityCoordinateSpec[
        ReducedGOOP.EqualityCoordinateSpec(;
            scope = :player,
            player = 1,
            equation_class = :initial_condition,
            equation_type = :initial_state,
            stage = 1,
            component = 1,
            shift_rule = :reset,
            tail_role = :initial_condition,
        ),
    ]
    for stage in 1:(T-1)
        push!(
            equality,
            ReducedGOOP.EqualityCoordinateSpec(;
                scope = :player,
                player = 1,
                equation_class = :dynamics,
                equation_type = :dynamics,
                stage,
                successor_stage = stage + 1,
                component = 1,
                shift_rule = :successor,
                tail_role =
                    stage == T - 1 ? :terminal_dynamics : :shifted,
            ),
        )
    end
    ReducedGOOP.GOOPSemanticLayout(;
        primal_by_player = [primal],
        equality_by_player = [equality],
    )
end

function _build_problem()
    T = PLANNING_HORIZON
    primal_dimension = 2 * T
    parameter_dimension = 1 + 2 * primal_dimension
    primal_template =
        BlockArray(zeros(Float64, primal_dimension), [primal_dimension])
    parameter_template =
        BlockArray(zeros(Float64, parameter_dimension), [parameter_dimension])

    function unpack_targets(parameters)
        block = parameters[Block(1)]
        outer_target = @view block[2:(1+primal_dimension)]
        inner_target =
            @view block[(2+primal_dimension):(1+2*primal_dimension)]
        (; outer_target, inner_target)
    end

    outer_objective = function (primals, parameters)
        target = unpack_targets(parameters).outer_target
        0.5 * sum(abs2, primals[Block(1)] .- target)
    end
    inner_objective = function (primals, parameters)
        target = unpack_targets(parameters).inner_target
        0.5 * sum(abs2, primals[Block(1)] .- target)
    end
    equality = function (primals, parameters)
        trajectory = reshape(primals[Block(1)], 2, :)
        initial_state = parameters[Block(1)][1]
        vcat(
            trajectory[1, 1] - initial_state,
            [
                trajectory[1, stage+1] -
                trajectory[1, stage] -
                DELTA_T * trajectory[2, stage] for
                stage in 1:(T-1)
            ],
        )
    end

    ReducedGOOP.ParametricGOOP(
        primal_template,
        parameter_template;
        preferences = [[outer_objective, inner_objective]],
        is_prioritized_constraint = [[false, false]],
        equality_constraints = [equality],
        inequality_constraints = [nothing],
        shared_equality_constraint = nothing,
        shared_inequality_constraint = nothing,
        semantic_layout = _semantic_layout(),
    )
end

function _build_kkt(problem, formulation::Symbol)
    generator =
        formulation === :reduced ?
        ReducedGOOP.generate_slacked_reduced_kkt_system :
        formulation === :quasi ?
        ReducedGOOP.generate_slacked_quasi_kkt_system :
        throw(ArgumentError("Unknown formulation $(formulation)."))
    generator(
        problem;
        backend = ReducedGOOP.SymbolicTracingUtils.SymbolicsBackend(),
        backend_options = (;),
        codegen = :fast_differentiation,
        fd_codegen_chunk_size = 128,
    )
end

function _source_data(A)
    T = PLANNING_HORIZON
    states = zeros(Float64, T)
    controls = [
        0.15 + 0.015 * stage + 0.08 * sin(0.47 * stage) for
        stage in 1:T
    ]
    states[1] = 0.3
    for stage in 1:(T-1)
        states[stage+1] =
            states[stage] + DELTA_T * controls[stage]
    end
    primal = _trajectory_vector(states, controls)
    lambda_inner = vcat(
        0.4,
        [
            0.2 + 0.05 * stage + 0.15 * sin(0.7 * stage) for
            stage in 1:(T-1)
        ],
    )
    lambda_outer = zeros(Float64, T)
    psi = _trajectory_vector(
        [
            0.3 * cos(0.4 * stage) + 0.02 * stage for
            stage in 1:T
        ],
        [
            -0.2 * sin(0.3 * stage) + 0.01 * stage for
            stage in 1:T
        ],
    )
    inner_target = primal - transpose(A) * lambda_inner
    outer_target = primal - psi - transpose(A) * lambda_outer
    parameters = vcat(states[1], outer_target, inner_target)
    (;
        primal,
        lambda_outer,
        lambda_inner,
        psi,
        parameters,
    )
end

function _shifted_destination(source, A)
    T = PLANNING_HORIZON
    source_trajectory = reshape(source.primal, 2, T)
    destination_trajectory = zeros(Float64, 2, T)
    destination_trajectory[:, 1:(T-1)] =
        source_trajectory[:, 2:T]
    destination_trajectory[1, T] =
        source_trajectory[1, T] +
        DELTA_T * source_trajectory[2, T]
    destination_trajectory[2, T] = 0.0
    primal = vec(destination_trajectory)

    lambda_outer = zeros(Float64, T)
    lambda_inner = zeros(Float64, T)
    lambda_inner[2:(T-1)] = source.lambda_inner[3:T]
    lambda_inner[T] = source.lambda_inner[T] + 0.37

    source_psi = reshape(source.psi, 2, T)
    destination_psi = zeros(Float64, 2, T)
    destination_psi[:, 1:(T-1)] = source_psi[:, 2:T]
    destination_psi[:, T] =
        [source_psi[1, T] + 0.31, -0.27]
    psi = vec(destination_psi)

    inner_target = primal - transpose(A) * lambda_inner
    outer_target = primal - psi - transpose(A) * lambda_outer
    parameters = vcat(primal[1], outer_target, inner_target)
    (;
        primal,
        lambda_outer,
        lambda_inner,
        psi,
        parameters,
    )
end

function _reference_direction(A)
    T = PLANNING_HORIZON
    state_direction = zeros(Float64, T)
    control_direction = [
        0.12 * cos(0.31 * stage) - 0.04 * sin(0.17 * stage) for
        stage in 1:T
    ]
    state_direction[1] = 0.35
    for stage in 1:(T-1)
        state_direction[stage+1] =
            state_direction[stage] +
            DELTA_T * control_direction[stage]
    end
    primal = _trajectory_vector(state_direction, control_direction)
    lambda_outer = zeros(Float64, T)
    lambda_inner = vcat(
        0.22,
        [
            0.09 * cos(0.23 * stage) + 0.035 * stage / T for
            stage in 1:(T-1)
        ],
    )
    psi = _trajectory_vector(
        [
            0.08 * sin(0.27 * stage) + 0.015 * stage / T for
            stage in 1:T
        ],
        [
            -0.07 * cos(0.19 * stage) + 0.01 * stage / T for
            stage in 1:T
        ],
    )
    outer_target =
        primal - psi - transpose(A) * lambda_outer
    inner_target =
        primal - transpose(A) * lambda_inner
    parameter = vcat(primal[1], outer_target, inner_target)
    scale = norm(parameter)
    scale > 0.0 || error("The deterministic parameter direction vanished.")
    (;
        primal = primal ./ scale,
        lambda_outer = lambda_outer,
        lambda_inner = lambda_inner ./ scale,
        psi = psi ./ scale,
        parameter = parameter ./ scale,
    )
end

function _assemble_kkt_point(
    kkt,
    primal,
    lambda_outer,
    lambda_inner,
    psi,
)
    point = zeros(Float64, kkt.variable_dimension)
    covered = falses(kkt.variable_dimension)
    for record in ReducedGOOP.kkt_variable_metadata(kkt)
        value = if record.family === :primal
            primal[
                _trajectory_coordinate(
                    something(record.primal_variable),
                    something(record.stage),
                )
            ]
        elseif record.family === :equality_multiplier
            multipliers =
                record.owner_level == 1 ? lambda_outer :
                record.owner_level == 2 ? lambda_inner :
                throw(
                    ArgumentError(
                        "Unexpected equality-multiplier level $(record.owner_level).",
                    ),
                )
            multipliers[
                _equality_coordinate(
                    record.equation_type,
                    record.stage,
                )
            ]
        elseif record.family === :stationarity_multiplier
            record.owner_level == 1 && record.target_level == 2 ||
                throw(
                    ArgumentError(
                        "Unexpected stationarity multiplier levels " *
                        "$(record.owner_level) -> $(record.target_level).",
                    ),
                )
            psi[
                _trajectory_coordinate(
                    something(record.primal_variable),
                    something(record.stage),
                )
            ]
        else
            throw(
                ArgumentError(
                    "The equality-only benchmark has unsupported KKT family " *
                    "$(record.family).",
                ),
            )
        end
        point[record.index] = value
        covered[record.index] = true
    end
    all(covered) || error(
        "Semantic metadata did not cover benchmark KKT coordinates " *
        "$(findall(!, covered)).",
    )
    point
end

function _residual(kkt, point, parameters)
    values = zeros(Float64, kkt.kkt_dimension)
    kkt.F!(
        values,
        point;
        θ = parameters,
        ϵ = BARRIER_EPSILON,
        η = 0.0,
    )
    values
end

function _jacobian(kkt, point, parameters)
    buffer = copy(kkt.∇F_z!.result_buffer)
    kkt.∇F_z!(
        buffer,
        point;
        θ = parameters,
        ϵ = BARRIER_EPSILON,
        η = 0.0,
    )
    Matrix{Float64}(buffer)
end

function _equation_region(record)
    if record.family === :equality_feasibility
        record.equation_class === :initial_condition && return :initial
        record.equation_class === :global_non_time_indexed && return :global
        isnothing(record.stage) && return :global
        record.stage == PLANNING_HORIZON - 1 && return :terminal
        return :interior
    elseif record.family === :stationarity
        isnothing(record.stage) && return :global
        if record.stage == 1 && record.primal_variable === :state
            return :initial
        elseif record.stage >= PLANNING_HORIZON - 1
            return :terminal
        end
        return :interior
    end
    :global
end

function _is_terminal_completion(record)
    record.family in (:equality_multiplier, :stationarity_multiplier) ||
        return false
    record.tail_role in (
        :terminal_dynamics,
        :terminal_path,
        :dynamic_completion,
        :zero_completion,
    )
end

function _is_initial_completion(record)
    record.family === :equality_multiplier &&
        record.equation_class === :initial_condition
end

function _dual_region(record)
    record.family in (:equality_multiplier, :stationarity_multiplier) ||
        return :not_dual
    _is_initial_completion(record) && return :initial
    _is_terminal_completion(record) && return :terminal
    record.equation_class === :global_non_time_indexed && return :global
    isnothing(record.stage) && return :global
    :interior
end

function _region_indices(equation_metadata)
    Dict(
        region => [
            record.row for record in equation_metadata if
            _equation_region(record) === region
        ] for region in RESIDUAL_REGIONS
    )
end

function _dual_region_indices(variable_metadata)
    Dict(
        region => [
            record.index for record in variable_metadata if
            _dual_region(record) === region
        ] for region in RESIDUAL_REGIONS
    )
end

function _numerical_factorization(jacobian)
    factor = svd(jacobian; full = false)
    largest = isempty(factor.S) ? 0.0 : maximum(factor.S)
    threshold = max(
        RANK_RTOL * largest,
        max(size(jacobian)...) * eps(Float64) * largest,
    )
    numerical_rank = count(value -> value > threshold, factor.S)
    numerical_rank > 0 ||
        error("The benchmark KKT Jacobian has zero numerical rank.")
    smallest_retained = factor.S[numerical_rank]
    (;
        factor,
        numerical_rank,
        threshold,
        largest,
        smallest_retained,
        pseudoinverse_norm = inv(smallest_retained),
        nullity = size(jacobian, 2) - numerical_rank,
    )
end

function _minimum_norm_correction(factorization, residual)
    rank_value = factorization.numerical_rank
    factor = factorization.factor
    coefficients =
        (transpose(factor.U[:, 1:rank_value]) * residual) ./
        factor.S[1:rank_value]
    transpose(factor.Vt[1:rank_value, :]) * coefficients
end

function _reference_at(fixture, epsilon)
    base = fixture.destination
    direction = fixture.direction
    primal = base.primal .+ epsilon .* direction.primal
    lambda_outer =
        base.lambda_outer .+ epsilon .* direction.lambda_outer
    lambda_inner =
        base.lambda_inner .+ epsilon .* direction.lambda_inner
    psi = base.psi .+ epsilon .* direction.psi
    parameters =
        base.parameters .+ epsilon .* direction.parameter
    point = _assemble_kkt_point(
        fixture.kkt,
        primal,
        lambda_outer,
        lambda_inner,
        psi,
    )
    residual = _residual(fixture.kkt, point, parameters)
    residual_norm = norm(residual)
    residual_norm <= STRICT_REFERENCE_TOL || error(
        "Strict $(fixture.formulation) reference at epsilon=$(epsilon) has " *
        "residual $(residual_norm), exceeding $(STRICT_REFERENCE_TOL).",
    )
    (;
        primal,
        lambda_outer,
        lambda_inner,
        psi,
        parameters,
        point,
        residual,
        residual_norm,
    )
end

function _base_candidates(fixture)
    kkt = fixture.kkt
    source = fixture.source_point
    shifted_primal = fixture.destination.primal
    Dict{Symbol, Vector{Float64}}(
        :identity_copy =>
            ReducedGOOP.build_selective_warmstart(
                shifted_primal,
                source,
                kkt,
                :all_duals;
                dual_transport = :identity_copy,
            ),
        :stage_shift_zero_tail =>
            ReducedGOOP.build_selective_warmstart(
                shifted_primal,
                source,
                kkt,
                :all_duals;
                dual_transport = :stage_shift_zero_tail,
            ),
        :stage_shift_hold_tail =>
            ReducedGOOP.build_selective_warmstart(
                shifted_primal,
                source,
                kkt,
                :all_duals;
                dual_transport = :stage_shift_hold_tail,
            ),
    )
end

function _candidate_points(fixture, reference)
    candidates = Dict(
        name => copy(point) for
        (name, point) in fixture.base_candidates
    )
    oracle = copy(candidates[:stage_shift_zero_tail])
    oracle[fixture.terminal_variable_indices] =
        reference.point[fixture.terminal_variable_indices]
    candidates[:diagnostic_oracle_terminal_completion] = oracle
    candidates
end

function _build_fixture(formulation::Symbol)
    A = _build_dynamics_matrix()
    problem = _build_problem()
    kkt = _build_kkt(problem, formulation)
    variable_metadata =
        sort(ReducedGOOP.kkt_variable_metadata(kkt); by = record -> record.index)
    equation_metadata =
        sort(ReducedGOOP.kkt_equation_metadata(kkt); by = record -> record.row)
    source = _source_data(A)
    destination = _shifted_destination(source, A)
    direction = _reference_direction(A)
    abs(norm(direction.parameter) - 1.0) <= 100 * eps(Float64) ||
        error("The benchmark parameter direction is not unit normalized.")

    source_point = _assemble_kkt_point(
        kkt,
        source.primal,
        source.lambda_outer,
        source.lambda_inner,
        source.psi,
    )
    source_residual = _residual(kkt, source_point, source.parameters)
    norm(source_residual) <= STRICT_REFERENCE_TOL ||
        error("The exact source reference failed its residual check.")

    destination_point = _assemble_kkt_point(
        kkt,
        destination.primal,
        destination.lambda_outer,
        destination.lambda_inner,
        destination.psi,
    )
    destination_residual =
        _residual(kkt, destination_point, destination.parameters)
    norm(destination_residual) <= STRICT_REFERENCE_TOL ||
        error("The exact destination reference failed its residual check.")

    direction_point = _assemble_kkt_point(
        kkt,
        direction.primal,
        direction.lambda_outer,
        direction.lambda_inner,
        direction.psi,
    )
    jacobian = _jacobian(kkt, destination_point, destination.parameters)
    factorization = _numerical_factorization(jacobian)
    factorization.numerical_rank == kkt.kkt_dimension || error(
        "Expected a full-row-rank affine KKT system; got rank " *
        "$(factorization.numerical_rank) for $(kkt.kkt_dimension) rows.",
    )

    terminal_variable_indices = [
        record.index for record in variable_metadata if
        _is_terminal_completion(record)
    ]
    initial_variable_indices = [
        record.index for record in variable_metadata if
        _is_initial_completion(record)
    ]
    isempty(terminal_variable_indices) &&
        error("No terminal-completion coordinates were identified.")
    region_indices = _region_indices(equation_metadata)
    dual_region_indices = _dual_region_indices(variable_metadata)
    base_candidates = _base_candidates((;
        kkt,
        source_point,
        destination,
    ))
    maximum_candidate_jacobian_drift = maximum(
        norm(
            _jacobian(kkt, point, destination.parameters) -
            jacobian,
        ) for point in values(base_candidates)
    )
    maximum_candidate_jacobian_drift <= VALIDATION_TOL || error(
        "The benchmark Jacobian is not affine over the transported candidates.",
    )

    zero_at_anchor = base_candidates[:stage_shift_zero_tail]
    nonterminal_direction = copy(direction_point)
    nonterminal_direction[terminal_variable_indices] .= 0.0
    data_constant = norm(jacobian * nonterminal_direction)
    terminal_constant = opnorm(jacobian[:, terminal_variable_indices])

    fixture = (;
        formulation,
        A,
        problem,
        kkt,
        source,
        destination,
        direction,
        source_point,
        destination_point,
        direction_point,
        jacobian,
        factorization,
        variable_metadata,
        equation_metadata,
        terminal_variable_indices,
        initial_variable_indices,
        region_indices,
        dual_region_indices,
        base_candidates,
        data_constant,
        terminal_constant,
        maximum_candidate_jacobian_drift,
    )

    anchor_reference = _reference_at(fixture, 0.0)
    nonterminal = setdiff(
        collect(1:kkt.variable_dimension),
        terminal_variable_indices,
    )
    norm(zero_at_anchor[nonterminal] - anchor_reference.point[nonterminal]) <=
        VALIDATION_TOL || error(
        "Zero-tail semantic transport disagrees with the exact anchor away " *
        "from terminal completion coordinates.",
    )
    fixture
end

function _region_norms(residual, indices)
    Dict(
        region =>
            isempty(indices[region]) ? 0.0 :
            norm(view(residual, indices[region])) for
        region in RESIDUAL_REGIONS
    )
end

function _dual_error_norms(error, indices)
    Dict(
        region =>
            isempty(indices[region]) ? 0.0 :
            norm(view(error, indices[region])) for
        region in RESIDUAL_REGIONS
    )
end

function _evaluate_candidate(fixture, epsilon, transport, point, reference)
    residual = _residual(fixture.kkt, point, reference.parameters)
    region_norms = _region_norms(residual, fixture.region_indices)
    variable_error = point - reference.point
    dual_error_norms =
        _dual_error_norms(variable_error, fixture.dual_region_indices)
    terminal_error =
        norm(view(variable_error, fixture.terminal_variable_indices))
    initial_error =
        norm(view(variable_error, fixture.initial_variable_indices))

    correction =
        _minimum_norm_correction(fixture.factorization, residual)
    projected_point = point - correction
    projected_residual =
        _residual(fixture.kkt, projected_point, reference.parameters)
    reference_gauge_distance = norm(variable_error)

    semantic_bound =
        fixture.data_constant * epsilon +
        fixture.terminal_constant * terminal_error +
        reference.residual_norm
    is_semantic = transport in SEMANTIC_TRANSPORTS
    bound_violation =
        is_semantic ? max(0.0, norm(residual) - semantic_bound) : NaN

    row = Dict{String, Any}(
        "case_id" =>
            "$(fixture.formulation)__epsilon_$(_epsilon_label(epsilon))__$(transport)",
        "fixture_id" => FIXTURE_ID,
        "optimizer_invoked" => false,
        "reference_construction" => :analytic_affine,
        "formulation" => fixture.formulation,
        "epsilon" => epsilon,
        "epsilon_anchor" => iszero(epsilon),
        "transport" => transport,
        "production_policy" =>
            transport !== :diagnostic_oracle_terminal_completion,
        "online_available" =>
            transport !== :diagnostic_oracle_terminal_completion,
        "reference_residual_norm2" => reference.residual_norm,
        "total_residual_norm2" => norm(residual),
        "total_residual_norm_inf" => norm(residual, Inf),
        "interior_residual_norm2" => region_norms[:interior],
        "initial_residual_norm2" => region_norms[:initial],
        "terminal_residual_norm2" => region_norms[:terminal],
        "global_residual_norm2" => region_norms[:global],
        "interior_dual_error_norm2" => dual_error_norms[:interior],
        "initial_dual_error_norm2" => dual_error_norms[:initial],
        "terminal_dual_error_norm2" => dual_error_norms[:terminal],
        "global_dual_error_norm2" => dual_error_norms[:global],
        "terminal_completion_error_norm2" => terminal_error,
        "initial_completion_error_norm2" => initial_error,
        "reference_gauge_distance" => reference_gauge_distance,
        "distance_to_solution_manifold" => norm(correction),
        "projected_solution_manifold_residual_norm2" =>
            norm(projected_residual),
        "semantic_bound_applicable" => is_semantic,
        "bound_data_constant" => fixture.data_constant,
        "bound_terminal_constant" => fixture.terminal_constant,
        "semantic_bound_rhs" => is_semantic ? semantic_bound : NaN,
        "semantic_bound_ratio" =>
            is_semantic && semantic_bound > 0.0 ?
            norm(residual) / semantic_bound : NaN,
        "semantic_bound_violation" => bound_violation,
    )
    (;
        row,
        residual,
        point,
        variable_error,
        correction,
    )
end

function _checkpoint_path(run_dir, formulation, epsilon)
    joinpath(
        run_dir,
        "checkpoints",
        "scaling_benchmark",
        String(formulation),
        "epsilon_$(_epsilon_label(epsilon)).jld2",
    )
end

function _compute_checkpoint(fixture, epsilon)
    reference = _reference_at(fixture, epsilon)
    candidates = _candidate_points(fixture, reference)
    results = Dict{String, Any}()
    for transport in TRANSPORTS
        evaluated = _evaluate_candidate(
            fixture,
            epsilon,
            transport,
            candidates[transport],
            reference,
        )
        results[String(transport)] = Dict{String, Any}(
            "row" => evaluated.row,
            "residual" => evaluated.residual,
            "point" => evaluated.point,
            "variable_error" => evaluated.variable_error,
            "correction" => evaluated.correction,
        )
    end

    jacobian = _jacobian(
        fixture.kkt,
        reference.point,
        reference.parameters,
    )
    jacobian_drift = norm(jacobian - fixture.jacobian)
    factorization = _numerical_factorization(jacobian)
    factorization.numerical_rank ==
        fixture.factorization.numerical_rank || error(
        "Jacobian rank changed at epsilon=$(epsilon).",
    )
    jacobian_drift <= VALIDATION_TOL || error(
        "The affine benchmark Jacobian drifted by $(jacobian_drift).",
    )

    Dict{String, Any}(
        "schema_version" => SCHEMA_VERSION,
        "fixture_id" => FIXTURE_ID,
        "formulation" => String(fixture.formulation),
        "epsilon" => epsilon,
        "reference_parameters" => reference.parameters,
        "reference_point" => reference.point,
        "reference_residual" => reference.residual,
        "reference_residual_norm2" => reference.residual_norm,
        "jacobian_drift_norm2" => jacobian_drift,
        "rank" => factorization.numerical_rank,
        "nullity" => factorization.nullity,
        "rank_threshold" => factorization.threshold,
        "largest_singular_value" => factorization.largest,
        "smallest_retained_singular_value" =>
            factorization.smallest_retained,
        "pseudoinverse_norm2" => factorization.pseudoinverse_norm,
        "results" => results,
    )
end

function _validate_checkpoint(checkpoint, fixture, epsilon)
    get(checkpoint, "schema_version", nothing) == SCHEMA_VERSION ||
        error("Scaling checkpoint schema mismatch.")
    get(checkpoint, "fixture_id", nothing) == FIXTURE_ID ||
        error("Scaling checkpoint fixture mismatch.")
    get(checkpoint, "formulation", nothing) ==
        String(fixture.formulation) ||
        error("Scaling checkpoint formulation mismatch.")
    get(checkpoint, "epsilon", nothing) == epsilon ||
        error("Scaling checkpoint epsilon mismatch.")

    reference = _reference_at(fixture, epsilon)
    checkpoint["reference_parameters"] == reference.parameters ||
        error("Scaling checkpoint parameter data drifted.")
    checkpoint["reference_point"] == reference.point ||
        error("Scaling checkpoint exact reference drifted.")
    Set(keys(checkpoint["results"])) == Set(String.(TRANSPORTS)) ||
        error("Scaling checkpoint transport set is incomplete.")
    checkpoint
end

function _load_or_compute_checkpoint(run_dir, fixture, epsilon)
    path = _checkpoint_path(run_dir, fixture.formulation, epsilon)
    if isfile(path)
        return _validate_checkpoint(
            JLD2.load_object(path),
            fixture,
            epsilon,
        )
    end
    checkpoint = _compute_checkpoint(fixture, epsilon)
    _atomic_save(path, checkpoint)
    checkpoint
end

function _metadata_rows(fixture)
    [
        Dict{String, Any}(
            "case_id" => "$(fixture.formulation)__row_$(record.row)",
            "fixture_id" => FIXTURE_ID,
            "formulation" => fixture.formulation,
            "row" => record.row,
            "equation_family" => record.family,
            "player" => something(record.player, ""),
            "preference_level" => something(record.level, ""),
            "physical_stage" => something(record.stage, ""),
            "region" => _equation_region(record),
            "exact_shift_invariant" =>
                _equation_region(record) === :interior,
            "equation_class" => record.equation_class,
            "equation_type" => record.equation_type,
            "primal_variable" =>
                something(record.primal_variable, ""),
            "component" => record.component,
        ) for record in fixture.equation_metadata
    ]
end

function _coordinate_rows(fixture, checkpoint)
    rows = Dict{String, Any}[]
    epsilon = checkpoint["epsilon"]
    for transport in TRANSPORTS
        result = checkpoint["results"][String(transport)]
        residual = result["residual"]
        for record in fixture.equation_metadata
            push!(
                rows,
                Dict{String, Any}(
                    "case_id" =>
                        "$(fixture.formulation)__epsilon_$(_epsilon_label(epsilon))__$(transport)__row$(record.row)",
                    "fixture_id" => FIXTURE_ID,
                    "formulation" => fixture.formulation,
                    "epsilon" => epsilon,
                    "transport" => transport,
                    "row" => record.row,
                    "residual" => residual[record.row],
                    "absolute_residual" => abs(residual[record.row]),
                    "equation_family" => record.family,
                    "player" => something(record.player, ""),
                    "preference_level" => something(record.level, ""),
                    "physical_stage" => something(record.stage, ""),
                    "region" => _equation_region(record),
                    "exact_shift_invariant" =>
                        _equation_region(record) === :interior,
                    "equation_class" => record.equation_class,
                    "equation_type" => record.equation_type,
                    "primal_variable" =>
                        something(record.primal_variable, ""),
                    "component" => record.component,
                ),
            )
        end
    end
    rows
end

function _rank_row(fixture, checkpoint)
    Dict{String, Any}(
        "case_id" =>
            "$(fixture.formulation)__epsilon_$(_epsilon_label(checkpoint["epsilon"]))",
        "fixture_id" => FIXTURE_ID,
        "formulation" => fixture.formulation,
        "epsilon" => checkpoint["epsilon"],
        "kkt_rows" => fixture.kkt.kkt_dimension,
        "kkt_variables" => fixture.kkt.variable_dimension,
        "rank" => checkpoint["rank"],
        "nullity" => checkpoint["nullity"],
        "rank_rtol" => RANK_RTOL,
        "rank_threshold" => checkpoint["rank_threshold"],
        "largest_singular_value" => checkpoint["largest_singular_value"],
        "smallest_retained_singular_value" =>
            checkpoint["smallest_retained_singular_value"],
        "pseudoinverse_norm2" => checkpoint["pseudoinverse_norm2"],
        "jacobian_drift_norm2" => checkpoint["jacobian_drift_norm2"],
        "reference_residual_norm2" =>
            checkpoint["reference_residual_norm2"],
    )
end

function _linear_fit(x, y)
    length(x) >= 2 || return nothing
    x_mean = sum(x) / length(x)
    y_mean = sum(y) / length(y)
    denominator = sum((value - x_mean)^2 for value in x)
    denominator > 0.0 || return nothing
    slope =
        sum(
            (x_value - x_mean) * (y_value - y_mean) for
            (x_value, y_value) in zip(x, y)
        ) / denominator
    intercept = y_mean - slope * x_mean
    total = sum((value - y_mean)^2 for value in y)
    error = sum(
        (
            y_value -
            (intercept + slope * x_value)
        )^2 for (x_value, y_value) in zip(x, y)
    )
    r_squared = total == 0.0 ? NaN : 1.0 - error / total
    (; slope, intercept, r_squared)
end

function _fit_log_slope(rows, metric)
    values = [
        (
            epsilon = Float64(row["epsilon"]),
            value = Float64(row[metric]),
        ) for row in rows if
        Float64(row["epsilon"]) > 0.0 &&
        isfinite(Float64(row[metric])) &&
        Float64(row[metric]) > 0.0
    ]
    sort!(values; by = item -> item.epsilon)
    primary = first(values, min(3, length(values)))

    function fit(items)
        length(items) >= 2 || return nothing
        _linear_fit(
            log10.([item.epsilon for item in items]),
            log10.([item.value for item in items]),
        )
    end
    (; values, primary, primary_fit = fit(primary), all_fit = fit(values))
end

function _slope_rows(result_rows)
    grouped = Dict{Tuple{String, String}, Vector{Dict{String, Any}}}()
    for row in result_rows
        key = (string(row["formulation"]), string(row["transport"]))
        push!(get!(grouped, key, Dict{String, Any}[]), row)
    end
    rows = Dict{String, Any}[]
    for ((formulation, transport), group) in sort!(collect(grouped); by = first)
        for metric in SLOPE_METRICS
            fit = _fit_log_slope(group, metric)
            push!(
                rows,
                Dict{String, Any}(
                    "case_id" =>
                        "$(formulation)__$(transport)__$(metric)",
                    "fixture_id" => FIXTURE_ID,
                    "formulation" => formulation,
                    "transport" => transport,
                    "metric" => metric,
                    "available_points" => length(fit.values),
                    "primary_points" => length(fit.primary),
                    "primary_epsilon_min" =>
                        isempty(fit.primary) ? NaN :
                        first(fit.primary).epsilon,
                    "primary_epsilon_max" =>
                        isempty(fit.primary) ? NaN :
                        last(fit.primary).epsilon,
                    "primary_slope" =>
                        isnothing(fit.primary_fit) ? NaN :
                        fit.primary_fit.slope,
                    "primary_intercept" =>
                        isnothing(fit.primary_fit) ? NaN :
                        fit.primary_fit.intercept,
                    "primary_r_squared" =>
                        isnothing(fit.primary_fit) ? NaN :
                        fit.primary_fit.r_squared,
                    "all_points_slope" =>
                        isnothing(fit.all_fit) ? NaN :
                        fit.all_fit.slope,
                    "all_points_intercept" =>
                        isnothing(fit.all_fit) ? NaN :
                        fit.all_fit.intercept,
                    "all_points_r_squared" =>
                        isnothing(fit.all_fit) ? NaN :
                        fit.all_fit.r_squared,
                    "fit_performed" => !isnothing(fit.primary_fit),
                ),
            )
        end
    end
    rows
end

function _constant_row(fixture, result_rows)
    anchor_identity = only(
        filter(
            row ->
                row["transport"] == :identity_copy &&
                iszero(row["epsilon"]),
            result_rows,
        ),
    )
    semantic_rows = filter(
        row -> row["transport"] in SEMANTIC_TRANSPORTS,
        result_rows,
    )
    Dict{String, Any}(
        "case_id" => String(fixture.formulation),
        "fixture_id" => FIXTURE_ID,
        "formulation" => fixture.formulation,
        "planning_horizon" => PLANNING_HORIZON,
        "dt" => DELTA_T,
        "strict_reference_tolerance" => STRICT_REFERENCE_TOL,
        "optimizer_invoked" => false,
        "reference_construction" => :analytic_affine,
        "data_direction_norm2" => norm(fixture.direction.parameter),
        "bound_data_constant" => fixture.data_constant,
        "bound_terminal_constant" => fixture.terminal_constant,
        "jacobian_operator_norm2" => opnorm(fixture.jacobian),
        "rank" => fixture.factorization.numerical_rank,
        "nullity" => fixture.factorization.nullity,
        "pseudoinverse_norm2" =>
            fixture.factorization.pseudoinverse_norm,
        "terminal_completion_coordinates" =>
            length(fixture.terminal_variable_indices),
        "initial_completion_coordinates" =>
            length(fixture.initial_variable_indices),
        "fixed_index_interior_limit_norm2" =>
            anchor_identity["interior_residual_norm2"],
        "fixed_index_total_limit_norm2" =>
            anchor_identity["total_residual_norm2"],
        "maximum_semantic_bound_violation" => maximum(
            Float64(row["semantic_bound_violation"]) for
            row in semantic_rows
        ),
        "maximum_reference_residual_norm2" => maximum(
            Float64(row["reference_residual_norm2"]) for
            row in result_rows
        ),
        "maximum_projected_solution_manifold_residual_norm2" => maximum(
            Float64(
                row["projected_solution_manifold_residual_norm2"],
            ) for row in result_rows
        ),
        "maximum_candidate_jacobian_drift_norm2" =>
            fixture.maximum_candidate_jacobian_drift,
    )
end

function _comparison_rows(result_rows)
    index = Dict(
        (
            string(row["formulation"]),
            Float64(row["epsilon"]),
            string(row["transport"]),
        ) => row for row in result_rows
    )
    rows = Dict{String, Any}[]
    for epsilon in EPSILONS, transport in TRANSPORTS
        reduced =
            index[("reduced", Float64(epsilon), String(transport))]
        quasi =
            index[("quasi", Float64(epsilon), String(transport))]
        fields = (
            "total_residual_norm2",
            "interior_residual_norm2",
            "initial_residual_norm2",
            "terminal_residual_norm2",
            "global_residual_norm2",
            "distance_to_solution_manifold",
        )
        differences = Dict(
            field =>
                abs(Float64(reduced[field]) - Float64(quasi[field])) for
            field in fields
        )
        push!(
            rows,
            Dict{String, Any}(
                "case_id" =>
                    "epsilon_$(_epsilon_label(epsilon))__$(transport)",
                "fixture_id" => FIXTURE_ID,
                "epsilon" => epsilon,
                "transport" => transport,
                "maximum_metric_difference" =>
                    maximum(values(differences)),
                (
                    "$(field)_difference" => differences[field] for
                    field in fields
                )...,
            ),
        )
    end
    rows
end

function _validate_results(
    result_rows,
    slope_rows,
    rank_rows,
    constant_rows,
    comparison_rows,
)
    expected_results =
        length(FORMULATIONS) * length(EPSILONS) * length(TRANSPORTS)
    length(result_rows) == expected_results ||
        error("Expected $(expected_results) benchmark result rows.")
    maximum(
        Float64(row["reference_residual_norm2"]) for row in result_rows
    ) <= STRICT_REFERENCE_TOL ||
        error("A strict benchmark reference exceeded its tolerance.")

    for formulation in FORMULATIONS
        rows = filter(
            row -> Symbol(row["formulation"]) === formulation,
            result_rows,
        )
        anchor(transport) = only(
            filter(
                row ->
                    Symbol(row["transport"]) === transport &&
                    iszero(row["epsilon"]),
                rows,
            ),
        )
        anchor(:identity_copy)["interior_residual_norm2"] > 1e-3 ||
            error("Fixed-index copying has no resolved interior limit.")
        for transport in SEMANTIC_TRANSPORTS
            anchor(transport)["interior_residual_norm2"] <= VALIDATION_TOL ||
                error(
                    "Semantic transport $(transport) is not interior-exact " *
                    "at the zero-perturbation anchor.",
                )
        end
        anchor(:diagnostic_oracle_terminal_completion)[
            "total_residual_norm2"
        ] <= VALIDATION_TOL ||
            error("Oracle completion is not exact at the anchor.")

        form_ranks = [
            Int(row["rank"]) for row in rank_rows if
            Symbol(row["formulation"]) === formulation
        ]
        length(unique(form_ranks)) == 1 ||
            error("The benchmark Jacobian rank is not constant.")
        all(isfinite(Float64(row["pseudoinverse_norm2"])) for row in rank_rows if
            Symbol(row["formulation"]) === formulation) ||
            error("The benchmark pseudoinverse norm is not bounded.")
    end

    function slope(formulation, transport, metric)
        row = only(
            filter(
                row ->
                    Symbol(row["formulation"]) === formulation &&
                    Symbol(row["transport"]) === transport &&
                    row["metric"] == metric,
                slope_rows,
            ),
        )
        Float64(row["primary_slope"])
    end
    for formulation in FORMULATIONS
        for transport in SEMANTIC_TRANSPORTS
            abs(
                slope(
                    formulation,
                    transport,
                    "interior_residual_norm2",
                ) - 1.0,
            ) <= 1e-5 ||
                error(
                    "Semantic interior residual did not scale linearly for " *
                    "$(formulation), $(transport).",
                )
        end
        abs(
            slope(
                formulation,
                :diagnostic_oracle_terminal_completion,
                "total_residual_norm2",
            ) - 1.0,
        ) <= 1e-5 ||
            error("Oracle total residual did not scale linearly.")
        abs(
            slope(
                formulation,
                :identity_copy,
                "interior_residual_norm2",
            ),
        ) <= 0.05 ||
            error("Fixed-index residual does not show an O(1) plateau.")
    end

    maximum(
        Float64(row["maximum_semantic_bound_violation"]) for
        row in constant_rows
    ) <= VALIDATION_TOL ||
        error("The semantic residual bound was violated.")
    maximum(
        Float64(
            row["maximum_projected_solution_manifold_residual_norm2"],
        ) for row in constant_rows
    ) <= STRICT_REFERENCE_TOL ||
        error("The affine solution-manifold projection was inaccurate.")
    maximum(
        Float64(row["maximum_metric_difference"]) for row in comparison_rows
    ) <= VALIDATION_TOL ||
        error("Reduced and quasi affine benchmarks are not comparable.")
    true
end

"""
    run_scaling_benchmark(run_dir)

Run the deterministic theorem-scaling benchmark and write resumable JLD2
checkpoints plus raw CSV tables below `run_dir`. No optimization solver is
invoked: every source and destination KKT reference is constructed
analytically and checked directly against the generated reduced/quasi KKT
residual.

The four compared starts are identity-copy transport, semantic zero-tail,
semantic hold-tail, and a diagnostic oracle that overwrites only terminal
completion coordinates from the exact destination reference. The oracle uses
online-unavailable information and is explicitly marked as diagnostic.
"""
function run_scaling_benchmark(run_dir::AbstractString)
    run_dir = abspath(run_dir)
    mkpath(run_dir)
    result_rows = Dict{String, Any}[]
    coordinate_rows = Dict{String, Any}[]
    metadata_rows = Dict{String, Any}[]
    rank_rows = Dict{String, Any}[]
    constant_rows = Dict{String, Any}[]

    for formulation in FORMULATIONS
        fixture = _build_fixture(formulation)
        append!(metadata_rows, _metadata_rows(fixture))
        form_rows = Dict{String, Any}[]
        for epsilon in EPSILONS
            checkpoint = _load_or_compute_checkpoint(
                run_dir,
                fixture,
                Float64(epsilon),
            )
            for transport in TRANSPORTS
                row = checkpoint["results"][String(transport)]["row"]
                push!(result_rows, row)
                push!(form_rows, row)
            end
            append!(coordinate_rows, _coordinate_rows(fixture, checkpoint))
            push!(rank_rows, _rank_row(fixture, checkpoint))
        end
        push!(constant_rows, _constant_row(fixture, form_rows))
    end

    slope_rows = _slope_rows(result_rows)
    comparison_rows = _comparison_rows(result_rows)
    _validate_results(
        result_rows,
        slope_rows,
        rank_rows,
        constant_rows,
        comparison_rows,
    )

    raw_dir = joinpath(run_dir, "raw")
    _write_csv(joinpath(raw_dir, "scaling_benchmark.csv"), result_rows)
    _write_csv(
        joinpath(raw_dir, "scaling_benchmark_residual_coordinates.csv"),
        coordinate_rows,
    )
    _write_csv(
        joinpath(raw_dir, "scaling_benchmark_metadata.csv"),
        metadata_rows,
    )
    _write_csv(
        joinpath(raw_dir, "scaling_benchmark_rank.csv"),
        rank_rows,
    )
    _write_csv(
        joinpath(raw_dir, "scaling_benchmark_slopes.csv"),
        slope_rows,
    )
    _write_csv(
        joinpath(raw_dir, "scaling_benchmark_constants.csv"),
        constant_rows,
    )
    _write_csv(
        joinpath(raw_dir, "scaling_benchmark_formulation_comparison.csv"),
        comparison_rows,
    )
    _atomic_write(
        joinpath(run_dir, "scaling_benchmark_complete"),
        "ok\n",
    )

    Dict{String, Any}(
        "run_dir" => run_dir,
        "fixture_id" => FIXTURE_ID,
        "optimizer_invoked" => false,
        "formulations" => String.(FORMULATIONS),
        "epsilons" => collect(EPSILONS),
        "transports" => String.(TRANSPORTS),
        "result_rows" => length(result_rows),
        "coordinate_rows" => length(coordinate_rows),
        "metadata_rows" => length(metadata_rows),
        "rank_rows" => length(rank_rows),
        "slope_rows" => length(slope_rows),
        "constant_rows" => length(constant_rows),
        "comparison_rows" => length(comparison_rows),
        "maximum_reference_residual_norm2" => maximum(
            Float64(row["reference_residual_norm2"]) for
            row in result_rows
        ),
        "maximum_semantic_bound_violation" => maximum(
            Float64(row["maximum_semantic_bound_violation"]) for
            row in constant_rows
        ),
        "minimum_fixed_index_interior_limit_norm2" => minimum(
            Float64(row["fixed_index_interior_limit_norm2"]) for
            row in constant_rows
        ),
        "maximum_formulation_difference" => maximum(
            Float64(row["maximum_metric_difference"]) for
            row in comparison_rows
        ),
    )
end

"""
    validate_scaling_benchmark([run_dir])

Execute and validate the optimizer-free scaling benchmark. With no argument,
all artifacts are written to a temporary directory that is removed on return.
Passing `run_dir` keeps the artifacts and exercises checkpoint resumption when
called again.
"""
function validate_scaling_benchmark(run_dir::AbstractString)
    summary = run_scaling_benchmark(run_dir)
    summary["maximum_reference_residual_norm2"] <= STRICT_REFERENCE_TOL ||
        error("Strict-reference validation failed.")
    summary["maximum_semantic_bound_violation"] <= VALIDATION_TOL ||
        error("Semantic-bound validation failed.")
    summary["minimum_fixed_index_interior_limit_norm2"] > 1e-3 ||
        error("Fixed-index inconsistency was not resolved.")
    summary["maximum_formulation_difference"] <= VALIDATION_TOL ||
        error("Reduced/quasi comparability validation failed.")
    summary
end

function validate_scaling_benchmark()
    mktempdir() do temporary
        validate_scaling_benchmark(temporary)
    end
end

end # module ScalingBenchmark
