module SolverTraceTests

using ReducedGOOP
using LinearAlgebra: dot, norm
using SparseArrays
using Test

struct ScalarJacobian{F}
    result_buffer::SparseMatrixCSC{Float64,Int}
    derivative::F
end

function (jacobian::ScalarJacobian)(result, z; θ, ϵ, η)
    nonzeros(result)[1] = jacobian.derivative(z[1], θ[1])
    result
end

function scalar_kkt(residual, derivative)
    residual! = function (result, z; θ, ϵ, η)
        result[1] = residual(z[1], θ[1])
        result
    end
    jacobian = ScalarJacobian(sparse([1], [1], [1.0], 1, 1), derivative)
    empty_range = 2:1
    empty_dims = Int[]
    ReducedGOOP.GOOPKKTSystem(
        residual!,
        jacobian,
        1:1,
        empty_range,
        empty_dims,
        empty_dims,
        empty_dims,
        empty_dims,
        empty_dims,
        empty_dims,
        1,
        1,
        Float64[],
        Float64[],
    )
end

struct TwoBlockJacobian
    result_buffer::SparseMatrixCSC{Float64,Int}
end

function (::TwoBlockJacobian)(result, z; θ, ϵ, η)
    nonzeros(result) .= 1.0
    result
end

function positive_block_kkt()
    residual! = function (result, z; θ, ϵ, η)
        @. result = z - θ
        result
    end
    jacobian = TwoBlockJacobian(sparse([1, 2], [1, 2], ones(2), 2, 2))
    empty_dims = Int[]
    ReducedGOOP.GOOPKKTSystem(
        residual!,
        jacobian,
        empty_dims,
        empty_dims,
        1:1,
        2:2,
        empty_dims,
        empty_dims,
        empty_dims,
        empty_dims,
        2,
        2,
        Float64[],
        Float64[],
    )
end

function solver_options(
    linesearch;
    max_inner_iters = 20,
    linear_solver = :svd,
    min_stepsize = 1e-12,
    max_eta_retries = 5,
)
    ReducedGOOP.InteriorPointOptions(;
        tol = 1e-10,
        η₀ = 1e-6,
        ϵ₀ = 0.1,
        max_inner_iters,
        max_outer_iters = 1,
        tightening_rate = 1.2,
        loosening_rate = 3.0,
        min_stepsize,
        linesearch,
        record_convergence = true,
        max_eta_retries,
        linear_solver,
        verbose = false,
    )
end

events_with(events, name) = filter(event -> event.event === name, events)

function test_same_result(untraced, traced)
    @test keys(untraced) == keys(traced)
    for key in keys(untraced)
        @test isequal(getproperty(untraced, key), getproperty(traced, key))
    end
end

function all_scalar_event_fields(events)
    all(events) do event
        all(values(event)) do value
            value isa Number || value isa Symbol || value isa Bool
        end
    end
end

@testset "nonnegative step-size cap" begin
    @test ReducedGOOP._nonnegative_stepsize_cap(Float64[], Float64[]) == 1.0
    @test ReducedGOOP._nonnegative_stepsize_cap([1.0, 2.0], [3.0, 4.0]) == 1.0
    @test ReducedGOOP._nonnegative_stepsize_cap([1.0, 2.0], [-4.0, -1.0]) ==
          0.25
    @test ReducedGOOP._nonnegative_stepsize_cap(
        [1.0, 2.0],
        [-4.0, -1.0];
        max_stepsize = 0.125,
    ) == 0.125
end

@testset "backtracking solver trace is observational" begin
    kkt = scalar_kkt((z, target) -> z^2 - target, (z, _) -> 2z)
    θ = [1.0]
    z₀ = [0.2]
    options = solver_options(:backtracking)

    untraced = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        trace_hook = nothing,
    )
    events = Any[]
    traced = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        trace_hook = event -> push!(events, event),
    )

    test_same_result(untraced, traced)
    @test traced.status === :solved
    @test all_scalar_event_fields(events)

    initial = only(events_with(events, :initial_residual))
    direct_initial = zeros(kkt.kkt_dimension)
    kkt.F!(direct_initial, z₀; θ, ϵ = options.ϵ₀, η = 0.0)
    @test initial.residual_norm2 == norm(direct_initial, 2)
    @test initial.residual_norm_inf == norm(direct_initial, Inf)

    accepted = events_with(events, :accepted_step)
    directions = events_with(events, :direction)
    trials = events_with(events, :line_search_trial)
    finish = only(events_with(events, :finish))
    first_step = first(accepted)
    first_direction = first(directions)
    first_iteration_trials = filter(event -> event.total_iter == 1, trials)
    @test first(first_iteration_trials).alpha == 1.0
    @test first_step.first_attempt_alpha == 1.0
    @test first_step.accepted_alpha == last(first_iteration_trials).alpha
    @test first_step.accepted_alpha < 1.0
    @test first_step.backtracking_count > 0
    @test last(first_iteration_trials).accepted

    initial_z = only(z₀)
    initial_F = initial_z^2 - only(θ)
    initial_J = 2initial_z
    δz = -initial_J * initial_F / (initial_J^2 + first_step.eta_used)
    first_z = initial_z + first_step.accepted_alpha * δz
    @test first_step.residual_after_norm2 ≈ abs(first_z^2 - only(θ))
    @test first_direction.total_iter == 1
    @test first_direction.direction_attempt == 1
    @test first_direction.step_norm2 ≈ abs(δz)
    @test first_direction.step_norm_inf ≈ abs(δz)
    @test first_direction.armijo_slope_raw ≈ initial_F * initial_J * δz
    @test first_direction.armijo_slope ==
          min(first_direction.armijo_slope_raw, 0.0)
    @test first_direction.slack_stepsize_cap == 1.0
    @test first_direction.dual_stepsize_cap == 1.0
    @test first_direction.combined_stepsize_cap == 1.0

    first_trial = first(first_iteration_trials)
    first_trial_linearized = initial_F + first_trial.alpha * initial_J * δz
    first_trial_actual = initial_F^2 - first_trial.residual_norm2^2
    first_trial_predicted = initial_F^2 - first_trial_linearized^2
    @test first_trial.predicted_reduction ≈ first_trial_predicted
    @test first_trial.actual_reduction ≈ first_trial_actual
    @test first_trial.reduction_ratio ≈
          first_trial_actual / first_trial_predicted
    @test first_trial.armijo_margin ≈
          first_trial.armijo_rhs_norm2_squared - first_trial.residual_norm2^2

    accepted_trial = last(first_iteration_trials)
    @test first_step.direction_attempt == accepted_trial.direction_attempt
    @test first_step.step_norm2 == accepted_trial.step_norm2
    @test first_step.predicted_reduction == accepted_trial.predicted_reduction
    @test first_step.actual_reduction == accepted_trial.actual_reduction
    @test first_step.reduction_ratio == accepted_trial.reduction_ratio
    @test first_step.armijo_slope == accepted_trial.armijo_slope
    @test first_step.combined_stepsize_cap == accepted_trial.combined_stepsize_cap

    @test finish.accepted_steps == length(accepted)
    @test finish.total_iters == traced.total_iters
    @test finish.actual_outer_iters == 1
    @test finish.total_backtracking_count ==
          sum(event.backtracking_count for event in accepted)
    @test finish.total_eta_retry_count == sum(event.eta_retry_count for event in accepted)
    @test finish.full_step_count == count(event -> event.full_step, accepted)
    @test finish.full_step_fraction == finish.full_step_count / finish.accepted_steps

    direct_final = zeros(kkt.kkt_dimension)
    kkt.F!(direct_final, traced.z; θ, ϵ = traced.ϵ, η = 0.0)
    @test finish.final_residual_norm2 == norm(direct_final, 2)
    @test finish.final_residual_norm_inf == norm(direct_final, Inf)
    @test isempty(events_with(events, :failure))
end

@testset "fraction-to-boundary trace observes accepted points" begin
    kkt = scalar_kkt((z, target) -> z - target, (_, _) -> 1.0)
    θ = [2.0]
    z₀ = [0.0]
    options = solver_options(:fraction_to_boundary; max_inner_iters = 6)

    untraced = ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀, options)
    events = Any[]
    traced = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        trace_hook = event -> push!(events, event),
    )

    test_same_result(untraced, traced)
    accepted = events_with(events, :accepted_step)
    finish = only(events_with(events, :finish))
    @test first(accepted).linesearch === :fraction_to_boundary
    @test first(accepted).residual_before_norm2 == 2.0
    @test first(accepted).residual_after_norm2 < 1e-5
    @test all(event -> event.accepted_alpha == 1.0, accepted)
    @test finish.full_step_fraction == 1.0
    @test finish.total_backtracking_count == 0
    @test finish.actual_outer_iters == 1
end

@testset "fraction-to-boundary trace counts positive-block backtracking" begin
    kkt = positive_block_kkt()
    θ = [-1.0, -3.0]
    z₀ = [1.0, 1.0]

    partial_options =
        solver_options(:fraction_to_boundary; max_inner_iters = 2)
    untraced = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options = partial_options,
    )
    partial_events = Any[]
    traced = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options = partial_options,
        trace_hook = event -> push!(partial_events, event),
    )

    test_same_result(untraced, traced)
    accepted = only(events_with(partial_events, :accepted_step))
    eta_change = only(filter(
        event ->
            event.event === :eta_change &&
                event.reason === :fraction_partial_step,
        partial_events,
    ))
    finish = only(events_with(partial_events, :finish))
    @test accepted.accepted_alpha_slack == 0.25
    @test accepted.accepted_alpha_dual == 0.125
    @test accepted.accepted_alpha == 0.125
    @test accepted.slack_backtracking_count == 2
    @test accepted.dual_backtracking_count == 3
    @test accepted.backtracking_count == 5
    @test !accepted.full_step
    @test eta_change.eta_before == accepted.eta_used
    @test eta_change.eta_after == accepted.eta_next
    @test finish.total_backtracking_count == 5
    @test finish.full_step_count == 0

    direct_after = zeros(kkt.kkt_dimension)
    kkt.F!(direct_after, traced.z; θ, ϵ = traced.ϵ, η = 0.0)
    @test accepted.residual_after_norm2 == norm(direct_after, 2)
    @test accepted.residual_after_norm_inf == norm(direct_after, Inf)

    failure_options = solver_options(
        :fraction_to_boundary;
        max_inner_iters = 3,
        min_stepsize = 0.75,
    )
    untraced_failure = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options = failure_options,
    )
    failure_events = Any[]
    traced_failure = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options = failure_options,
        trace_hook = event -> push!(failure_events, event),
    )

    test_same_result(untraced_failure, traced_failure)
    failure = only(events_with(failure_events, :failure))
    failure_finish = only(events_with(failure_events, :finish))
    @test traced_failure.status === :failed
    @test failure.reason === :fraction_to_boundary_exhausted
    @test isnan(failure.alpha)
    @test failure.backtracking_count == 2
    @test failure.eta_retry_count == 0
    @test failure_finish.total_backtracking_count == 2
    @test failure_finish.accepted_steps == 0
    @test failure_finish.full_step_fraction == 0.0
    @test isempty(events_with(failure_events, :accepted_step))
    @test isempty(events_with(failure_events, :eta_change))
end

@testset "KLU trace is observational" begin
    kkt = scalar_kkt((z, target) -> z - target, (_, _) -> 1.0)
    θ = [2.0]
    z₀ = [0.0]
    options = solver_options(:backtracking; linear_solver = :klu)

    untraced = ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀, options)
    events = Any[]
    traced = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        trace_hook = event -> push!(events, event),
    )

    test_same_result(untraced, traced)
    finish = only(events_with(events, :finish))
    @test finish.status === :solved
    @test finish.accepted_steps == traced.total_iters
    @test finish.accepted_steps > 0
    @test finish.klu_singular_retries == 0
    @test finish.svd_fallback_count == 0
end

@testset "rich diagnostic snapshots are copy-isolated and sequence-invariant" begin
    kkt = scalar_kkt((z, target) -> z^2 - target, (z, _) -> 2z)
    θ = [1.0]
    z₀ = [0.2]
    options = solver_options(:backtracking; linear_solver = :klu)

    untraced =
        ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀, options)
    basic_events = Any[]
    basic = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        trace_hook = event -> push!(basic_events, event),
    )

    rich_events = Any[]
    direction_summaries = NamedTuple[]
    accepted_summaries = NamedTuple[]
    diagnostic_hook = function (snapshot)
        if snapshot.event === :direction_snapshot
            push!(direction_summaries, (;
                total_iter = snapshot.total_iter,
                direction_attempt = snapshot.direction_attempt,
                eta_retry_count = snapshot.eta_retry_count,
                eta = snapshot.eta,
                z = copy(snapshot.z),
                residual = copy(snapshot.residual),
                jacobian = copy(snapshot.jacobian),
                step = copy(snapshot.step),
                jacobian_step = copy(snapshot.jacobian_step),
                step_norm2 = snapshot.step_norm2,
                armijo_slope = snapshot.armijo_slope,
            ))
            # Deliberately corrupt every callback-owned array. Exact equality with
            # the basic and untraced solves proves these are not live workspaces.
            fill!(snapshot.z, 123.0)
            fill!(snapshot.residual, 124.0)
            fill!(nonzeros(snapshot.jacobian), 125.0)
            fill!(snapshot.step, 126.0)
            fill!(snapshot.jacobian_step, 127.0)
        elseif snapshot.event === :accepted_snapshot
            push!(accepted_summaries, (;
                total_iter = snapshot.total_iter,
                direction_attempt = snapshot.direction_attempt,
                accepted_alpha = snapshot.accepted_alpha,
                z = copy(snapshot.z),
                residual = copy(snapshot.residual),
            ))
            fill!(snapshot.z, 128.0)
            fill!(snapshot.residual, 129.0)
        else
            error("Unexpected diagnostic event $(snapshot.event).")
        end
        nothing
    end
    rich = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        trace_hook = event -> push!(rich_events, event),
        diagnostic_hook,
    )
    diagnostic_only_events = Any[]
    diagnostic_only = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        diagnostic_hook = event -> push!(diagnostic_only_events, event),
    )

    test_same_result(untraced, basic)
    test_same_result(untraced, rich)
    test_same_result(untraced, diagnostic_only)
    @test isequal(basic_events, rich_events)
    @test all_scalar_event_fields(rich_events)
    @test count(event -> event.event === :direction_snapshot, diagnostic_only_events) ==
          length(direction_summaries)
    @test count(event -> event.event === :accepted_snapshot, diagnostic_only_events) ==
          length(accepted_summaries)

    basic_directions = events_with(basic_events, :direction)
    basic_accepted = events_with(basic_events, :accepted_step)
    @test length(direction_summaries) == length(basic_directions)
    @test length(accepted_summaries) == length(basic_accepted)
    for (summary, event) in zip(direction_summaries, basic_directions)
        @test summary.total_iter == event.total_iter
        @test summary.direction_attempt == event.direction_attempt
        @test summary.eta_retry_count == event.eta_retry_count
        @test summary.eta == event.eta
        @test norm(summary.step, 2) == event.step_norm2
        @test norm(summary.residual .+ summary.jacobian_step, 2) ≈
              event.linearized_residual_norm2
        @test dot(summary.residual, summary.jacobian_step) ==
              event.armijo_slope_raw
        @test summary.armijo_slope == event.armijo_slope
        @test summary.jacobian * summary.step ≈ summary.jacobian_step
    end
    for (summary, event) in zip(accepted_summaries, basic_accepted)
        @test summary.total_iter == event.total_iter
        @test summary.direction_attempt == event.direction_attempt
        @test summary.accepted_alpha == event.accepted_alpha
        @test norm(summary.residual, 2) == event.residual_after_norm2
    end
    @test last(accepted_summaries).z == rich.z
end

@testset "trace reports an iteration-budget failure" begin
    kkt = scalar_kkt((z, target) -> z^2 - target, (z, _) -> 2z)
    θ = [1.0]
    events = Any[]
    result = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀ = [0.2],
        options = solver_options(:backtracking; max_inner_iters = 2),
        trace_hook = event -> push!(events, event),
    )

    @test result.status === :failed
    failure = only(events_with(events, :failure))
    finish = only(events_with(events, :finish))
    @test failure.reason === :not_converged
    @test finish.accepted_steps == 1
    @test finish.total_iters == 1
    @test finish.actual_outer_iters == 1
    @test finish.full_step_fraction == 0.0
end

@testset "trace counts exhausted eta retries" begin
    kkt = scalar_kkt((_, _) -> 1.0, (_, _) -> 0.0)
    θ = [1.0]
    z₀ = [0.0]
    options = solver_options(:backtracking; min_stepsize = 0.25, max_eta_retries = 2)

    untraced = ReducedGOOP.solve(ReducedGOOP.InteriorPoint(), kkt, θ; z₀, options)
    events = Any[]
    traced = ReducedGOOP.solve(
        ReducedGOOP.InteriorPoint(),
        kkt,
        θ;
        z₀,
        options,
        trace_hook = event -> push!(events, event),
    )

    test_same_result(untraced, traced)
    failure = only(events_with(events, :failure))
    finish = only(events_with(events, :finish))
    retry_changes = filter(
        event -> event.event === :eta_change && event.reason === :line_search_retry,
        events,
    )
    @test traced.status === :failed
    @test failure.reason === :eta_retries_exhausted
    @test length(retry_changes) == options.max_eta_retries
    @test finish.total_eta_retry_count == options.max_eta_retries
    @test finish.total_backtracking_count == failure.backtracking_count
    @test finish.accepted_steps == 0
end

end # module
