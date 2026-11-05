# PrecompileTools workload: build and solve a small two-player problem along the package's paths -- the
# Scholtes system by ProjectedLM with every linear solver, and the perturbed reduced system (5.2) by
# PDIP -- so the symbolic construction, the code generator and both solvers are compiled at package
# precompile time instead of on every session's first call. (The generated functions of a user's own
# problem still compile on its first solve.)
@setup_workload begin
    x = BlockArray(zeros(2), [1, 1])
    θ = BlockArray(zeros(0), [0])
    goop = ParametricGOOP(
        x,
        θ;
        preferences = [
            Function[(x, θ) -> x[1]^2, (x, θ) -> (x[1] - (1.0 + 0.25 * x[2]))^2],
            Function[(x, θ) -> x[2]^2, (x, θ) -> (x[2] - (-1.0 + 0.25 * x[1]))^2],
        ],
        is_prioritized_constraint = [[false, false], [false, false]],
        equality_constraints = [nothing, nothing],
        inequality_constraints = [(x, θ) -> [x[1] + 2.0], (x, θ) -> [x[2] + 2.0]],
    )
    @compile_workload begin
        # chunk size 2 splits this small system, so the chunked code path (the default for real problems) is
        # compiled too
        generate_slacked_scholtes_reduced_kkt_system(goop; fd_codegen_chunk_size = 2)
        kkt = generate_slacked_scholtes_reduced_kkt_system(goop)
        z₀ = initialize_scholtes!(zeros(kkt.variable_dimension), kkt, Float64[], [0.0, 0.0], 1.0; warm = false)
        for linear_solver in (:normal, :klu, :ldl)
            solve(ProjectedLM(), kkt, Float64[]; z₀, options = ProjectedLMOptions(; linear_solver, record_convergence = true))
        end
        reduced = generate_slacked_reduced_kkt_system(goop)
        solve(PDIP(), reduced, Float64[]; z₀ = [0.0, 0.0], options = PDIPOptions(; record_convergence = true))
    end
end
