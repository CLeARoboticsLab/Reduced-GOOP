# PrecompileTools workload: build and solve a small two-player Scholtes problem with
# every sparse backend, so the symbolic construction, the code generator and the solver
# are compiled at package precompile time instead of on every session's first call.
# (The generated functions of a user's own problem still compile on its first solve.)
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
        # chunk size 2 splits this small system, so the chunked code path (the default
        # for real problems) is compiled too
        generate_slacked_reduced_kkt_system(goop; complementarity = :scholtes, fd_codegen_chunk_size = 2)
        kkt = generate_slacked_reduced_kkt_system(goop; complementarity = :scholtes)
        for linear_solver in (:normal, :klu_eta, :klu_sqrt_eta)
            solve(
                Scholtes(),
                kkt,
                Float64[];
                z₀ = [0.0, 0.0],
                options = ScholtesOptions(; linear_solver, eta_schedule = true),
            )
        end
    end
end
