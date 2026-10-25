# Complete systems use the projected Newton engine in scholtes_solver.jl without
# the reduced system's rho relaxation. solver.jl's InteriorPoint path stalls on the eight constrained
# phase-1 benchmarks; it is not used for these complete-system solves.

"""A complete KKT residual and its sign bounds for `solve(Scholtes(), ...)`."""
struct CompleteKKTSystem{K,M} <: AbstractScholtesKKTSystem
    system::K
    data::M
end

@inline function Base.getproperty(kkt::CompleteKKTSystem, name::Symbol)
    name === :system && return getfield(kkt, :system)
    name === :data && return getfield(kkt, :data)
    data = getfield(kkt, :data)
    haskey(data, name) && return getproperty(data, name)
    getproperty(getfield(kkt, :system), name)
end

Base.propertynames(kkt::CompleteKKTSystem) =
    (:system, :data, propertynames(kkt.system)..., keys(kkt.data)...)
_direct_sign_dims(kkt::CompleteKKTSystem) = kkt.nonnegative_dims

function _initialize_direct_signs!(kkt::CompleteKKTSystem, y)
    y[kkt.nonnegative_dims] .= max.(y[kkt.nonnegative_dims], 1.0)
    y
end

function _norm_K₀(R::ScholtesResiduals{<:CompleteKKTSystem}, r, w, rho)
    R.kkt.formulation === :exact && return norm(r)
    R.kkt.complete_residual!(R.Fb, view(w, 1:R.n), R.kkt.θ, 0.0)
    norm(R.Fb)
end

"""
    generated_exact_complete_kkt_system(goop; codegen = :fast_differentiation, ...)

Section 2.2, equations (2.12)-(2.17) of manuscript/main.pdf: recursively retain
every lower-level equality, componentwise complementarity product, inequality,
and multiplier. Differentiate each full Lagrangian with respect to the player's
physical variables, all lower-level multipliers, and its preference slacks.

The returned `CompleteKKTSystem` works with `Scholtes()`'s projected Newton engine.
No rho or epsilon relaxation is applied. Nonnegative auxiliary slacks express
the final inequalities as equalities; eliminating them recovers the manuscript
system. Prioritized constraints use sum(s), h+s >= 0, and s >= 0.
"""
generated_exact_complete_kkt_system(goop::ParametricGOOP; kwargs...) =
    _generate_complete_kkt_system(goop, :exact; kwargs...)

"""
    generate_scholtes_complete_kkt_system(goop; rho = 1e-9, ...)

Build the existing slacked complete formulation directly for `Scholtes()`'s
projected Newton engine. Lower levels inherit scalar aggregate inequalities
rho - gamma'G >= 0; outer aggregate products equal rho. Here rho replaces the
legacy generator's epsilon. These are the
equations of `generate_slacked_complete_kkt_system`, not a Scholtes rho relaxation.
Use `ScholtesOptions(rho_schedule = [...])` for continuation. The solver reports
the unrelaxed aggregate residual at rho=0 and the relaxed residual separately.
"""
generate_scholtes_complete_kkt_system(goop::ParametricGOOP; kwargs...) =
    _generate_complete_kkt_system(goop, :aggregate; kwargs...)

function _generate_complete_kkt_system(
    goop::ParametricGOOP, formulation::Symbol;
    rho = formulation === :exact ? 0.0 : 1e-9,
    codegen = :fast_differentiation,
    fd_codegen_chunk_size = SCHOLTES_FD_CHUNK_SIZE,
    backend_options = (;),
)
    formulation in (:exact, :aggregate) || throw(ArgumentError("unknown complete formulation"))
    exact = formulation === :exact
    isfinite(rho) && rho >= 0 || throw(ArgumentError("rho must be finite and nonnegative"))
    exact && rho != 0 && throw(ArgumentError("the exact complete system has no rho relaxation"))
    codegen in (:native, :fast_differentiation) || throw(ArgumentError("unknown codegen option: $codegen"))
    for (i, preferences) in enumerate(goop.preferences)
        length(preferences) >= 2 || throw(ArgumentError(
            "Complete KKT construction requires K >= 2 for every player; player $i has K = $(length(preferences)).",
        ))
    end
    backend = SymbolicTracingUtils.SymbolicsBackend()
    x = SymbolicTracingUtils.make_variables(backend, :x, sum(goop.primal_dims))
    theta = SymbolicTracingUtils.make_variables(backend, :θ, sum(goop.parameter_dims))
    xb, tb = to_blockvector(goop.primal_dims)(x), to_blockvector(goop.parameter_dims)(theta)
    rho_symbol = Symbolics.variable(:complete_rho)
    eta = Symbolics.variable(:complete_unused_eta)
    T = Symbolics.Num
    sym(name, n) = SymbolicTracingUtils.make_variables(backend, Symbol(name), n)
    vector(value) = collect(T, value isa AbstractVector ? value : [value])
    preference_slacks, inequality_slacks, equality_duals, inequality_duals = (T[] for _ in 1:4)
    residuals, inequalities, slack_expressions, player_levels =
        Vector{T}[], Vector{T}[], Vector{T}[], Vector{Any}[]

    for i in 1:goop.num_players
        K = length(goop.preferences[i])
        xi = collect(xb[BlockArrays.Block(i)])
        f = goop.equality_dims[i] == 0 ? T[] : vector(goop.equality_constraints[i](xb, tb))
        g = goop.inequality_dims[i] == 0 ? T[] : vector(goop.inequality_constraints[i](xb, tb))
        F, G, z = copy(f), copy(g), copy(xi)
        inherited_gammas, preference_inequalities = T[], T[]
        slack_G = copy(g)
        groups = (; stationarity = Int[], equality = collect(eachindex(f)), complementarity = Int[])
        levels = Vector{Any}(undef, K)

        for k in K:-1:1
            h = vector(goop.preferences[i][k](xb, tb))
            prioritized = goop.is_prioritized_constraint[i][k]
            e = sym(exact ? "bar_s_$(i)_$(k)" : "s_$(i)_$(k)", prioritized ? length(h) : 0)
            gp = sym(exact ? "bar_gamma_preference_$(i)_$(k)" : "γₚ_$(i)_$(k)", length(e))
            mu = sym(exact ? "bar_mu_$(i)_$(k)" : "μₛ_$(i)_$(k)", length(e))
            append!(preference_slacks, e)
            !prioritized && length(h) != 1 && throw(ArgumentError("expected scalar objective at player $i level $k"))
            objective = prioritized ? sum(e; init = T(0)) : only(h)
            own_inequalities = prioritized ? [h .+ e; e] : T[]

            lambda = if exact && k < K
                dual = Vector{T}(undef, length(F))
                dual[groups.stationarity] = sym("bar_psi_$(i)_$(k)", length(groups.stationarity))
                dual[groups.equality] = sym("bar_lambda_$(i)_$(k)", length(groups.equality))
                dual[groups.complementarity] = sym("bar_phi_$(i)_$(k)", length(groups.complementarity))
                dual
            else
                sym(exact ? "bar_lambda_$(i)_$(k)" : "λ_$(i)_$(k)", length(F))
            end
            gamma = sym(exact ? "bar_gamma_$(i)_$(k)" : "γ_$(i)_$(k)", length(G))
            append!(equality_duals, lambda)
            append!(inequality_duals, [gamma; gp; mu])
            L = objective - _sdot(lambda, F) - _sdot(gamma, G) -
                (prioritized ? _sdot(gp, h .+ e) + _sdot(mu, e) : 0)
            variables = [z; e]
            stationarity = _gradient(L, variables)
            current_G, current_gamma = [G; own_inequalities], [gamma; gp; mu]

            if exact
                products = current_G .* current_gamma
                nstat, ncomp = length(stationarity), length(products)
                if k == K
                    Fnew = [stationarity; F; products]
                    groups = (; stationarity = collect(1:nstat),
                        equality = collect((nstat+1):(nstat+length(F))),
                        complementarity = collect((nstat+length(F)+1):(nstat+length(F)+ncomp)))
                else
                    offset = nstat + ncomp
                    Fnew = [stationarity; products; F]
                    groups = (; stationarity = [collect(1:nstat); offset .+ groups.stationarity],
                        equality = offset .+ groups.equality,
                        complementarity = [collect((nstat+1):offset); offset .+ groups.complementarity])
                end
                preference_inequalities = [own_inequalities; preference_inequalities]
                prioritized && append!(slack_G, h .+ e)
                inherited_gammas = [current_gamma; inherited_gammas]
                Gnew = [g; preference_inequalities; inherited_gammas]
            else
                Fnew = [stationarity; F]
                Gnew = [prioritized ? [h .+ e; gp;
                    isempty(e) ? T[] : [rho_symbol - gp' * (h .+ e)]; e; mu;
                    isempty(e) ? T[] : [rho_symbol - mu' * e]] : T[];
                    gamma; G; isempty(G) ? T[] : [rho_symbol - gamma' * G]]
            end
            # Preserve the existing aggregate generator's differentiation order.
            z = !exact && k == K ? [xi; e; gp; mu; lambda; gamma] :
                [z; e; lambda; gamma; gp; mu]
            levels[k] = (; F = Fnew, G = Gnew, L, variables, lambda, gamma,
                preference_slack = e, preference_gamma = gp, preference_mu = mu,
                stationarity, products = exact ? current_G .* current_gamma : T[])

            if k == 1
                if exact
                    # Coordinate bounds on multipliers and preference slacks are already explicit.
                    sigma = sym("bar_sigma_$(i)_1", length(slack_G))
                    append!(inequality_slacks, sigma)
                    push!(residuals, [Fnew; slack_G .- sigma])
                    push!(slack_expressions, copy(slack_G))
                else
                    sigma = sym("σ_g_$(i)_1", length(G))
                    sp = sym("σₚ_$(i)_1", length(e))
                    ss = sym("σₛ_$(i)_1", length(e))
                    append!(inequality_slacks, [sigma; sp; ss])
                    push!(slack_expressions, [G; prioritized ? [h .+ e; e] : T[]])
                    push!(residuals, [Fnew;
                        prioritized ? [h .+ e .- sp;
                            isempty(e) ? T[] : [rho_symbol - gp' * (h .+ e)]; e .- ss;
                            isempty(e) ? T[] : [rho_symbol - mu' * e]] : T[];
                        G .- sigma; isempty(G) ? T[] : [rho_symbol - gamma' * G]])
                end
                push!(inequalities, Gnew)
            end
            F, G = Fnew, Gnew
        end
        push!(player_levels, levels)
    end

    y = T[x; preference_slacks; inequality_slacks; equality_duals; inequality_duals]
    F = reduce(vcat, residuals; init = T[])
    exact || filter!(!iszero, F)
    G = reduce(vcat, inequalities; init = T[])
    nx, ne, ns, nl = length(x), length(preference_slacks), length(inequality_slacks), length(equality_duals)
    preference_dims = (nx+1):(nx+ne)
    slack_dims = collect((nx+ne+1):(nx+ne+ns))
    equality_dims = collect((nx+ne+ns+1):(nx+ne+ns+nl))
    gamma_dims = collect((nx+ne+ns+nl+1):length(y))
    signs = [collect(preference_dims); slack_dims; gamma_dims]
    block = _compile_scholtes_block(F, y, theta, rho_symbol, eta;
        codegen, fd_codegen_chunk_size, backend_options)
    f_raw!, j_raw!, J, pattern = block
    F! = (out, y, theta, rho) -> f_raw!(out, y, theta, exact ? 0.0 : rho)
    J! = (out, y, theta, rho) -> j_raw!(out, y, theta, exact ? 0.0 : rho)
    empty_F!, empty_J!, empty_buffer, empty_pattern = _compile_scholtes_block(
        T[], y, theta, rho_symbol, eta; codegen, fd_codegen_chunk_size, backend_options)
    system = ScholtesKKTSystem(length(y), length(F), 0, 1:nx,
        preference_dims, Int[], length(theta), F!, empty_F!, empty_F!,
        J!, empty_J!, empty_J!, J, empty_buffer, copy(empty_buffer),
        pattern, empty_pattern, empty_pattern, Int[], 0, NamedTuple[], y,
        zeros(length(theta)), Ref(float(rho)), zeros(length(y)), Ref(false))
    CompleteKKTSystem(system, (; formulation, complete_residual! = f_raw!,
        F_symbolic = F, G_symbolic = G, z_symbolic = y, levels = player_levels,
        slack_expressions = reduce(vcat, slack_expressions; init = T[]),
        nonnegative_dims = signs, interior_point_slack_dims = slack_dims,
        inequality_constraint_dual_dims = gamma_dims, equality_constraint_dual_dims = equality_dims,
        kkt_dimension = length(F), variable_dimension = length(y)))
end
