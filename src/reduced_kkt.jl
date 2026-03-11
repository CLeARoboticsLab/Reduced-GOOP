# goop_reduced_kkt_general.jl
#
# Reduced GOOP KKT system — N players, K^i preference levels each.
# Section 4, equations (4.1)–(4.10) of:
# "Breaking Exponential Complexity in Games of Ordered Preference"
#
# ── Reduced vs. complete ───────────────────────────────────────────────────────
# At each upper level k < K^i, stationarity is taken w.r.t. z^i ONLY
# (not induced primal dual variables from lower levels).  Variable count grows
# polynomially in K^i instead of exponentially (Proposition 4.3):
#
#   |F^i|    = K^i n^i + m^i_E + K^i m^i_I          (equations)
#   |vars^i| = (1 + K^i(K^i-1)/2) n^i + K^i m^i_E + K^i(K^i+1)/2 m^i_I
#   |G^i|    = (K^i + 1) m^i_I                       (inequalities)
#
# ── Innermost level k = K^i  (identical to complete, eq. 4.1–4.4) ─────────────
#   L^i_{K^i}  = J^i_{K^i} − λ^{i⊤}_{K^i} h^i − γ^{i⊤}_{K^i} g^i       (4.1)
#   π^i_{K^i}  = ∇_{z^i} L^i_{K^i}                                        (4.2)
#   F^i_{K^i}  = [∇_{z^i}L^i_{K^i};  h^i;  g^i⊙γ^i_{K^i}] = 0           (4.3)
#   G^i_{K^i}  = [g^i;  γ^i_{K^i}] ≥ 0                                    (4.4)
#
# ── Upper levels k = K^i−1,…,1  (reduced, eq. 4.5–4.8) ──────────────────────
#   L^i_k = J^i_k − λ^{i⊤}_k h^i − γ^{i⊤}_k g^i
#               − ψ^{i⊤}_k π^i_{k+1}
#               − Σ_{ℓ=1}^{K^i−k}  ϕ^{i⊤}_{k,ℓ} [g^i ⊙ γ^i_{K^i−ℓ+1}]  (4.5)
#   π^i_k      = [∇_{z^i}L^i_k;  π^i_{k+1}]                               (4.6)
#   F^i_k      = [∇_{z^i}L^i_k;  g^i⊙γ^i_k;  F^i_{k+1}] = 0             (4.7)
#   G^i_k      = [g^i;  γ^i_{k:K^i}] ≥ 0                                  (4.8)
#
# ── Compact reordered form (eq. 4.9–4.10) ────────────────────────────────────
#   F^i = [∇_{z^i}L^i_1; …; ∇_{z^i}L^i_{K^i};
#          h^i;
#          g^i⊙γ^i_1; …; g^i⊙γ^i_{K^i}] = 0
#   G^i = [g^i;  γ^i_{1:K^i}] ≥ 0
#
# ── PDIP formulation (Section 6, eq. 6.3) ────────────────────────────────────
# Slack s^i_k per level: g^i − s^i_k = 0,  s^i_k ⊙ γ^i_k = ρ.
# ─────────────────────────────────────────────────────────────────────────────

using Symbolics
using LinearAlgebra
using Printf
using Plots
using Infiltrator
ENV["GKSwstype"] = "nul"

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Symbolic builder
# ═══════════════════════════════════════════════════════════════════════════════
"""
    build_reduced_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)

Build the reduced symbolic GOOP KKT system (Section 4, eq. 4.1–4.10).
Coupled constraints (h^i, g^i may depend on all players' z) are supported.

Returns the same 10-tuple layout as `build_complete_goop_kkt`:
  zv, player_F, player_G, player_vars, all_F, all_vars,
  comp_idx_per_player, all_comp_idx, all_comp_a, all_comp_b
"""
function build_reduced_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)
    n_total  = sum(ni)
    z_ranges = [sum(ni[1:i-1])+1 : sum(ni[1:i]) for i in 1:N]

    @variables z_sym[1:n_total]
    zv = collect(z_sym)

    player_F            = Vector{Vector{Num}}(undef, N)
    player_G            = Vector{Vector{Num}}(undef, N)
    player_vars         = Vector{Vector{Num}}(undef, N)
    comp_idx_per_player = Vector{Vector{Int}}(undef, N)
    comp_a_per_player   = Vector{Vector{Num}}(undef, N)
    comp_b_per_player   = Vector{Vector{Num}}(undef, N)

    for i in 1:N
        Ki = K_vec[i]
        zi = zv[z_ranges[i]]

        # Constraint functions (may depend on all z)
        h_val = h_fns[i](zv)   # Vector{Num}, length mEi[i]
        g_val = g_fns[i](zv)   # Vector{Num}, length mIi[i]

        # ── Innermost level k = K^i  (eq. 4.1–4.4) ───────────────────────────
        λK_v = [Symbolics.variable(Symbol("lK_p$(i)_$j")) for j in 1:mEi[i]]
        γK_v = [Symbolics.variable(Symbol("gK_p$(i)_$j")) for j in 1:mIi[i]]

        L_Ki  = J_fns[i][Ki](zv) - sum(λK_v .* h_val) - sum(γK_v .* g_val)
        ∇L_Ki = Symbolics.gradient(L_Ki, zi)   # ∂/∂z^i only

        # π^i_{K^i} = ∇_{z^i} L^i_{K^i}  (eq. 4.2)
        π_cur = copy(∇L_Ki)          # will grow as we move to upper levels

        # Accumulate stationarity gradients (K^i … 1), reversed at the end
        stat_grads = Vector{Num}[∇L_Ki]

        # Accumulate γ^i_k variables (K^i … 1), reversed at the end
        γ_all = Vector{Num}[γK_v]

        # Dual variables introduced for player i (excluding shared z)
        duals_i = Num[λK_v; γK_v]

        # ── Upper levels k = K^i−1,…,1  (eq. 4.5–4.8) ───────────────────────
        for k in Ki-1:-1:1
            n_phi = Ki - k   # number of ϕ_{k,ℓ} groups

            # ψ^i_k : multiplier for π^i_{k+1}  (dim = n_phi × ni[i])
            ψk_v = [Symbolics.variable(Symbol("psi_p$(i)_k$(k)_$j"))
                    for j in 1:n_phi*ni[i]]

            # ϕ^i_{k,ℓ} for ℓ = 1 … n_phi  (each dim = mIi[i])
            # ℓ=1 → multiplies g^i ⊙ γ^i_{K^i}
            # ℓ=2 → multiplies g^i ⊙ γ^i_{K^i−1}  …
            ϕk_vs = [
                [Symbolics.variable(Symbol("phi_p$(i)_k$(k)_l$(ℓ)_$j"))
                 for j in 1:mIi[i]]
                for ℓ in 1:n_phi
            ]

            λk_v = [Symbolics.variable(Symbol("lam_p$(i)_k$(k)_$j")) for j in 1:mEi[i]]
            γk_v = [Symbolics.variable(Symbol("gam_p$(i)_k$(k)_$j")) for j in 1:mIi[i]]

            # Lagrangian L^i_k  (eq. 4.5)
            L_k = J_fns[i][k](zv) -
                  sum(λk_v .* h_val) -
                  sum(γk_v .* g_val) -
                  sum(ψk_v .* π_cur)
            for ℓ in 1:n_phi
                # γ_all[ℓ] = γ^i_{K^i−ℓ+1}
                # (γ_all[1] = γ_{K^i}, γ_all[2] = γ_{K^i−1}, …)
                L_k -= sum(ϕk_vs[ℓ] .* (g_val .* γ_all[ℓ]))
            end

            # Stationarity w.r.t. z^i ONLY  (reduced: no induced-primal stationarity)
            ∇L_k = Symbolics.gradient(L_k, zi)

            # Update π: π^i_k = [∇_{z^i}L^i_k;  π^i_{k+1}]  (eq. 4.6)
            π_cur = [∇L_k; π_cur]

            push!(stat_grads, ∇L_k)
            push!(γ_all,      γk_v)
            duals_i = [duals_i; ψk_v; vcat(ϕk_vs...); λk_v; γk_v]
        end

        # ── Compact form F^i  (eq. 4.9) ───────────────────────────────────────
        # stat_grads was built K^i → 1; reverse → 1 … K^i
        # γ_all was built K^i → 1; reverse → γ^i_1 … γ^i_{K^i}
        stat_ordered = reverse(stat_grads)
        γ_ordered    = reverse(γ_all)

        stat_part = vcat(stat_ordered...)                            # K^i · ni[i]   rows
        h_part    = collect(Num, h_val)                             # mEi[i]        rows
        comp_part = vcat([g_val .* γ_ordered[k] for k in 1:Ki]...) # K^i · mIi[i]  rows

        F_i = [stat_part; h_part; comp_part]

        # G^i = [g^i;  γ^i_1; …; γ^i_{K^i}]  (eq. 4.10)
        G_i = [collect(Num, g_val); vcat(γ_ordered...)]

        # ── Complementarity indices & pairs ───────────────────────────────────
        stat_len   = Ki * ni[i]
        h_len      = mEi[i]
        comp_start = stat_len + h_len + 1
        comp_idx_i = collect(comp_start : comp_start + Ki*mIi[i] - 1)

        # a-side: g^i (replicated Ki times, one per level); b-side: γ^i_k
        comp_a_i = vcat([collect(Num, g_val) for _ in 1:Ki]...)
        comp_b_i = vcat(γ_ordered...)

        player_F[i]            = F_i
        player_G[i]            = G_i
        player_vars[i]         = [zi; duals_i]
        comp_idx_per_player[i] = comp_idx_i
        comp_a_per_player[i]   = comp_a_i
        comp_b_per_player[i]   = comp_b_i
    end

    # ── Stacked system ─────────────────────────────────────────────────────────
    all_F    = vcat(player_F...)
    all_vars = copy(zv)
    for i in 1:N
        all_vars = [all_vars; player_vars[i][ni[i]+1:end]]
    end

    all_comp_idx = Int[]
    off = 0
    for i in 1:N
        append!(all_comp_idx, comp_idx_per_player[i] .+ off)
        off += length(player_F[i])
    end

    all_comp_a = vcat(comp_a_per_player...)
    all_comp_b = vcat(comp_b_per_player...)

    # ── Dimension check (Proposition 4.3) ─────────────────────────────────────
    println("\n── Dimension check (Proposition 4.3) ──")
    for i in 1:N
        Ki = K_vec[i]
        exp_eqs  = Ki*ni[i] + mEi[i] + Ki*mIi[i]
        exp_vars = (1 + Ki*(Ki-1)÷2)*ni[i] + Ki*mEi[i] + Ki*(Ki+1)÷2*mIi[i]
        exp_ineq = (Ki+1)*mIi[i]
        @printf("  Player %d: |F^%d|=%d (exp %d)  |G^%d|=%d (exp %d)  vars=%d (exp %d)\n",
            i, i, length(player_F[i]), exp_eqs,
            i, length(player_G[i]), exp_ineq,
            length(player_vars[i]), exp_vars)
        @assert length(player_F[i])    == exp_eqs  "Player $i: |F| mismatch"
        @assert length(player_vars[i]) == exp_vars "Player $i: |vars| mismatch"
        @assert length(player_G[i])    == exp_ineq "Player $i: |G| mismatch"
    end
    @assert length(all_comp_a) == length(all_comp_idx)
    @assert length(all_comp_b) == length(all_comp_idx)
    println("  All assertions passed ✓")

    return zv, player_F, player_G, player_vars, all_F, all_vars,
           comp_idx_per_player, all_comp_idx, all_comp_a, all_comp_b
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PDIP solver — Algorithm 6.1 with s^i_k ⊙ γ^i_k = ρ  (eq. 6.3–6.6)
# ═══════════════════════════════════════════════════════════════════════════════
"""
    solve_reduced_goop_pdip(all_F, all_vars,
                            all_comp_idx, all_comp_a, all_comp_b; ...)

Solve F(y) = 0, G(y) ≥ 0 via Algorithm 6.1.
Every comp. condition  a_j(y)·b_j(y) = 0  (a_j = g^i component, b_j = γ^i_k)
is replaced by:

    a_j(y) − s_j = 0        (primal feasibility)
    s_j · b_j(y) = ρ        (perturbed complementarity, ρ → 0)

Augmented variable: w = [y; s].  Jacobian ∂K_ρ/∂w:

    [ ∂F_nc/∂y        0       ]
    [ ∂a/∂y           −I      ]
    [ diag(s)·∂b/∂y   diag(b) ]

Line search keeps s > 0 and b(y) > 0 at every step.

Returns (y_sol, s_sol, res_history).
"""
function solve_reduced_goop_pdip(
        all_F, all_vars,
        all_comp_idx, all_comp_a, all_comp_b;
        y0          = nothing,
        s0          = nothing,
        rho_init    = 0.1,
        rho_factor  = 0.1,
        rho_min     = 1e-10,
        max_outer   = 30,
        max_inner   = 100,
        tol_inner   = 1e-6,
        tol_outer   = 1e-8,
        c1          = 1e-4,
        beta_ls     = 0.5,
        verbose     = true)

    n           = length(all_vars)
    n_eq        = length(all_F)
    n_comp      = length(all_comp_idx)
    noncomp_idx = setdiff(1:n_eq, all_comp_idx)

    verbose && println("Compiling numerical functions…")

    F_nc_sym = all_F[noncomp_idx]
    F_nc_num = Symbolics.build_function(F_nc_sym, all_vars; expression=Val{false})[1]
    a_num    = Symbolics.build_function(all_comp_a, all_vars; expression=Val{false})[1]
    b_num    = Symbolics.build_function(all_comp_b, all_vars; expression=Val{false})[1]

    verbose && println("Building symbolic Jacobians…  (may take a moment)")
    J_nc_num = Symbolics.build_function(
        Symbolics.jacobian(F_nc_sym,   all_vars), all_vars; expression=Val{false})[1]
    J_a_num  = Symbolics.build_function(
        Symbolics.jacobian(all_comp_a, all_vars), all_vars; expression=Val{false})[1]
    J_b_num  = Symbolics.build_function(
        Symbolics.jacobian(all_comp_b, all_vars), all_vars; expression=Val{false})[1]
    verbose && println("Symbolic Jacobians ready.")

    # Perturbed system K_ρ(w) = 0,  w = [y; s]  (eq. 6.3)
    function K_rho(w, rho)
        y = w[1:n];  s = w[n+1:end]
        [F_nc_num(y);  a_num(y) .- s;  s .* b_num(y) .- rho]
    end

    n_nc = n_eq - n_comp   # number of non-complementarity equations

    # Augmented Jacobian (ρ-independent structure)
    # J_aug is (n_eq + n_comp) × (n + n_comp)  [generally non-square for reduced system]
    function J_aug(w)
        y = w[1:n];  s = w[n+1:end]
        bv = b_num(y)
        [J_nc_num(y)             zeros(n_nc, n_comp);
         J_a_num(y)              -I(n_comp);
         diagm(s) * J_b_num(y)  diagm(bv)]
    end

    # Initialise
    y_init = isnothing(y0) ? fill(0.1, n) : copy(float.(y0))
    a0     = a_num(y_init)
    s_init = isnothing(s0) ? max.(a0, fill(rho_init, n_comp)) : copy(float.(s0))
    @assert all(s_init .> 0)        "Initial slacks s0 must be > 0"
    @assert all(b_num(y_init) .> 0) "Initial γ duals b(y0) must be > 0"

    w   = [y_init; s_init]
    rho = rho_init
    K0(w) = K_rho(w, 0.0)

    verbose && println("="^65)
    verbose && @printf("  Reduced GOOP KKT — PDIP  (s⊙b = ρ, Algorithm 6.1)\n")
    verbose && @printf("  vars=%d  eqs=%d  n_comp=%d  w_dim=%d  rho_0=%.2e\n",
                       n, n_eq, n_comp, n + n_comp, rho)
    verbose && println("="^65)

    res_history = Float64[]
    converged   = false

    for outer in 1:max_outer
        verbose && @printf("\n  -- rho = %.2e  (outer %d/%d) --\n", rho, outer, max_outer)
        verbose && @printf("  %4s  %15s  %12s\n", "iter", "||K_rho(w)||", "alpha")
        verbose && println("  " * "-"^50)

        for inner in 1:max_inner
            Kw  = K_rho(w, rho)
            res = norm(Kw)
            push!(res_history, res)

            if res < tol_inner
                verbose && @printf("  %4d  %15.6e  (inner converged)\n", inner, res)
                break
            end

            # Newton step  (eq. 6.5)
            J  = J_aug(w)
            # @infiltrate
            Δw = pinv(J) * (-Kw)

            Δy = Δw[1:n];  Δs = Δw[n+1:end]
            y_c = w[1:n];  s_c = w[n+1:end]

            # Armijo backtracking, enforcing s > 0 and b(y) > 0  (Alg 6.1, line 5)
            α = 1.0
            for _ in 1:60
                yn = y_c .+ α .* Δy
                sn = s_c .+ α .* Δs
                if all(sn .> 0) && all(b_num(yn) .> 0) &&
                   norm(K_rho([yn; sn], rho)) ≤ (1.0 - c1*α) * res
                    break
                end
                α *= beta_ls
                α < 1e-16 && break
            end

            verbose && @printf("  %4d  %15.6e  %12.8f\n", inner, res, α)
            w .+= α .* Δw
        end

        res_full = norm(K0(w))
        verbose && @printf("  -> ||K0(w)|| = %.4e  (rho=%.2e)\n", res_full, rho)

        if res_full < tol_outer
            converged = true
            verbose && println("\n  Converged!")
            break
        end

        rho = max(rho * rho_factor, rho_min)
    end

    y_sol = w[1:n];  s_sol = w[n+1:end]

    verbose && println("\n" * "="^65)
    verbose && @printf("  Final ||K0(w)||  = %.6e\n", norm(K0(w)))
    verbose && @printf("  min s            = %.6e\n", minimum(s_sol))
    verbose && @printf("  min b(y)         = %.6e\n", minimum(b_num(y_sol)))
    verbose && @printf("  max |a(y) − s|   = %.6e\n", maximum(abs.(a_num(y_sol) .- s_sol)))
    verbose && println("  converged        = $converged")

    return y_sol, s_sol, res_history
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Example — Double-well inner layer (multiple inner optima)
#     Same design as goop_complete_kkt_general.jl for direct comparison.
# ═══════════════════════════════════════════════════════════════════════════════
#
#  2-player, 2-level GOOP.  z = [x₁; y₁; x₂; y₂]
#  Inner level (k=2): J^i_2 = (x^i² - 1)²  — double-well, optima at x^i = ±1
#  Outer level (k=1): J^1_1 = x₁  (prefers left well x₁ = -1)
#                     J^2_1 = -x₂ (prefers right well x₂ = +1)
#  Expected GOOP solution: x₁* = -1, y₁* = 0, x₂* = +1, y₂* = 0
# ═══════════════════════════════════════════════════════════════════════════════



let
    N     = 2
    K     = 3
    ni    = [3, 3]
    mEi   = [1, 1]
    mIi   = [1, 1]
    K_vec = fill(K, N)
    n_z   = sum(ni)   # 6


    J_fns = [
        [   # Player 1
            z -> (z[3]-1)^2,
            z -> z[1],                # J¹₁ outer: minimize x₁ → selects x₁ = -1
            z -> (z[1]^2 - 1)^2,     # J¹₂ inner: double-well (optima at x₁ = ±1)
        ],
        [   # Player 2
            z -> (z[6]+1)^2,
            z -> -z[4],              # J²₁ outer: minimize -x₂ → selects x₂ = +1
            z -> (z[4]^2 - 1)^2,    # J²₂ inner: double-well (optima at x₂ = ±1)
        ],
    ]

    # Equality constraints: pin y-component to 0 (decoupled)
    h_fns = [
        z -> [z[2] - z[6]],
        z -> [z[5] - z[3]],
    ]

    # Inequality constraints: keep x^i inside (-2, 2) so both wells ±1 are feasible
    g_fns = [
        z -> [4.0 - z[1]^2],
        z -> [4.0 - z[4]^2],
    ]

    # ── Build reduced KKT system ───────────────────────────────────────────────
    println("\n── Building symbolic reduced GOOP KKT system ────────────────────────")
    zv, player_F, player_G, player_vars, all_F, all_vars,
        comp_idx_per_player, all_comp_idx, all_comp_a, all_comp_b =
        build_reduced_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)

    # ── Initial point ──────────────────────────────────────────────────────────
    n_tot = length(all_vars)
    y0    = fill(0.0, n_tot)
    y0[1:n_z] .= [-1.5, 0.0, 0.0, 1.5, 0.0, 0.0]   # x₁,y₁,x₂,y₂
    off = n_z
    for i in 1:N
        n_extra = length(player_vars[i]) - ni[i]
        y0[off+1 : off+n_extra] .= 0.1
        off += n_extra
    end

    z0 = y0[1:n_z]

    # ── Solve ──────────────────────────────────────────────────────────────────
    println("\n── Running PDIP solver  (s ⊙ b = ρ) ────────────────────────────────")
    y_sol, s_sol, res_hist = solve_reduced_goop_pdip(
        all_F, all_vars,
        all_comp_idx, all_comp_a, all_comp_b;
        y0         = y0,
        rho_init   = 1.0,
        rho_factor = 0.1,
        rho_min    = 1e-10,
        max_outer  = 1,
        max_inner  = 60,
        tol_inner  = 1e-7,
        tol_outer  = 1e-8,
        # c1 = 0.0
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    z_sol = y_sol[1:n_z]
    println("\n── Solution ─────────────────────────────────────────────────────────")
    @printf("  z* = \n")
    println(z_sol)

    println("\nConstraint satisfaction at z*:")
    @printf("  h¹(z*) = % .2e  (should be 0)\n",    h_fns[1](z_sol)[1])
    @printf("  h²(z*) = % .2e  (should be 0)\n",    h_fns[2](z_sol)[1])
    @printf("  g¹(z*) = %.6f  (> 0 required)\n", g_fns[1](z_sol)[1])
    @printf("  g²(z*) = %.6f  (> 0 required)\n", g_fns[2](z_sol)[1])

    println("\nObjectives at z*:")
    for i in 1:N, k in 1:K
        @printf("  J^%d_%d(z*) = %.6f\n", i, k, J_fns[i][k](z_sol))
    end

    if !isempty(res_hist)
        plt = plot(res_hist;
                   yscale=:log10, 
                   xlabel="Newton iteration",
                   ylabel="‖K_ρ(w)‖",
                   title="Reduced GOOP PDIP",
                   legend=false, 
                   lw=2, 
                   marker=:circle, 
                   ms=3,
                   grid=true, 
                   framestyle=:box)
        savefig(plt, "goop_reduced_convergence.png")
    end
end