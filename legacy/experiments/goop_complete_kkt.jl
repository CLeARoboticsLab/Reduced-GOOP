# Complete GOOP KKT system — N players, K preference levels each
# Following Section 3.2 (eq. 3.8–3.12) of:
# "Breaking Exponential Complexity in Games of Ordered Preference"
#
# Nash game structure: player i controls z^i ∈ R^{ni[i]},
# with cost J^i_k(z) that may depend on all players' variables z = [z^1;…;z^N].
# Constraints h^i(z^i) = 0 and g^i(z^i) ≥ 0 may depend only on z^i (or all of z).
#
# Section 3.2 recursive construction for player i:
#
#   Level K (innermost):
#     L̄^i_K = J^i_K(z) − λ̄^{i⊤}_K h^i(z^i) − γ̄^{i⊤}_K g^i(z^i)          (3.8)
#     F̄^i_K = [∇_{z^i} L̄^i_K;  h^i(z^i);  g^i(z^i) ⊙ γ̄^i_K]  = 0         (3.9)
#     Ḡ^i_K = [g^i(z^i);  γ̄^i_K]  ≥ 0                                       (3.10)
#
#   Level k < K (dual variables η̄^i_k have dim |F̄^i_{k+1}|,
#                               γ̄^i_k have dim |Ḡ^i_{k+1}|):
#     L̄^i_k = J^i_k(z) − η̄^{i⊤}_k F̄^i_{k+1} − γ̄^{i⊤}_k Ḡ^i_{k+1}
#     F̄^i_k = [∇_{z^i} L̄^i_k;                                               (3.11)
#              ∇_{η̄^i_{k+1:K}} L̄^i_k;
#              Ḡ^i_{k+1} ⊙ γ̄^i_k;
#              F̄^i_{k+1}]  = 0
#     Ḡ^i_k = [g^i(z^i);  γ̄^i_{k:K}]  ≥ 0                                   (3.12)
#
# Variable count per player i: 2^{K-1}(ni[i] + mEi[i] + K·mIi[i])   (Prop. 3.4)
#
# ── Usage ─────────────────────────────────────────────────────────────────────
# Call build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)
# where:
#   J_fns[i][k](z_sym)  → scalar Num   (objective for player i at level k)
#   h_fns[i](z_sym)     → Vector{Num}  (equality constraints for player i, length mEi[i])
#   g_fns[i](z_sym)     → Vector{Num}  (inequality constraints g ≥ 0, length mIi[i])
# All functions receive the FULL joint symbolic vector z_sym = [z^1;…;z^N].
# ──────────────────────────────────────────────────────────────────────────────

using Symbolics
using LinearAlgebra
using Printf
using Plots
using Infiltrator

ENV["GKSwstype"] = "nul"

# ── Core builder ──────────────────────────────────────────────────────────────
"""
    build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)

Build the complete GOOP KKT system (F̄, Ḡ) following Section 3.2 of the paper.

Returns the symbolic system plus complementarity index sets needed for PDIP.

# Returns
- `z_sym`               : joint symbolic variable [z^1;…;z^N]
- `player_F`            : player_F[i] = F̄^i_1
- `player_G`            : player_G[i] = Ḡ^i_1
- `player_vars`         : player_vars[i] = all variables for player i
- `all_F`               : stacked F̄ = [F̄^1_1; …; F̄^N_1]
- `all_vars`            : stacked variable vector
- `comp_idx_per_player` : comp_idx_per_player[i] = indices into player_F[i] that are
                          complementarity conditions (g_j · γ_j = 0)
- `all_comp_idx`        : corresponding indices into all_F
"""
function build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)
    n_total  = sum(ni)
    z_ranges = [sum(ni[1:i-1])+1 : sum(ni[1:i]) for i in 1:N]

    # ── Symbolic joint variable ───────────────────────────────────────────────
    @variables z_sym[1:n_total]
    zv = collect(z_sym)

    player_F            = Vector{Vector{Num}}(undef, N)
    player_G            = Vector{Vector{Num}}(undef, N)
    player_vars         = Vector{Vector{Num}}(undef, N)
    comp_idx_per_player = Vector{Vector{Int}}(undef, N)

    for i in 1:N
        Ki = K_vec[i]
        zi = zv[z_ranges[i]]

        # ── Level K (innermost base system) ──────────────────────────────────
        λK_v = [Symbolics.variable(Symbol("lK_p$(i)_$j"))  for j in 1:mEi[i]]
        γK_v = [Symbolics.variable(Symbol("gK_p$(i)_$j"))  for j in 1:mIi[i]]

        h_val = h_fns[i](zv)
        g_val = g_fns[i](zv)

        L_K     = J_fns[i][Ki](zv) - sum(λK_v .* h_val) - sum(γK_v .* g_val)
        ∇zi_LK  = Symbolics.gradient(L_K, zi)

        F_cur    = [∇zi_LK; h_val; g_val .* γK_v]   # F̄^i_K  (ni + mEi + mIi)  (3.9)
        G_cur    = [g_val;  γK_v]                    # Ḡ^i_K  (2·mIi)           (3.10)
        vars_cur = [zi; λK_v; γK_v]                  # η^i_K = [λ̄; γ̄]  per (3.7)

        # Track complementarity indices in F_cur:
        # at level K the last mIi[i] entries of F_cur are g ⊙ γK.
        comp_indices_i = collect(ni[i] + mEi[i] + 1 : ni[i] + mEi[i] + mIi[i])

        # ── Recursive levels k = K-1 down to 1 ───────────────────────────────
        for k in Ki-1:-1:1
            n_F = length(F_cur)
            n_G = length(G_cur)

            # η̄^i_k : one multiplier per equation in F̄^i_{k+1}   (size n_F)
            # γ̄^i_k : one dual per inequality in Ḡ^i_{k+1}       (size n_G)
            η_v = [Symbolics.variable(Symbol("eta_p$(i)_k$(k)_$j")) for j in 1:n_F]
            γ_v = [Symbolics.variable(Symbol("gam_p$(i)_k$(k)_$j")) for j in 1:n_G]

            L_k    = J_fns[i][k](zv) - sum(η_v .* F_cur) - sum(γ_v .* G_cur)
            ∇_Lk   = Symbolics.gradient(L_k, vars_cur)   # stationarity (3.11)
            comp_k = G_cur .* γ_v                        # complementarity (3.11)

            # ── Update complementarity index tracking BEFORE appending ────────
            # New F_cur will be [∇_Lk (|vars_cur|);  comp_k (n_G);  F_cur (n_F)]
            # ⟹ old comp_indices shift forward by |vars_cur| + n_G
            offset_new = length(vars_cur) + n_G
            # comp at this level k occupies positions |vars_cur|+1 .. |vars_cur|+n_G
            new_comp_here  = collect(length(vars_cur) + 1 : length(vars_cur) + n_G)
            comp_indices_i = [new_comp_here; comp_indices_i .+ offset_new]

            F_cur    = [∇_Lk; comp_k; F_cur]
            G_cur    = [G_cur; γ_v]
            vars_cur = [vars_cur; η_v; γ_v]
        end

        player_F[i]            = F_cur
        player_G[i]            = G_cur
        player_vars[i]         = vars_cur
        comp_idx_per_player[i] = comp_indices_i

        expected_F = 2^(Ki-1) * (ni[i] + mEi[i] + Ki*mIi[i])
        println("Player $i:  |F̄^$(i)_1| = $(length(F_cur))  (expected $expected_F)")
    end

    # ── Assemble full stacked system ──────────────────────────────────────────
    all_F    = vcat(player_F...)
    all_vars = copy(zv)
    for i in 1:N
        all_vars = [all_vars; player_vars[i][ni[i]+1:end]]
    end

    # Map player-level comp indices into the stacked all_F
    all_comp_idx = Int[]
    off = 0
    for i in 1:N
        append!(all_comp_idx, comp_idx_per_player[i] .+ off)
        off += length(player_F[i])
    end

    println("\n── Dimension check (Proposition 3.4) ──")
    N_exp = sum(2^(K_vec[i]-1) * (ni[i] + mEi[i] + K_vec[i]*mIi[i]) for i in 1:N)
    G_exp = sum(2^K_vec[i] * mIi[i] for i in 1:N)
    println("  |F̄|    = $(length(all_F))  (expected $N_exp)")
    println("  |vars|  = $(length(all_vars))  (expected $N_exp)")
    println("  |Ḡ|    = $(sum(length.(player_G)))  (expected $G_exp)")
    println("  |comp|  = $(length(all_comp_idx))  (complementarity conditions)")
    @assert length(all_F)    == N_exp  "Equation count mismatch"
    @assert length(all_vars) == N_exp  "Variable count mismatch"
    println("  All assertions passed ✓")

    return zv, player_F, player_G, player_vars, all_F, all_vars,
           comp_idx_per_player, all_comp_idx
end

# ══════════════════════════════════════════════════════════════════════════════
# PDIP solver for the complete GOOP KKT system
# ══════════════════════════════════════════════════════════════════════════════
"""
    solve_complete_goop_pdip(player_G, all_F, all_vars,
                             comp_idx_per_player, all_comp_idx; ...)

Solve the complete GOOP KKT system F̄(y) = 0  s.t. Ḡ(y) ≥ 0 using a
Primal-Dual Interior Point method following Algorithm 6.1 of the paper.

The perturbed system at barrier parameter ρ replaces every complementarity
condition  g_j · γ_j = 0  with  g_j · γ_j = ρ,  and drives ρ → 0.

The Jacobian of F̄ is computed once symbolically and compiled; Newton
iterations use this compiled Jacobian with an Armijo backtracking line search
that enforces Ḡ(y) > 0 at every step.

# Keyword arguments
- `y0`          : initial point; must satisfy Ḡ(y0) > 0  (default: 0.1-vector)
- `rho_init`    : initial barrier parameter ρ₀             (default 0.1)
- `rho_factor`  : multiplicative reduction per outer step  (default 0.1)
- `rho_min`     : minimum ρ (stop reducing below this)     (default 1e-10)
- `max_outer`   : number of outer ρ-reduction steps        (default 30)
- `max_inner`   : Newton iterations per outer step         (default 100)
- `tol_inner`   : inner convergence threshold on ‖F_ρ(y)‖  (default 1e-6)
- `tol_outer`   : outer convergence threshold on ‖F(y)‖    (default 1e-8)
- `c1`          : Armijo sufficient-decrease constant       (default 1e-4)
- `beta_ls`     : backtracking factor                       (default 0.5)
- `verbose`     : print iteration log                       (default true)

# Returns
- `y_sol`       : solution vector in the ordering of `all_vars`
- `res_history` : ‖F_ρ(y)‖ at every inner Newton step
"""
function solve_complete_goop_pdip(
        player_G, all_F, all_vars,
        comp_idx_per_player, all_comp_idx;
        y0          = nothing,
        rho_init    = 0.1,
        rho_factor  = 0.1,
        rho_min     = 1e-10,
        max_outer   = 20,
        max_inner   = 100,
        tol_inner   = 1e-6,
        tol_outer   = 1e-8,
        c1          = 1e-4,
        beta_ls     = 0.5,
        verbose     = true)

    n    = length(all_vars)
    n_eq = length(all_F)

    verbose && println("Building numerical functions from symbolic expressions…")

    # ── Numerical F: R^n → R^{n_eq} ─────────────────────────────────────────
    F_fns = Symbolics.build_function(all_F, all_vars; expression = Val{false})
    F_num = F_fns[1]   # F_num(y::Vector{<:Real}) -> Vector

    # ── Numerical G: R^n → R^{n_G}  (for feasibility enforcement) ───────────
    G_sym = vcat(player_G...)
    G_fns = Symbolics.build_function(G_sym, all_vars; expression = Val{false})
    G_num = G_fns[1]   # G_num(y) -> Vector (must stay > 0)

    # ── Symbolic Jacobian ∂F/∂y: compiled once, reused every Newton step ─────
    verbose && println("Building symbolic Jacobian ($(n_eq)×$(n))…  (this may take a moment)")
    J_sym = Symbolics.jacobian(all_F, all_vars)
    J_fns = Symbolics.build_function(J_sym, all_vars; expression = Val{false})
    J_num = J_fns[1]   # J_num(y) -> Matrix
    verbose && println("Symbolic Jacobian ready.")

    # ── Complementarity mask ─────────────────────────────────────────────────
    # comp_mask[j] = 1  iff entry j of all_F is a complementarity condition.
    # Perturbed system: F_ρ(y) = F(y) - ρ · comp_mask,  so the j-th comp
    # condition becomes g_j · γ_j = ρ instead of 0.
    comp_mask        = zeros(n_eq)
    comp_mask[all_comp_idx] .= 1.0

    F_rho(y, rho)    = F_num(y) .- rho .* comp_mask

    # ── Initialise ────────────────────────────────────────────────────────────
    y   = isnothing(y0) ? fill(0.1, n) : copy(float.(y0))
    rho = rho_init

    verbose && println("="^65)
    verbose && @printf("  GOOP Complete KKT — PDIP Solver\n")
    verbose && @printf("  n_vars=%d  n_eq=%d  n_comp=%d  rho_0=%.2e\n",
                       n, n_eq, length(all_comp_idx), rho)
    verbose && println("="^65)

    res_history = Float64[]
    converged   = false

    # ── Outer loop: reduce ρ ─────────────────────────────────────────────────
    for outer in 1:max_outer
        verbose && @printf("\n  -- rho = %.2e  (outer %d/%d) --\n", rho, outer, max_outer)
        verbose && @printf("  %4s  %15s  %12s\n", "iter", "||F_rho(y)||", "step alpha")
        verbose && println("  " * "-"^55)

        inner_conv = false

        # ── Inner loop: Newton iterations at fixed ρ ──────────────────────
        for inner in 1:max_inner
            Main.@infiltrate
            Fy_rho = F_rho(y, rho)
            res    = norm(Fy_rho)
            push!(res_history, res)

            Main.@infiltrate

            if res < tol_inner
                verbose && @printf("  %4d  %15.6e  (inner converged)\n", inner, res)
                inner_conv = true
                break
            end

            # Newton direction: J(y) · Δy = -F_ρ(y)
            Jac = J_num(y)
            # @infiltrate
            # report when Jac has no full row rank
            # @warn if rank(Jac) < n_eq
            #     println("  Warning: Jacobian is rank-deficient at iteration $inner (rank $(rank(Jac)) < $n_eq)")
            # end
            Δy  = pinv(Jac) * (-Fy_rho)

            Main.@infiltrate

            # Armijo backtracking: keep G(y + α·Δy) > 0
            α = 1.0
            for _ls in 1:60
                y_cand = y .+ α .* Δy
                if all(G_num(y_cand) .> 0)
                    if norm(F_rho(y_cand, rho)) ≤ (1.0 - c1*α) * res
                        break
                    end
                end
                α *= beta_ls
                α < 1e-16 && break
            end

            verbose && @printf("  %4d  %15.6e  %12.8f\n", inner, res, α)
            y .+= α .* Δy
        end

        # ── Check full (unperturbed) residual ─────────────────────────────
        Fy_full  = F_num(y)
        res_full = norm(Fy_full)
        verbose && @printf("  -> ||F(y)|| = %.4e  (rho=%.2e)\n", res_full, rho)

        if res_full < tol_outer
            converged = true
            verbose && println("\n  Converged!")
            break
        end

        rho = max(rho * rho_factor, rho_min)
    end

    verbose && println("\n" * "="^65)
    verbose && @printf("  Final ||F(y)||   = %.6e\n", norm(F_num(y)))
    verbose && @printf("  min G(y)         = %.6e  (should be > 0)\n", minimum(G_num(y)))
    verbose && println("  converged        = $converged")

    return y, res_history
end

# ══════════════════════════════════════════════════════════════════════════════
# Example 1 — Quadratic objectives, linear constraints
# ══════════════════════════════════════════════════════════════════════════════
println("=" ^ 60)
println("Example 1: Quadratic objectives, linear constraints")
println("=" ^ 60)



let 
    N   = 2 # number of players
    K   = 2 # number of preference levels per player
    ni  = [2, 2] # number of decision variables per player
    mEi = [1, 1] # number of equality constraints per player
    mIi = [2, 2] # number of inequality constraints per player
    K_vec = fill(K, N)

    # Cost matrices/vectors (quadratic)
    Qmat = [
        [   # Player 1
            [0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0], # level 1
            [0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0], # level 2
        ],
        [   # Player 2
            [0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0;  0.0 0.0 2.0 0.0;  0.0 0.0 0.0 0.0], # level 1
            [0.0 0.0 0.0 0.0;  0.0 0.0 0.0 0.0;  0.0 0.0 1.0 0.0;  0.0 0.0 0.0 0.0], # level 2
        ],
    ]
    qvec = [
        [
            [0.0; 0.0; 0.0; 0.0], 
            [0.0; 0.0; 0.0; 0.0]
        ],
        [
            [0.0; 0.0; -1.0; 0.0], 
            [0.0; 0.0; -0.5; 0.0]
        ],
    ]

    # Linear constraint data
    # Ae * z^i = be  (equality)
    # Ag * z^i ≤ bg  (inequality)
    Ae = [[1.0  1.0], [1.0  1.0]]
    be = [[0.5],      [0.5]]
    Ag = [
        [
            1.0  0.0;  
            0.0  1.0
        ], 
        [
            1.0  0.0;  
            0.0  1.0
        ]]
    bg = [[100.0; 100.0 ],             [100.0; 100.0 ]]

    z_ranges_ex = [sum(ni[1:i-1])+1 : sum(ni[1:i]) for i in 1:N]

    J_fns = [
        [z -> (0.5 * (z' * Qmat[i][k] * z)[1] + (qvec[i][k]' * z)[1])  for k in 1:K]
        for i in 1:N
    ]
    h_fns = [z -> Ae[i] * z[z_ranges_ex[i]] .- be[i]  for i in 1:N]
    g_fns = [z -> bg[i] .- Ag[i] * z[z_ranges_ex[i]]  for i in 1:N]

    zv, player_F, player_G, player_vars, all_F, all_vars,
        comp_idx_per_player, all_comp_idx =
        build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)

    println("\nComplementarity indices per player:")
    for i in 1:N
        println("  Player $i: $(comp_idx_per_player[i])")
    end
    println("  All (in stacked all_F): $all_comp_idx")

    # ── Construct a feasible initial point ────────────────────────────────────
    # z = [0.25, 0.25, 0.25, 0.25]: satisfies h^i (0.25+0.25=0.5) and g^i > 0 (0.75)
    # Dual variables: λ = 0, γ's = 0.1 (positive to satisfy Ḡ > 0)
    n_tot = length(all_vars)
    y0 = fill(0.0, n_tot)

    # Primal z
    n_z = sum(ni)
    y0[1:n_z] .= 0.25

    # Set all γ variables > 0.  They appear in player_vars[i][ni[i]+mEi[i]+1 : end]
    # more precisely after the λK block.  We just set ALL dual entries to 0.1;
    # PDIP backtracking will fix any that should be zero.
    offset = n_z
    for i in 1:N
        n_extra = length(player_vars[i]) - ni[i]   # extras for player i
        y0[offset+1 : offset+n_extra] .= 0.1
        offset += n_extra
    end

    println("\n── Solving with PDIP ────────────────────────────────────────")
    y_sol, res_hist = solve_complete_goop_pdip(
        player_G, all_F, all_vars,
        comp_idx_per_player, all_comp_idx;
        y0         = y0,
        rho_init   = 1,
        rho_factor = 0.1,
        rho_min    = 1e-10,
        max_outer  = 1,
        max_inner  = 50,
        tol_inner  = 1e-7,
        tol_outer  = 1e-8,
    )

    println("\n── Solution ─────────────────────────────────────────────────")

    plot(res_hist, yscale=:log10, xlabel="Iteration", ylabel="||F_ρ(y)||", title="PDIP Convergence")
    savefig("goop_complete_kkt_pdip_convergence.png")

    # Expected: z^1* = [0.5, 0.0], z^2* = [0.5, 0.0]
    # (Level-2 KKT uniquely pins each player to their unconstrained optimum
    #  on the equality h^i, which is independent of the other player.)
    @printf("  z^1* = [%.6f, %.6f]  (expected [0.5, 0.0])\n",
            y_sol[1], y_sol[2])
    @printf("  z^2* = [%.6f, %.6f]  (expected [0.5, 0.0])\n",
            y_sol[3], y_sol[4])
end



# # ══════════════════════════════════════════════════════════════════════════════
# # Example 2 — Non-quadratic objectives, nonlinear constraints
# # ══════════════════════════════════════════════════════════════════════════════
# println("\n" * "=" ^ 60)
# println("Example 2: Non-quadratic objectives, nonlinear constraints")
# println("=" ^ 60)

# let
#     N   = 2
#     K   = 2
#     ni  = [2, 2]
#     mEi = [1, 1]
#     mIi = [1, 1]
#     K_vec = fill(K, N)

#     J_fns = [
#         [
#             z -> (z[1]^2 + z[2]^2)^2 + z[3]*z[1],
#             z -> exp(z[1]) + z[2]^2,
#         ],
#         [
#             z -> (z[3] - 1)^4 + z[4]^2,
#             z -> z[3]^2 * z[4]^2 + z[3],
#         ],
#     ]
#     h_fns = [
#         z -> [z[1]^2 + z[2]^2 - 1],
#         z -> [z[3] + z[4]^3 - 0.5],
#     ]
#     g_fns = [
#         z -> [1 - z[1]^4],
#         z -> [z[3] * (1 - z[4]^2)],
#     ]

#     zv, player_F, player_G, player_vars, all_F, all_vars,
#         comp_idx_per_player, all_comp_idx =
#         build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)

#     println("\nSystem built ($(length(all_F)) equations, $(length(all_vars)) variables)")
#     println("Complementarity indices (all_F): $all_comp_idx")
# end












