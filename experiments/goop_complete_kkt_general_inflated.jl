# goop_complete_kkt_general.jl
#
# Complete GOOP KKT system — N players, K preference levels each.
# Following Section 3.2 (eq. 3.8–3.12) of:
# "Breaking Exponential Complexity in Games of Ordered Preference"
#
# ── Coupled constraints ────────────────────────────────────────────────────────
# h^i and g^i may depend on the FULL joint state z = [z^1;…;z^N].
# Stationarity is computed as a PARTIAL derivative w.r.t. z^i only:
#   ∂L̄^i_K/∂z^i = ∇_{z^i}J^i_K − (∂h^i/∂z^i)ᵀ λ̄^i_K
#                                 − (∂g^i/∂z^i)ᵀ γ̄^i_K = 0
# Symbolics.gradient(L_K, zi) computes this correctly for any expression of z,
# treating z^j (j≠i) as parameters.
#
# ── Recursive construction (Section 3.2) ──────────────────────────────────────
#   Level K:
#     F̄^i_K = [∇_{z^i}L̄^i_K;  h^i(z);  g^i(z)⊙γ̄^i_K] = 0          (3.9)
#     Ḡ^i_K = [g^i(z);  γ̄^i_K]  ≥ 0                                  (3.10)
#   Level k < K:
#     F̄^i_k = [∇_{vars^i}L̄^i_k;  Ḡ^i_{k+1}⊙γ̄^i_k;  F̄^i_{k+1}] = 0 (3.11)
#     Ḡ^i_k = [Ḡ^i_{k+1};  γ̄^i_k]  ≥ 0                              (3.12)
#
# ── PDIP formulation ──────────────────────────────────────────────────────────
# Every complementarity condition  a_j(y)·b_j(y) = 0  is split into:
#   Primal feasibility:      a_j(y) − s_j = 0          (s_j > 0 slack)
#   Perturbed complementarity:  s_j · b_j(y) = ρ       (drives to 0)
# Augmented variables: w = [y; s].  Line search keeps s > 0 and b_j(y) > 0.
# ──────────────────────────────────────────────────────────────────────────────

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
    build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)

Build the complete symbolic GOOP KKT system following Section 3.2.
Coupled constraints (h^i, g^i depending on all players' z) are supported.

# Returns (10-tuple)
- `zv`                  : joint symbolic variable [z^1;…;z^N]
- `player_F[i]`         : F̄^i_1 — complete KKT equations for player i
- `player_G[i]`         : Ḡ^i_1 — inequality system for player i
- `player_vars[i]`      : all symbolic variables for player i
- `all_F`               : stacked [F̄^1_1;…;F̄^N_1]
- `all_vars`            : stacked variable vector
- `comp_idx_per_player` : indices into player_F[i] that are comp. conditions
- `all_comp_idx`        : same indices into all_F
- `all_comp_a`          : symbolic `a_j(y)` for each comp. condition a_j·b_j=0
- `all_comp_b`          : symbolic `b_j(y)` (the γ dual variable) for each pair
"""
function build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)
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

        # ── Level K (innermost) ───────────────────────────────────────────────
        λK_v = [Symbolics.variable(Symbol("lK_p$(i)_$j")) for j in 1:mEi[i]]
        γK_v = [Symbolics.variable(Symbol("gK_p$(i)_$j")) for j in 1:mIi[i]]

        h_val = h_fns[i](zv)
        g_val = g_fns[i](zv)

        L_K    = J_fns[i][Ki](zv) - sum(λK_v .* h_val) - sum(γK_v .* g_val)
        ∇zi_LK = Symbolics.gradient(L_K, zi)

        F_cur    = [∇zi_LK; h_val; g_val .* γK_v]   # F̄^i_K  (3.9)
        G_cur    = [g_val;  γK_v]                    # Ḡ^i_K  (3.10)
        vars_cur = [zi; λK_v; γK_v]

        # Complementarity index tracking
        comp_indices_i = collect(ni[i] + mEi[i] + 1 : ni[i] + mEi[i] + mIi[i])

        # Complementarity pair tracking: (a, b) for each pair a·b = 0
        # At level K: a = g_val, b = γK_v
        comp_a_i = collect(Num, g_val)
        comp_b_i = collect(Num, γK_v)

        # ── Recursive levels k = K-1 down to 1 ───────────────────────────────
        for k in Ki-1:-1:1
            n_G = length(G_cur)
            n_F = length(F_cur)

            η_v = [Symbolics.variable(Symbol("eta_p$(i)_k$(k)_$j")) for j in 1:n_F]
            γ_v = [Symbolics.variable(Symbol("gam_p$(i)_k$(k)_$j")) for j in 1:n_G]

            L_k    = J_fns[i][k](zv) - sum(η_v .* F_cur) - sum(γ_v .* G_cur)
            ∇_Lk   = Symbolics.gradient(L_k, vars_cur)
            comp_k = G_cur .* γ_v

            # Update index tracking BEFORE prepending new block
            # New F = [∇_Lk (|vars_cur|);  comp_k (n_G);  F_cur (n_F)]
            offset_new     = length(vars_cur) + n_G
            new_comp_here  = collect(length(vars_cur) + 1 : length(vars_cur) + n_G)
            comp_indices_i = [new_comp_here; comp_indices_i .+ offset_new]

            # At level k: comp pairs are (G_cur[j], γ_v[j]) for j in 1:n_G.
            # These are PREPENDED to match the order of comp_indices_i.
            comp_a_i = [collect(Num, G_cur); comp_a_i]
            comp_b_i = [collect(Num, γ_v);   comp_b_i]

            F_cur    = [∇_Lk; comp_k; F_cur]
            G_cur    = [G_cur; γ_v]
            vars_cur = [vars_cur; η_v; γ_v]
        end

        player_F[i]            = F_cur
        player_G[i]            = G_cur
        player_vars[i]         = vars_cur
        comp_idx_per_player[i] = comp_indices_i
        comp_a_per_player[i]   = comp_a_i
        comp_b_per_player[i]   = comp_b_i

        expected = 2^(Ki-1) * (ni[i] + mEi[i] + Ki * mIi[i])
        println("Player $i: |F̄^$(i)_1| = $(length(F_cur))  (expected $expected per Prop 3.4)")
    end

    # ── Stacked system ────────────────────────────────────────────────────────
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

    all_comp_a = vcat(comp_a_per_player...)   # a_j: first factor of j-th comp pair
    all_comp_b = vcat(comp_b_per_player...)   # b_j: second factor (γ variable)

    println("\n── Dimension check (Proposition 3.4) ──")
    N_exp = sum(2^(K_vec[i]-1) * (ni[i] + mEi[i] + K_vec[i]*mIi[i]) for i in 1:N)
    G_exp = sum(2^K_vec[i] * mIi[i] for i in 1:N)
    @printf("  |F̄|    = %d  (expected %d)\n", length(all_F), N_exp)
    @printf("  |vars|  = %d  (expected %d)\n", length(all_vars), N_exp)
    @printf("  |Ḡ|    = %d  (expected %d)\n", sum(length.(player_G)), G_exp)
    @printf("  |comp|  = %d  (comp pairs a·b=0)\n", length(all_comp_idx))
    @assert length(all_F)    == N_exp "Equation count mismatch"
    @assert length(all_vars) == N_exp "Variable count mismatch"
    @assert length(all_comp_a) == length(all_comp_idx) "comp_a length mismatch"
    @assert length(all_comp_b) == length(all_comp_idx) "comp_b length mismatch"
    println("  All assertions passed ✓")

    return zv, player_F, player_G, player_vars, all_F, all_vars,
           comp_idx_per_player, all_comp_idx, all_comp_a, all_comp_b
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PDIP solver with slack variables  s ⊙ b = ρ  (Algorithm 6.1)
# ═══════════════════════════════════════════════════════════════════════════════
"""
    solve_complete_goop_pdip(all_F, all_vars,
                             all_comp_idx, all_comp_a, all_comp_b; ...)

Solve  F̄(y) = 0,  Ḡ(y) ≥ 0  with a slack-variable Primal-Dual Interior Point
method.  Every complementarity condition  a_j(y)·b_j(y) = 0  is replaced by

    a_j(y) − s_j = 0          (primal feasibility for the slack)
    s_j · b_j(y) = ρ          (perturbed complementarity; b_j are the γ duals)

Augmented variable: `w = [y; s]` ∈ R^{n + n_comp}.
Augmented perturbed system `K_ρ(w) = 0` (size n + n_comp):

    F̄_noncomp(y)      = 0     (n − n_comp equations; stationarity + equalities)
    a(y) − s          = 0     (n_comp equations)
    s ⊙ b(y) − ρ·1   = 0     (n_comp equations)

Augmented Jacobian ∂K_ρ/∂w (assembled once per Newton step from pre-compiled
symbolic sub-Jacobians):

    [ ∂F̄_nc/∂y      0       ]
    [ ∂a/∂y          −I      ]
    [ diag(s)·∂b/∂y  diag(b) ]

Line search enforces `s + α·Δs > 0` and `b(y + α·Δy) > 0` at every step.

# Arguments
- `all_F`, `all_vars`         : stacked system (from builder)
- `all_comp_idx`              : complementarity indices into all_F (from builder)
- `all_comp_a`, `all_comp_b`  : symbolic a/b for each pair (from builder)

# Keyword arguments
| kwarg        | meaning                                          | default  |
|:------------ |:------------------------------------------------ |:-------- |
| `y0`         | initial y; must have b(y0) > 0 and a(y0) ≥ 0   | 0.1·ones |
| `s0`         | initial slacks; must be > 0  (default: ones)    | nothing  |
| `rho_init`   | initial ρ                                        | 0.1      |
| `rho_factor` | multiply ρ by this each outer step               | 0.1      |
| `rho_min`    | stop reducing ρ below this                       | 1e-10    |
| `max_outer`  | outer ρ-reduction iterations                     | 30       |
| `max_inner`  | Newton iterations per outer step                 | 100      |
| `tol_inner`  | inner convergence: ‖K_ρ(w)‖ < tol_inner        | 1e-6     |
| `tol_outer`  | outer convergence: ‖K_0(w)‖ < tol_outer        | 1e-8     |
| `c1`         | Armijo constant                                  | 1e-4     |
| `beta_ls`    | step-size backtracking factor                    | 0.5      |
| `verbose`    | print iteration log                              | true     |

# Returns
- `y_sol`       : solution in the ordering of `all_vars`
- `s_sol`       : slack variables at solution (should satisfy s ≈ a(y_sol))
- `res_history` : ‖K_ρ(w)‖ at every inner Newton step
"""
function solve_complete_goop_pdip(
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

    n      = length(all_vars)
    n_eq   = length(all_F)
    n_comp = length(all_comp_idx)
    n_aug  = n + n_comp             # augmented dimension
    noncomp_idx = setdiff(1:n_eq, all_comp_idx)

    verbose && println("Compiling numerical functions…")

    # ── Non-complementarity part of F̄ ────────────────────────────────────────
    F_nc_sym  = all_F[noncomp_idx]   # stationarity + equality constraint rows
    F_nc_fns  = Symbolics.build_function(F_nc_sym, all_vars; expression = Val{false})
    F_nc_num  = F_nc_fns[1]          # y → R^{n − n_comp}

    # ── a_j(y) and b_j(y) for each complementarity pair ─────────────────────
    a_fns = Symbolics.build_function(all_comp_a, all_vars; expression = Val{false})
    a_num = a_fns[1]   # y → R^{n_comp}   (the "primal" side: g or inner γ)

    b_fns = Symbolics.build_function(all_comp_b, all_vars; expression = Val{false})
    b_num = b_fns[1]   # y → R^{n_comp}   (the γ dual variable side)

    # ── Symbolic sub-Jacobians (compiled once) ────────────────────────────────
    verbose && println("Building symbolic Jacobians…  (may take a moment)")
    J_nc_fns = Symbolics.build_function(
        Symbolics.jacobian(F_nc_sym, all_vars), all_vars; expression = Val{false})
    J_nc_num = J_nc_fns[1]   # y → R^{(n-n_comp) × n}

    J_a_fns  = Symbolics.build_function(
        Symbolics.jacobian(all_comp_a, all_vars), all_vars; expression = Val{false})
    J_a_num  = J_a_fns[1]    # y → R^{n_comp × n}

    J_b_fns  = Symbolics.build_function(
        Symbolics.jacobian(all_comp_b, all_vars), all_vars; expression = Val{false})
    J_b_num  = J_b_fns[1]    # y → R^{n_comp × n}
    verbose && println("Symbolic Jacobians ready.")

    # ── Perturbed augmented system K_ρ(w) = 0,  w = [y; s] ──────────────────
    #   F_nc(y)        = 0      (stationarity + equalities)
    #   a(y) − s       = 0      (primal feasibility)
    #   s ⊙ b(y) − ρ  = 0      (perturbed complementarity)
    function K_rho(w, rho)
        y = w[1:n];  s = w[n+1:end]
        [F_nc_num(y);  a_num(y) .- s;  s .* b_num(y) .- rho]
    end

    # ── Augmented Jacobian ∂K_ρ/∂w ───────────────────────────────────────────
    #   [ ∂F_nc/∂y      0_nc×nc  ]    (n-n_comp rows)
    #   [ ∂a/∂y         −I_nc    ]    (n_comp rows)
    #   [ diag(s)·∂b/∂y  diag(b) ]    (n_comp rows)
    function J_aug(w)
        y  = w[1:n];  s = w[n+1:end]
        bv = b_num(y)
        Jnc = J_nc_num(y)                           # (n-n_comp) × n
        Ja  = J_a_num(y)                            # n_comp × n
        Jb  = J_b_num(y)                            # n_comp × n
        [Jnc                    zeros(n - n_comp, n_comp);
         Ja                     -I(n_comp);
         diagm(s) * Jb          diagm(bv)]
    end

    # ── Initialise w = [y; s] ─────────────────────────────────────────────────
    y_init = isnothing(y0) ? fill(0.1, n) : copy(float.(y0))

    # Default s0: start at a(y0) if a(y0) > 0, else at rho_init.
    # This makes the feasibility residual a(y0) − s0 = 0 (ideal start).
    a0 = a_num(y_init)
    s_init = if isnothing(s0)
        max.(a0, fill(rho_init, n_comp))
    else
        copy(float.(s0))
    end
    @assert all(s_init .> 0) "Initial slacks s0 must be strictly positive"
    @assert all(b_num(y_init) .> 0) "Initial b(y0) must be strictly positive (γ duals > 0)"

    w   = [y_init; s_init]
    rho = rho_init

    verbose && println("="^65)
    verbose && @printf("  GOOP Complete KKT — PDIP  (s⊙b = ρ formulation)\n")
    verbose && @printf("  n=%d  n_comp=%d  n_aug=%d  rho_0=%.2e\n",
                       n, n_comp, n_aug, rho)
    verbose && println("="^65)

    res_history = Float64[]
    converged   = false

    # ── Full (unperturbed) residual for outer convergence check ───────────────
    # We check ‖K_0(w)‖ = ‖[F_nc(y); a(y)-s; s⊙b(y)]‖ → 0
    K0(w) = K_rho(w, 0.0)

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

            # Newton direction: J(w)·Δw = −K_ρ(w)
            J = J_aug(w)
            # @infiltrate
            Δw = pinv(J) * (-Kw)

            Δy = Δw[1:n];  Δs = Δw[n+1:end]
            y_curr = w[1:n];  s_curr = w[n+1:end]

            # backtracking: enforce  s + α·Δs > 0  and  b(y+α·Δy) > 0
            α = 1.0
            for _ in 1:60
                y_new = y_curr .+ α .* Δy
                s_new = s_curr .+ α .* Δs
                if all(s_new .> 0) && all(b_num(y_new) .> 0)
                    if norm(K_rho([y_new; s_new], rho)) ≤ res
                        break
                    end
                end
                α *= beta_ls
                α < 1e-16 && break
            end

            verbose && @printf("  %4d  %15.6e  %12.8f\n", inner, res, α)
            w .+= α .* Δw
        end

        # Outer convergence: full (unperturbed) KKT residual
        res_full = norm(K0(w))
        verbose && @printf("  -> ||K0(w)|| = %.4e  (rho=%.2e)\n", res_full, rho)

        if res_full < tol_outer
            converged = true
            verbose && println("\n  Converged!")
            break
        end

        rho = max(rho * rho_factor, rho_min)
    end

    y_sol = w[1:n]
    s_sol = w[n+1:end]

    verbose && println("\n" * "="^65)
    verbose && @printf("  Final ||K0(w)||  = %.6e\n", norm(K0(w)))
    verbose && @printf("  min s            = %.6e  (> 0 required)\n", minimum(s_sol))
    verbose && @printf("  min b(y)         = %.6e  (> 0 required)\n", minimum(b_num(y_sol)))
    verbose && @printf("  max |a(y) - s|   = %.6e  (slack feasibility)\n",
                       maximum(abs.(a_num(y_sol) .- s_sol)))
    verbose && println("  converged        = $converged")

    return y_sol, s_sol, res_history
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Example 
println("=" ^ 70)
println("GOOP — Coupled Inflated Example  (PDIP with s ⊙ b = ρ)")
println("  N=2 players, K=2 levels, ni=[2,2], mEi=[1,1], mIi=[4,4]")
println("=" ^ 70)

let
    N     = 2
    K     = 2
    ni    = [2, 2]
    mEi   = [1, 1]
    mIi   = [4, 4]
    K_vec = fill(K, N)

    # z[1]=x₁, z[2]=y₁  (player 1)
    # z[3]=x₂, z[4]=y₂  (player 2)
    Qmat = [
        [   # Player 1
            [0.416968 0.944866 1.048411 0.140553; 0.944866 2.141105 2.37574 0.318499; 1.048411 2.37574 2.636089 0.353402; 0.140553 0.318499 0.353402 0.047378], # level 1
            [1.799915 -0.55296 -0.795839 1.030905; -0.55296 0.169877 0.244493 -0.316709; -0.795839 0.244493 0.351883 -0.455818; 1.030905 -0.316709 -0.455818 0.590452], # level 2
            
        ],
        [   # Player 2
            [0.042467 -0.209993 0.104132 -0.11093; -0.209993 1.03839 -0.514919 0.548533; 0.104132 -0.514919 0.255339 -0.272008; -0.11093 0.548533 -0.272008 0.289765], # level 1
            [0.011885 0.029826 0.006225 0.043515; 0.029826 0.074851 0.015621 0.109203; 0.006225 0.015621 0.00326 0.02279; 0.043515 0.109203 0.02279 0.159321], # level 2
        ],
    ]
    qvec = [
        [
            [1.433707; 3.248838; 3.604866; 0.483279], 
            [-0.444909; 0.136682; 0.196718; -0.254822]
        ],
        [
            [0.110712; -0.547456; 0.271474; -0.289196], 
            [0.045815; 0.114977; 0.023995; 0.167745]
        ],
    ]
    J_fns = [
        [z -> (0.5 * (z' * Qmat[i][k] * z)[1] + (qvec[i][k]' * z)[1])  for k in 1:K]
        for i in 1:N
    ]


    # ── Coupled equality constraints: h^i(z) = 0 ─────────────────────────────
    h_fns = [
        z -> [z[1] + z[2] + z[3] + z[4] - 0.0],   # h¹(z) = 0
        z -> [z[1] + z[2] + z[3] + z[4] - 0.0],   # h²(z) = 0
    ]

    # ── Coupled inequality constraints: g^i(z) ≥ 0 ───────────────────────────
    g_fns = [
        z -> [1.0 - z[1]; 1.0 - z[2]; 1.0 - z[3]; 1.0 - z[4]],   # g¹(z) ≥ 0, for player 1
        z -> [1.0 - z[1]; 1.0 - z[2]; 1.0 - z[3]; 1.0 - z[4]],   # g²(z) ≥ 0, for player 2
    ]

    # ── Build symbolic KKT system ─────────────────────────────────────────────
    println("\n── Building symbolic GOOP KKT system ────────────────────────────────")
    zv, player_F, player_G, player_vars, all_F, all_vars,
        comp_idx_per_player, all_comp_idx, all_comp_a, all_comp_b =
        build_complete_goop_kkt(N, K_vec, ni, mEi, mIi, J_fns, h_fns, g_fns)

    println("\nComplementarity pairs (a·b = 0):")
    for i in 1:N
        println("  Player $i indices into player_F: $(comp_idx_per_player[i])")
    end
    println("  Stacked indices into all_F: $all_comp_idx")

    # ── Initial point ─────────────────────────────────────────────────────────
    # y0: z = [0.2, 0.2, -0.2, 0.2], all duals = 0.1
    #   g¹(z0) = 4 - 0.04 - 0.5*0.04 = 3.94 > 0 ✓
    #   g²(z0) = 4 - 0.04 - 0.5*0.04 = 3.94 > 0 ✓
    # All γ duals = 0.1 > 0  ⟹  b(y0) = [γ's] > 0 ✓
    n_tot = length(all_vars)
    n_z   = sum(ni)

    y0        = fill(0.0, n_tot)
    y0[1:4]  .= [0.2, 0.2, 0.2, 0.2]
    off = n_z
    for i in 1:N
        n_extra = length(player_vars[i]) - ni[i]
        y0[off+1 : off+n_extra] .= 0.1
        off += n_extra
    end

    # ── Solve ─────────────────────────────────────────────────────────────────
    println("\n── Running PDIP solver  (s ⊙ b = ρ) ────────────────────────────────")
    y_sol, s_sol, res_hist = solve_complete_goop_pdip(
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
    )

    # ── Report ────────────────────────────────────────────────────────────────
    println("\n── Solution ─────────────────────────────────────────────────────────")
    x1, y1, x2, y2 = y_sol[1], y_sol[2], y_sol[3], y_sol[4]
    @printf("  z¹* = (x₁, y₁) = (%.6f, %.6f)\n", x1, y1)
    @printf("  z²* = (x₂, y₂) = (%.6f, %.6f)\n", x2, y2)

    println("\nConstraint satisfaction at z*:")
    @printf("  h¹(z*) = % .2e  (should be 0)\n", x1^2 + y1^2 + 0.2*x2*y2 - 1.0)
    @printf("  h²(z*) = % .2e  (should be 0)\n", x2^2 + y2^2 + 0.2*x1*y1 - 1.0)
    @printf("  g¹(z*) = %.6f  (> 0 required)\n", 4.0 - x1^2 - 0.5*x2^2)
    @printf("  g²(z*) = %.6f  (> 0 required)\n", 4.0 - x2^2 - 0.5*x1^2)

    println("\nObjectives at z*:")
    @printf("  J¹₁ = %.6f\n", (x1-1)^4)
    @printf("  J¹₂ = %.6f\n", (x1-0.5)^2)
    @printf("  J²₁ = %.6f\n", (x2+1)^4)
    @printf("  J²₂ = %.6f\n", (x2+0.5)^2)

    if !isempty(res_hist)
        plt = plot(res_hist;
                   yscale     = :log10,
                   xlabel     = "Newton iteration",
                   ylabel     = "||K_rho(w)||",
                   title      = "GOOP PDIP Convergence  (s⊙b = ρ, coupled nonlinear)",
                   legend     = false,
                   lw         = 2,
                   marker     = :circle,
                   ms         = 3,
                   grid       = true,
                   framestyle = :box)
        savefig(plt, "goop_coupled_inflated_convergence.pdf")
        println("\nConvergence plot → goop_coupled_inflated_convergence.pdf")
    end
end
