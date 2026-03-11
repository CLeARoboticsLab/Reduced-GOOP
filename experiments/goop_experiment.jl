# goop_experiments.jl
#
# Complete vs. Reduced GOOP KKT solvers on two problem classes:
#   (A) Nonlinear: coupled-quartic costs (a^T z)^4/n^3 + b^T z,  linear constraints
#   (B) LQ:        quadratic costs,                               linear constraints
#
# Each (problem type × KKT formulation) is compiled ONCE as a parametric system.
# Random instances share the same symbolic structure; only the numerical
# parameter vector is substituted at solve time — no recompilation per instance.
#
# Compilation count: 4  (NL×Complete, NL×Reduced, LQ×Complete, LQ×Reduced)
# ──────────────────────────────────────────────────────────────────────────────

using Symbolics, LinearAlgebra, Printf, Random, Statistics, Plots, LaTeXStrings
ENV["GKSwstype"] = "nul"

const _SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(_SRC, "complete_kkt.jl"))   # → build_complete_goop_kkt
include(joinpath(_SRC, "reduced_kkt.jl"))    # → build_reduced_goop_kkt

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ParametricGOOP — compiled once, params substituted at solve time
# ═══════════════════════════════════════════════════════════════════════════════

if !@isdefined(ParametricGOOP)
"""
    ParametricGOOP

KKT system compiled ONCE for a fixed problem structure (N, K, ni, mEi, mIi).
The six numerical functions accept `(y, params)` where `params` is the flat
vector produced by the corresponding `pack_*_params` function.
"""
struct ParametricGOOP
    name      :: String
    n         :: Int
    n_comp    :: Int
    n_nc      :: Int
    dual_dims :: Vector{Int}
    N         :: Int
    K         :: Int
    ni        :: Vector{Int}
    mEi       :: Vector{Int}
    mIi       :: Vector{Int}
    F_nc_num  :: Function        # (y, params) → R^{n_nc}
    a_num     :: Function        # (y, params) → R^{n_comp}
    b_num     :: Function        # (y, params) → R^{n_comp}
    J_nc_num  :: Function        # (y, params) → R^{n_nc × n}
    J_a_num   :: Function        # (y, params) → R^{n_comp × n}
    J_b_num   :: Function        # (y, params) → R^{n_comp × n}
end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Symbolic function factories
#     Each returns (J_fns, h_fns, g_fns, sym_params).
#     sym_params order must match the corresponding pack_*_params function.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    make_nl_sym_fns(N, K, ni, mEi, mIi)

Symbolic parametric functions for the nonlinear structure:
  J^i_k(z) = (a^i_k)ᵀ z)^4 / n_total^3 + (b^i_k)ᵀ z   (coupled quartic + linear)
  h^i(z)   = A^i z − b^i = 0                             (linear equality)
  g^i(z)   = d^i − C^i z ≥ 0                            (linear inequality)

The quartic term (a^T z)^4 / n^3 is non-quadratic, involves all variables
coupled through the inner product, and stays O(1) for ||a||, ||z|| = O(1).
Using linear constraints removes the rank-loss issue of cubic Jacobians near zero.
"""
function make_nl_sym_fns(N, K, ni, mEi, mIi)
    n_total = sum(ni)
    n3      = n_total^3

    mksym(name, n)     = Num[Symbolics.variable(name, j)    for j in 1:n]
    mksym2(name, r, c) = [Symbolics.variable(name, i, j) for i in 1:r, j in 1:c]

    a_sym  = [[mksym(Symbol("aNL_i$(i)k$(k)"), n_total) for k in 1:K] for i in 1:N]
    b_sym  = [[mksym(Symbol("bNL_i$(i)k$(k)"), n_total) for k in 1:K] for i in 1:N]
    A_sym  = [mksym2(Symbol("ANL_i$(i)"), mEi[i], n_total) for i in 1:N]
    bE_sym = [mksym(Symbol("bENL_i$(i)"), mEi[i])          for i in 1:N]
    C_sym  = [mksym2(Symbol("CNL_i$(i)"), mIi[i], n_total) for i in 1:N]
    d_sym  = [mksym(Symbol("dNL_i$(i)"),  mIi[i])          for i in 1:N]

    J_fns = [[let a = a_sym[i][k], b = b_sym[i][k]
                  z -> dot(a, z)^4 / n3 + dot(b, z)
              end for k in 1:K] for i in 1:N]
    h_fns = [let A = A_sym[i], bE = bE_sym[i]; z -> collect(A * z .- bE)  end for i in 1:N]
    g_fns = [let C = C_sym[i], d = d_sym[i];   z -> collect(d .- C * z)   end for i in 1:N]

    sym_params = Num[]
    for i in 1:N
        for k in 1:K
            append!(sym_params, a_sym[i][k]);  append!(sym_params, b_sym[i][k])
        end
        append!(sym_params, vec(A_sym[i]));  append!(sym_params, bE_sym[i])
        append!(sym_params, vec(C_sym[i]));  append!(sym_params, d_sym[i])
    end
    return J_fns, h_fns, g_fns, sym_params
end

"""
    make_lq_sym_fns(N, K, ni, mEi, mIi)

Symbolic parametric functions for the LQ structure:
  J^i_k(z) = ½ zᵀ Q^i_k z + (q^i_k)ᵀ z   [Q^i_k PSD]
  h^i(z)   = A^i z − b^i = 0
  g^i(z)   = d^i − C^i z ≥ 0
"""
function make_lq_sym_fns(N, K, ni, mEi, mIi)
    n_total = sum(ni)

    mksym(name, n)     = Num[Symbolics.variable(name, j)    for j in 1:n]
    mksym2(name, r, c) = [Symbolics.variable(name, i, j) for i in 1:r, j in 1:c]

    Q_sym = [[mksym2(Symbol("QLQ_i$(i)k$(k)"), n_total, n_total) for k in 1:K] for i in 1:N]
    q_sym = [[mksym(Symbol("qLQ_i$(i)k$(k)"), n_total)           for k in 1:K] for i in 1:N]
    A_sym = [mksym2(Symbol("ALQ_i$(i)"), mEi[i], n_total) for i in 1:N]
    b_sym = [mksym(Symbol("bLQ_i$(i)"), mEi[i])           for i in 1:N]
    C_sym = [mksym2(Symbol("CLQ_i$(i)"), mIi[i], n_total) for i in 1:N]
    d_sym = [mksym(Symbol("dLQ_i$(i)"), mIi[i])           for i in 1:N]

    J_fns = [[let Qik = Q_sym[i][k], qik = q_sym[i][k]
                  z -> 0.5 * dot(z, Qik * z) + dot(qik, z)
              end for k in 1:K] for i in 1:N]
    h_fns = [let A = A_sym[i], b = b_sym[i]; z -> collect(A * z .- b)  end for i in 1:N]
    g_fns = [let C = C_sym[i], d = d_sym[i]; z -> collect(d .- C * z)  end for i in 1:N]

    sym_params = Num[]
    for i in 1:N
        for k in 1:K
            append!(sym_params, vec(Q_sym[i][k]));  append!(sym_params, q_sym[i][k])
        end
        append!(sym_params, vec(A_sym[i]));  append!(sym_params, b_sym[i])
        append!(sym_params, vec(C_sym[i]));  append!(sym_params, d_sym[i])
    end
    return J_fns, h_fns, g_fns, sym_params
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Parametric compiler
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compile_parametric_goop(N, K, ni, mEi, mIi; builder, make_sym_fns, verbose)
        → (ParametricGOOP, compile_time_s)

Compile the GOOP KKT system once for a fixed structure.
- `make_sym_fns` : `make_nl_sym_fns` or `make_lq_sym_fns`
- `builder`      : `build_complete_goop_kkt` or `build_reduced_goop_kkt`
"""
function compile_parametric_goop(N, K, ni, mEi, mIi;
                                  builder,
                                  make_sym_fns,
                                  verbose = true)
    name = builder === build_complete_goop_kkt ? "Complete" : "Reduced"
    t0   = time()

    verbose && println("  [compile:$name] Building symbolic KKT (N=$N, K=$K)…")
    J_fns, h_fns, g_fns, sym_params = make_sym_fns(N, K, ni, mEi, mIi)

    _, _, _, player_vars, all_F, all_vars, _, all_comp_idx, all_comp_a, all_comp_b =
        builder(N, fill(K, N), ni, mEi, mIi, J_fns, h_fns, g_fns)

    n_comp      = length(all_comp_idx)
    noncomp_idx = setdiff(1:length(all_F), all_comp_idx)
    F_nc_sym    = all_F[noncomp_idx]
    yp_sym      = [all_vars; sym_params]

    Main.@infiltrate

    verbose && println("  [compile:$name] Compiling residual and Jacobian functions…")

    F_nc_fn = Symbolics.build_function(F_nc_sym,   yp_sym; expression=Val{false})[1]
    a_fn    = Symbolics.build_function(all_comp_a, yp_sym; expression=Val{false})[1]
    b_fn    = Symbolics.build_function(all_comp_b, yp_sym; expression=Val{false})[1]

    J_nc_fn = Symbolics.build_function(
        Symbolics.jacobian(F_nc_sym,   all_vars), yp_sym; expression=Val{false})[1]
    J_a_fn  = Symbolics.build_function(
        Symbolics.jacobian(all_comp_a, all_vars), yp_sym; expression=Val{false})[1]
    J_b_fn  = Symbolics.build_function(
        Symbolics.jacobian(all_comp_b, all_vars), yp_sym; expression=Val{false})[1]

    n         = length(all_vars)
    n_nc      = length(F_nc_sym)
    dual_dims = [length(player_vars[i]) - ni[i] for i in 1:N]
    t_compile = time() - t0

    verbose && @printf("  [compile:%s] Done.  |y|=%d  |comp|=%d  |params|=%d  t=%.2fs\n",
                       name, n, n_comp, length(sym_params), t_compile)

    wrap(f) = (y, p) -> f([y; p])
    pg = ParametricGOOP(
        name, n, n_comp, n_nc, dual_dims, N, K, ni, mEi, mIi,
        wrap(F_nc_fn), wrap(a_fn), wrap(b_fn),
        wrap(J_nc_fn), wrap(J_a_fn), wrap(J_b_fn))
    return pg, t_compile
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Problem data structs, generators, and pack functions
# ═══════════════════════════════════════════════════════════════════════════════

if !@isdefined(NLGOOPData)
struct NLGOOPData
    a  :: Vector{Vector{Vector{Float64}}}   # a[i][k] : linear projection vector for quartic
    b  :: Vector{Vector{Vector{Float64}}}   # b[i][k] : linear cost
    A  :: Vector{Matrix{Float64}}            # A[i]    : mEi[i] × n_total  (linear equality)
    bE :: Vector{Vector{Float64}}            # bE[i]   : mEi[i]
    C  :: Vector{Matrix{Float64}}            # C[i]    : mIi[i] × n_total  (linear inequality)
    d  :: Vector{Vector{Float64}}            # d[i]    : mIi[i]
    z0 :: Vector{Float64}
end
end

if !@isdefined(LQGOOPData)
struct LQGOOPData
    Q  :: Vector{Vector{Matrix{Float64}}}        # Q[i][k] : n_total × n_total PSD
    q  :: Vector{Vector{Vector{Float64}}}        # q[i][k] : linear cost
    A  :: Vector{Matrix{Float64}}                # A[i]    : mEi[i] × n_total
    b  :: Vector{Vector{Float64}}                # b[i]    : mEi[i]
    C  :: Vector{Matrix{Float64}}                # C[i]    : mIi[i] × n_total
    d  :: Vector{Vector{Float64}}                # d[i]    : mIi[i]
    z0 :: Vector{Float64}
end
end

"""
    generate_random_nl_goop(N, K, ni, mEi, mIi, rng) → NLGOOPData

Nonlinear instance: 4th-power extension of the LQ quadratic example.

  J^i_k(z) = (aᵢₖᵀ z)⁴ / n³  +  bᵢₖᵀ z

where aᵢₖ is a random unit vector — the quartic extends the LQ squared term
½(aᵀz)² to 4th power in the same direction.  bᵢₖ is a small random linear term.

Constraints are always feasible at z0 = 0:
  h^i(z) = Aᵢ z = 0  (bE = 0)
  g^i(z) = 1 - Cᵢ z ≥ 0  (slack = 1 at z0)
"""
function generate_random_nl_goop(N, K, ni, mEi, mIi, rng)
    n_total = sum(ni)
    z0 = zeros(n_total)   # origin is always strictly feasible

    # aᵢₖ: random unit vector (quartic direction); bᵢₖ: small random linear term
    a = [[let v = randn(rng, n_total); v ./ norm(v) end for _ in 1:K] for _ in 1:N]
    b = [[randn(rng, n_total) .* 0.05                for _ in 1:K] for _ in 1:N]

    # Equality: Aᵢ z = 0, satisfied exactly at z0 = 0
    A  = [randn(rng, mEi[i], n_total) for i in 1:N]
    bE = [zeros(mEi[i])               for i in 1:N]

    # Inequality: 1 - Cᵢ z ≥ 0, slack = 1 at z0 = 0
    C = [randn(rng, mIi[i], n_total) for i in 1:N]
    d = [ones(mIi[i])                for i in 1:N]

    return NLGOOPData(a, b, A, bE, C, d, z0)
end

"""
    pack_nl_params(data::NLGOOPData, N, K) → Vector{Float64}

Flatten NL instance data in the order expected by a compiled NL `ParametricGOOP`.
"""
function pack_nl_params(data::NLGOOPData, N, K)
    p = Float64[]
    for i in 1:N
        for k in 1:K
            append!(p, data.a[i][k]);  append!(p, data.b[i][k])
        end
        append!(p, vec(data.A[i]));   append!(p, data.bE[i])
        append!(p, vec(data.C[i]));   append!(p, data.d[i])
    end
    return p
end

"""
    generate_random_lq_goop(N, K, ni, mEi, mIi, rng) → LQGOOPData

LQ instance: quadratic costs with Q sampled as Rᵀ R (r real eigenvalues).

  J^i_k(z) = ½ zᵀ Qᵢₖ z  +  qᵢₖᵀ z,    Qᵢₖ = Rᵀ R,  R ∈ ℝ^{r×n}

The quadratic ½(aᵀz)² = ½ zᵀ(aaᵀ)z is the r=1 special case that corresponds
directly to the NL quartic (aᵀz)⁴/n³.  Same constraint structure as NL:
  h^i(z) = Aᵢ z = 0  (bE = 0)
  g^i(z) = 1 - Cᵢ z ≥ 0  (slack = 1 at z0 = 0)
"""
function generate_random_lq_goop(N, K, ni, mEi, mIi, rng)
    n_total = sum(ni)
    z0 = zeros(n_total)

    # Q = Rᵀ R with r rows → r real (non-negative) eigenvalues
    r = max(1, n_total ÷ 2)
    Q = [[let R = randn(rng, r, n_total); R' * R end for _ in 1:K] for _ in 1:N]
    q = [[randn(rng, n_total) .* 0.05                              for _ in 1:K] for _ in 1:N]

    # Same constraint structure as NL; feasible at z0 = 0 by construction
    A = [randn(rng, mEi[i], n_total) for i in 1:N]
    b = [zeros(mEi[i])               for i in 1:N]
    C = [randn(rng, mIi[i], n_total) for i in 1:N]
    d = [ones(mIi[i])                for i in 1:N]

    return LQGOOPData(Q, q, A, b, C, d, z0)
end

"""
    pack_lq_params(data::LQGOOPData, N, K) → Vector{Float64}

Flatten LQ instance data in the order expected by a compiled LQ `ParametricGOOP`.
"""
function pack_lq_params(data::LQGOOPData, N, K)
    p = Float64[]
    for i in 1:N
        for k in 1:K
            append!(p, vec(data.Q[i][k]));  append!(p, data.q[i][k])
        end
        append!(p, vec(data.A[i]));  append!(p, data.b[i])
        append!(p, vec(data.C[i]));  append!(p, data.d[i])
    end
    return p
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PDIP Newton solver
# ═══════════════════════════════════════════════════════════════════════════════

"""
    make_initial_point(pg::ParametricGOOP, z0) → Vector{Float64}

PDIP initial point: primal z = z0 (strictly feasible), all duals = 0.1.
"""
function make_initial_point(pg::ParametricGOOP, z0::Vector{Float64})
    n_total = sum(pg.ni)
    y0      = zeros(pg.n)
    y0[1:n_total] .= z0
    off = n_total
    for i in 1:pg.N
        y0[off+1 : off+pg.dual_dims[i]] .= 0.1
        off += pg.dual_dims[i]
    end
    return y0
end

"""
    run_pdip(pg, params, y0; kwargs...) → (y_sol, s_sol, res_history, solve_time_s)

PDIP Newton solver.  `params = pack_*_params(data, N, K)` supplies the instance.
`solve_time_s` covers only Newton iterations; compilation is excluded.
"""
function run_pdip(pg::ParametricGOOP, params::Vector{Float64}, y0::Vector{Float64};
                  rho_init   = 0.1,
                  rho_factor = 0.1,
                  rho_min    = 1e-10,
                  max_outer  = 30,
                  max_inner  = 100,
                  tol_inner  = 1e-6,
                  tol_outer  = 1e-8,
                  c1         = 1e-4,
                  beta_ls    = 0.5,
                  verbose    = false)

    n = pg.n;  n_comp = pg.n_comp;  n_nc = pg.n_nc

    F_nc = y -> pg.F_nc_num(y, params)
    a_fn = y -> pg.a_num(y, params)
    b_fn = y -> pg.b_num(y, params)
    Jnc  = y -> pg.J_nc_num(y, params)
    Ja   = y -> pg.J_a_num(y, params)
    Jb   = y -> pg.J_b_num(y, params)

    K_rho(w, ρ) = let y = w[1:n], s = w[n+1:end]
        [F_nc(y); a_fn(y) .- s; s .* b_fn(y) .- ρ]
    end

    J_aug(w) = let y = w[1:n], s = w[n+1:end], bv = b_fn(y)
        [Jnc(y)            zeros(n_nc, n_comp);
         Ja(y)             -I(n_comp);
         diagm(s) * Jb(y)  diagm(bv)]
    end

    s0 = max.(a_fn(y0), fill(rho_init, n_comp))
    @assert all(s0 .> 0)       "Initial slacks must be > 0"
    @assert all(b_fn(y0) .> 0) "Initial b(y0) must be > 0"

    w   = [y0; s0]
    rho = rho_init
    K0  = w -> K_rho(w, 0.0)

    res_history = Float64[]
    t_start     = time()

    for _ in 1:max_outer
        for _ in 1:max_inner
            Kw  = K_rho(w, rho)
            res = norm(Kw)
            push!(res_history, res)
            res < tol_inner && break

            J  = J_aug(w)
            Δw = try
                J \ (-Kw)
            catch e
                e isa LinearAlgebra.SingularException || rethrow(e)
                (J + 1e-8 * I(size(J, 1))) \ (-Kw)
            end
            yc = w[1:n];  sc = w[n+1:end]
            α  = 1.0
            for _ in 1:60
                yn = yc .+ α .* Δw[1:n];  sn = sc .+ α .* Δw[n+1:end]
                if all(sn .> 0) && all(b_fn(yn) .> 0) &&
                   norm(K_rho([yn; sn], rho)) ≤ (1.0 - c1*α) * res
                    break
                end
                α *= beta_ls;  α < 1e-16 && (α = 0.0; break)
            end
            α > 0 && (w .+= α .* Δw)
        end

        norm(K0(w)) < tol_outer && break
        rho = max(rho * rho_factor, rho_min)
    end

    solve_time = time() - t_start
    verbose && @printf("  [%s] ||K0||=%.2e  t=%.4fs\n", pg.name, norm(K0(w)), solve_time)

    return w[1:n], w[n+1:end], res_history, solve_time
end

run_pdip_fixed_rho(pg, params, y0, rho; kw...) =
    run_pdip(pg, params, y0;
             rho_init=rho, max_outer=1, rho_factor=1.0,
             tol_inner=1e-14, tol_outer=1e-14, kw...)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Experiment configuration — 4 one-time compilations
# ═══════════════════════════════════════════════════════════════════════════════

const EXP_N   = 2
const EXP_K   = 2
const EXP_ni  = [2, 2]
const EXP_mEi = [1, 1]
const EXP_mIi = [1, 1]
const N_PROBS = 10

println("\n" * "="^65)
println("  One-time compilation (4 systems: NL×{Complete,Reduced}, LQ×{Complete,Reduced})")
println("="^65)

# PGOOP_NL_C, T_NL_C = compile_parametric_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi;
#     builder=build_complete_goop_kkt, make_sym_fns=make_nl_sym_fns)
# PGOOP_NL_R, T_NL_R = compile_parametric_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi;
#     builder=build_reduced_goop_kkt,  make_sym_fns=make_nl_sym_fns)
PGOOP_LQ_C, T_LQ_C = compile_parametric_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi;
    builder=build_complete_goop_kkt, make_sym_fns=make_lq_sym_fns)
PGOOP_LQ_R, T_LQ_R = compile_parametric_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi;
    builder=build_reduced_goop_kkt,  make_sym_fns=make_lq_sym_fns)

println("\n  System sizes (K = $EXP_K, same for NL and LQ with identical dimensions):")
@printf("    Complete KKT: |y| = %d   augmented = %d\n",
        PGOOP_NL_C.n, PGOOP_NL_C.n + PGOOP_NL_C.n_comp)
@printf("    Reduced  KKT: |y| = %d   augmented = %d\n",
        PGOOP_NL_R.n, PGOOP_NL_R.n + PGOOP_NL_R.n_comp)

# Random instances — generated once, reused across solvers
INSTANCES_NL = [generate_random_nl_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi,
                                         MersenneTwister(s)) for s in 1:N_PROBS]
INSTANCES_LQ = [generate_random_lq_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi,
                                         MersenneTwister(s)) for s in 1:N_PROBS]
































# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Experiment 1 — Solve-time comparison table
#     Compilation is one-time overhead (reported once, not per instance).
# ═══════════════════════════════════════════════════════════════════════════════

# function print_timing_table(label, pg_C, pg_R, instances, pack_fn; rho=nothing)
#     println("\n  ── $label ──")
#     @printf("  %-8s  %-16s  %-14s\n", "Problem", "Solve C (s)", "Solve R (s)")
#     println("  " * "-"^42)

#     times_C = Float64[];  times_R = Float64[]
#     for (idx, data) in enumerate(instances)
#         params = pack_fn(data, pg_C.N, pg_C.K)   # same params for both solvers
#         if isnothing(rho)
#             _, _, _, t_c = run_pdip(pg_C, params, make_initial_point(pg_C, data.z0))
#             _, _, _, t_r = run_pdip(pg_R, params, make_initial_point(pg_R, data.z0))
#         else
#             _, _, _, t_c = run_pdip_fixed_rho(pg_C, params, make_initial_point(pg_C, data.z0), rho)
#             _, _, _, t_r = run_pdip_fixed_rho(pg_R, params, make_initial_point(pg_R, data.z0), rho)
#         end
#         push!(times_C, t_c);  push!(times_R, t_r)
#         @printf("  %-8d  %-16.4f  %.4f\n", idx, t_c, t_r)
#     end

#     println("  " * "-"^42)
#     @printf("  %-8s  %-16s  %-14s\n", "Mean±Std",
#             @sprintf("%.4f±%.4f", mean(times_C), std(times_C)),
#             @sprintf("%.4f±%.4f", mean(times_R), std(times_R)))
# end

# println("\n" * "="^65)
# @printf("  EXPERIMENT 1: Solve-time comparison  (N=%d, K=%d,  %d instances)\n",
#         EXP_N, EXP_K, N_PROBS)
# @printf("  One-time compile: NL C=%.2fs R=%.2fs   LQ C=%.2fs R=%.2fs\n",
#         T_NL_C, T_NL_R, T_LQ_C, T_LQ_R)
# println("="^65)

# print_timing_table("Example A: Nonlinear (coupled-quartic costs, linear constraints)",
#                    PGOOP_NL_C, PGOOP_NL_R, INSTANCES_NL, pack_nl_params)
# print_timing_table("Example B: LQ (quadratic costs, linear constraints)",
#                    PGOOP_LQ_C, PGOOP_LQ_R, INSTANCES_LQ, pack_lq_params)














# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Experiment 2 — Convergence at fixed ρ  (nonlinear example)
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^65)
println("  EXPERIMENT 2: KKT residual convergence at fixed ρ  (nonlinear example)")
println("="^65)

const RHO_VALUES = [0.1] #[1.0, 0.1, 0.01, 0.001]
const MAX_INNER  = 50   # stop early; numerical noise dominates beyond ~10 iters
const N_INITS    = 10   # random initializations (fixed-problem experiment)
const N_PROBS_CV = 1   # random problems       (random-problem experiment)

# ── helpers ──────────────────────────────────────────────────────────────────

function sample_random_init(pg::ParametricGOOP, z0::Vector{Float64}, rng)
    n_total = sum(pg.ni)
    y0      = zeros(pg.n)
    y0[1:n_total] .= z0
    off = n_total
    for i in 1:pg.N
        y0[off+1 : off+pg.dual_dims[i]] .= exp.(randn(rng, pg.dual_dims[i]) .* 0.2 .- 1.0)
        off += pg.dual_dims[i]
    end
    return y0
end

nanmean(M; dims) = mapslices(v -> mean(filter(!isnan, v)), M; dims=dims)
nanvar(M;  dims) = mapslices(v -> (f = filter(!isnan, v); length(f) > 1 ? var(f) : 0.0), M; dims=dims)

# Collect raw KKT residuals (not log). Returns max_steps × n_runs matrix.
# Only converging runs (final < initial) are kept.
function collect_residuals(run_iter, max_steps, n_target)
    cols = Vector{Vector{Float64}}()
    attempts = 0
    while length(cols) < n_target && attempts < 10 * n_target
        attempts += 1
        hist = run_iter()
        L    = min(length(hist), max_steps)
        hist[L] < hist[1] && push!(cols, hist[1:L])
    end
    length(cols) < n_target &&
        @warn "Only $(length(cols))/$n_target runs converged"
    mat = fill(NaN, max_steps, length(cols))
    for (j, col) in enumerate(cols)
        mat[1:length(col), j] .= col
    end
    return mat
end

# Each run as an individual log10 curve
function plot_runs!(plt, res_mat, label, color; ls=:solid)
    iters = 1:size(res_mat, 1)
    for j in 1:size(res_mat, 2)
        col   = res_mat[:, j]
        valid = .!isnan.(col)
        any(valid) || continue
        plot!(plt, iters[valid], log10.(max.(col[valid], 1e-30));
              label=(j == 1 ? label : ""), color=color, lw=1.2, alpha=0.7, ls=ls)
    end
end

function convergence_plot(title_str)
    plot(; xlabel="Newton iteration",
           ylabel=L"\log_{10}(\|K_\rho(w)\|)",
           title=title_str,
           legend=:topright, grid=true, framestyle=:box, size=(700, 450))
end

# # ── fixed problem data ────────────────────────────────────────────────────────
# CONV_DATA   = generate_random_nl_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi,
#                                       MersenneTwister(42))
# CONV_PARAMS = pack_nl_params(CONV_DATA, EXP_N, EXP_K)

# # ── Plot set A: fixed problem, random initializations ────────────────────────
# println("\n  -- Plot set A: fixed problem, random initializations --")
# for rho in RHO_VALUES
#     @printf("  ρ = %.4g …\n", rho)

#     function run_init(pg)
#         rng = MersenneTwister(rand(UInt))
#         y0  = sample_random_init(pg, CONV_DATA.z0, rng)
#         _, _, hist, _ = run_pdip_fixed_rho(pg, CONV_PARAMS, y0, rho; max_inner=MAX_INNER)
#         return hist
#     end

#     res_C = collect_residuals(() -> run_init(PGOOP_NL_C), MAX_INNER, N_INITS)
#     res_R = collect_residuals(() -> run_init(PGOOP_NL_R), MAX_INNER, N_INITS)

#     plt = convergence_plot(
#         "Fixed problem, random init  (ρ=$rho, K=$EXP_K, $N_INITS runs)")
#     plot_runs!(plt, res_C, "Complete", :steelblue)
#     plot_runs!(plt, res_R, "Reduced",  :crimson; ls=:dash)
#     savefig(plt, joinpath(@__DIR__,
#         "conv_fixedproblem_rho$(replace(string(rho),"."=>"p"))_K$(EXP_K).pdf"))
# end

# # ── Plot set B: random problems, canonical initialization ─────────────────────
# println("\n  -- Plot set B: random problems, canonical initialization --")
# PROBS_CV = [generate_random_nl_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi,
#                                     MersenneTwister(s)) for s in 1:N_PROBS_CV]

# for rho in RHO_VALUES
#     @printf("  ρ = %.4g …\n", rho)

#     function run_prob(pg, data)
#         params = pack_nl_params(data, pg.N, pg.K)
#         y0     = make_initial_point(pg, data.z0)
#         _, _, hist, _ = run_pdip_fixed_rho(pg, params, y0, rho; max_inner=MAX_INNER)
#         return hist
#     end

#     prob_iter_C = let probs = shuffle(MersenneTwister(1), PROBS_CV), idx = Ref(0)
#         () -> (idx[] = mod1(idx[]+1, length(probs)); run_prob(PGOOP_NL_C, probs[idx[]]))
#     end
#     prob_iter_R = let probs = shuffle(MersenneTwister(1), PROBS_CV), idx = Ref(0)
#         () -> (idx[] = mod1(idx[]+1, length(probs)); run_prob(PGOOP_NL_R, probs[idx[]]))
#     end

#     res_C = collect_residuals(prob_iter_C, MAX_INNER, N_PROBS_CV)
#     res_R = collect_residuals(prob_iter_R, MAX_INNER, N_PROBS_CV)

#     plt = convergence_plot(
#         "Random problems, canonical init  (ρ=$rho, K=$EXP_K, $N_PROBS_CV problems)")
#     plot_runs!(plt, res_C, "Complete", :steelblue)
#     plot_runs!(plt, res_R, "Reduced",  :crimson; ls=:dash)
#     savefig(plt, joinpath(@__DIR__,
#         "conv_randprob_rho$(replace(string(rho),"."=>"p"))_K$(EXP_K).pdf"))
# end


# ── Plot set C: random QP problems, canonical initialization ─────────────────────
println("\n  -- Plot set C: random QP problems, canonical initialization --")
PROBS_CV = [generate_random_lq_goop(EXP_N, EXP_K, EXP_ni, EXP_mEi, EXP_mIi,
                                    MersenneTwister(s)) for s in 1:N_PROBS_CV]

for rho in RHO_VALUES
    @printf("  ρ = %.4g …\n", rho)

    function run_prob(pg, data)
        params = pack_lq_params(data, pg.N, pg.K)
        y0     = make_initial_point(pg, data.z0)
        primal_sol, _, hist, _ = run_pdip_fixed_rho(pg, params, y0, rho; max_inner=MAX_INNER)
        return hist
    end
    
    prob_iter_C = let probs = shuffle(MersenneTwister(1), PROBS_CV), idx = Ref(0)
        () -> (idx[] = mod1(idx[]+1, length(probs)); run_prob(PGOOP_LQ_C, probs[idx[]]))
    end
    prob_iter_R = let probs = shuffle(MersenneTwister(1), PROBS_CV), idx = Ref(0)
        () -> (idx[] = mod1(idx[]+1, length(probs)); run_prob(PGOOP_LQ_R, probs[idx[]]))
    end

    Main.@infiltrate

    res_C = collect_residuals(prob_iter_C, MAX_INNER, N_PROBS_CV)
    res_R = collect_residuals(prob_iter_R, MAX_INNER, N_PROBS_CV)

    plt = convergence_plot(
        "Random QP problems, canonical init  (ρ=$rho, K=$EXP_K, $N_PROBS_CV problems)")
    plot_runs!(plt, res_C, "Complete", :steelblue)
    plot_runs!(plt, res_R, "Reduced",  :crimson; ls=:dash)
    savefig(plt, joinpath(@__DIR__,
        "conv_randprob_rho$(replace(string(rho),"."=>"p"))_K$(EXP_K).pdf"))
    
    Main.@infiltrate
end

println("\nAll experiments complete.")