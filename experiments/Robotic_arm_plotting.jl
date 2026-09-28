# Robotic_arm_plotting.jl -- the robotic-arm figures: one PDF and one HTML per solve.
#
# Ported from ScholtesReducedGOOP.jl (Jingqi Li), `examples/robotic_arm_plot.jl`, with
# `stop_reason` and `write_robotic_arm_plots` from `examples/robotic_arm.jl`. The drawing
# code below the adapter is the source's VERBATIM, on the same Plots / GR versions, so a
# run draws the same figures as the source's `demo()`: `robotic_arm_guess_<name>.{pdf,html}`
# for the two initial guesses, `robotic_arm_rho<ρ>.{pdf,html}` for every cold row and
# `robotic_arm_rho<ρ>_warm-z-eq.{pdf,html}` for every warm row. The only changes: this is a
# module over `experiments/robotic_arm_core.jl` (the adapter below binds a
# `ScenarioConfig` to the source's global names), `mkpath` makes the file's own directory
# instead of the source's fixed `examples/figures/robot_arm`, and `new_traces` comes
# from the core, where the sweep collects the traces.
#
# Loaded by Robotic_arm.jl only when figures are drawn, after the solves.
module RoboticArmPlotting

using Plots
using Printf
using LaTeXStrings
using Logging
using LinearAlgebra: norm

isdefined(Main, :RoboticArmCore) ||
    Base.include(Main, joinpath(@__DIR__, "robotic_arm_core.jl"))
const Core_ = Main.RoboticArmCore

"Default output directory; `write_robotic_arm_plots` is normally given the run's own."
const OUT = normpath(joinpath(@__DIR__, "..", "data", "robotic_arm_scholtes", "figures"))

# --- adapter: the source's scenario globals, bound to one `ScenarioConfig` -------------
const SCENARIO = Ref{Any}(nothing)
T = 0
X_INIT = Vector{Float64}[]
D_MIN = D_HANDLE = V_ARM = V_CHILD = BALANCE = REACH_MAX = CHILD_REACH_MAX = 0.0
POT_GOAL = Float64[]
BASES = Vector{Float64}[]

"Bind `sc` to the names the drawing code reads (the source's `configure_scenario!`)."
function use_scenario!(sc)
    SCENARIO[] = sc
    global T = sc.horizon
    global X_INIT = sc.x_init
    global D_MIN = sc.d_min
    global D_HANDLE = sc.d_handle
    global POT_GOAL = sc.pot_goal
    global V_ARM = sc.v_arm
    global V_CHILD = sc.v_child
    global BASES = sc.bases
    global REACH_MAX = sc.reach_max
    global CHILD_REACH_MAX = sc.child_reach_max
    global BALANCE = sc.balance
    return sc
end

gripper(z, t, a) = Core_.gripper(SCENARIO[], z, t, a)
child(z, t) = Core_.child(SCENARIO[], z, t)
control(z, i, t) = Core_.control(SCENARIO[], z, i, t)
pot_tilt(z, t) = Core_.pot_tilt(SCENARIO[], z, t)
effort_at(i, t, z) = sum(abs2, control(z, i, t))

# --- palette ----------------------------------------------------------------
const GRIP     = ["#1f5fa9", "#d2691e"]   # gripper 1 (blue), gripper 2 (orange)
const POT      = "#5b3a9e"                # the carried object
const CHILD    = "#1f7a4d"                # the pursuer
const NEUTRAL  = "#9a9a9a"                # handle rungs, connectors: support only
const CRITICAL = "#c0392b"                # reserved status: a constraint bound
const INK      = "#1a1a1a"

# One hue per convergence panel, matched to `trajectory_game_plot.jl:95-99` so
# the two figures can be read side by side: a reader who knows the lane-change
# figure already knows which curve is which here.
const COMP_C   = "#a3467c"                # (e) complementarity slackness
const ETA_C    = "#b77924"                # (j) Tikhonov parameter
const ALPHA_C  = "#1f7a8c"                # (i) accepted Newton step size

# IEEE-ish: serif face, tick labels legible at column width, no chartjunk.
const FONT = "Computer Modern"

const BASE = (framestyle = :box, background_color = :white,
              background_color_inside = :white, foreground_color_axis = INK,
              foreground_color_border = INK, foreground_color_text = INK,
              foreground_color_guide = INK, fontfamily = FONT,
              grid = false, titlefontsize = 10, guidefontsize = 10,
              legendfontsize = 8, tickfontsize = 9, titlelocation = :left,
              legend_background_color = :white, legend_foreground_color = NEUTRAL)

"A short filename-safe tag for one rho: 1.0e-3 -> \"rho1e-03\"."
rho_tag(rho) = "rho" * (@sprintf "%.0e" rho)

"""
    trajectories(z) -> (g1, g2, pot, ch)

The four paths as vectors of 3D points, with `x_0` prepended -- it is DATA, not
a variable, so the drawn path starts where the robot actually starts rather
than one step in. Index `k` therefore corresponds to time step `k - 1`.
"""
function trajectories(z)
    steps = 1:T
    g1 = [[X_INIT[1][1:3]]; [gripper(z, t, 1) for t in steps]]
    g2 = [[X_INIT[1][4:6]]; [gripper(z, t, 2) for t in steps]]
    ch = [[X_INIT[2]]; [child(z, t) for t in steps]]
    return g1, g2, 0.5 .* (g1 .+ g2), ch
end

coord(v, k) = [p[k] for p in v]

"Indices for the representative pot configurations: t = 0, T/4, T/2, 3T/4, T."
config_indices() = unique(round.(Int, range(1, T + 1; length = 5)))

"Indices for the synchronized time markers -- a handful, not one per step."
marker_indices(; count = 6) = unique(round.(Int, range(1, T + 1; length = count)))

# ---------------------------------------------------------------------------
# (a) the trajectories, orthographically projected
# ---------------------------------------------------------------------------

# Camera. Azimuth is chosen so NEITHER horizontal axis is near edge-on: at 48
# degrees the x and y directions land about 96 degrees apart on the page. The
# elevation is high enough to open the floor plane out and read the child's
# path across it, low enough that the lift still reads as height.
const AZIM, ELEV = 48.0, 27.0

"""
    project(p) -> (u, v)

Orthographic projection of a 3D point onto the screen plane, in metres. Screen
right and up are the camera basis vectors for (`AZIM`, `ELEV`); there is no
perspective divide, so distances scale identically everywhere and a 2D panel
with `aspect_ratio = :equal` renders the scene to true metric proportion.
"""
function project(p)
    t, f = deg2rad(AZIM), deg2rad(ELEV)
    u = -sin(t) * p[1] + cos(t) * p[2]
    v = -cos(t) * sin(f) * p[1] - sin(t) * sin(f) * p[2] + cos(f) * p[3]
    return u, v
end

"Project a list of 3D points into (us, vs) for plotting."
project_path(ps) = ([project(p)[1] for p in ps], [project(p)[2] for p in ps])

"""
    keepout_outline(centre, radius, zlo, zhi; n = 180) -> (rim_lo, rim_hi, gens)

The keep-out as it actually is: a VERTICAL CYLINDER, `||p[1:2] - c[1:2]|| =
radius`, unbounded in z and therefore drawn across the whole frame.

IT WAS A SPHERE, and that was right until `safety` became horizontal
(robotic_arm.jl, section 4). A sphere drew a keep-out the pot could leave by
climbing, which is exactly the escape the horizontal form removes -- so the
picture showed the reader a way out that the solver does not have, and the
trajectories in these figures duck straight through where the old shell's top
and bottom used to be.

Returned in three pieces, because a cylinder's silhouette is not one curve the
way a sphere's is: the two rim ellipses, and the two vertical generators that
join them. `project` is orthographic and linear, with vertical mapping to
screen-vertical `(0, cos(ELEV))`, so those generators stand at the rim's
extreme screen-`u` points -- the leftmost and rightmost of the projected
ellipse. No silhouette search is needed beyond an argmin/argmax.
"""
function keepout_outline(centre, radius, zlo, zhi; n = 180)
    a   = range(0, 2pi; length = n)
    rim(zz) = project_path([[centre[1] + radius*cos(t),
                             centre[2] + radius*sin(t), zz] for t in a])
    lo, hi = rim(zlo), rim(zhi)
    i, j = argmax(lo[1]), argmin(lo[1])
    gens = (([lo[1][i], hi[1][i]], [lo[2][i], hi[2][i]]),
            ([lo[1][j], hi[1][j]], [lo[2][j], hi[2][j]]))
    return lo, hi, gens
end

"""
    shell_outline(centre, radius; n = 180) -> (us, vs)

The safety sphere as ONE closed curve in screen space: its silhouette.

Under an orthographic projection a sphere's silhouette is exactly a circle of
the true radius about the projected centre, so this is not a simplification of
`safety_shell` -- it is the same surface, drawn as the only part of it that
carries information from a fixed viewpoint. Everything a wireframe adds is
interior structure the reader must mentally discard to follow a trajectory
through it.

Used by panel (a), which has one fixed camera. The HTML companion rotates, so a
silhouette computed here would be wrong the moment the reader drags it; that
one keeps `safety_shell`.
"""
function shell_outline(centre, radius; n = 180)
    cu, cv = project(centre)
    a = range(0, 2pi; length = n)
    return cu .+ radius .* cos.(a), cv .+ radius .* sin.(a)
end

"""
    shell_equator(centre, radius; n = 120) -> (us, vs)

The sphere's horizontal great circle, which the projection turns into an
ellipse. One depth cue, so the outline reads as a sphere standing in the scene
rather than a flat disc pasted onto it. Drawn for the closest approach only.
"""
function shell_equator(centre, radius; n = 120)
    a = range(0, 2pi; length = n)
    return project_path([[centre[1] + radius*cos(t), centre[2] + radius*sin(t),
                          centre[3]] for t in a])
end

"""
    safety_shell(centre, radius) -> Vector of 3D polylines

The set ||p - centre|| = radius, sampled as latitude rings and meridian
circles. Every point satisfies the equation exactly, so the drawn shell IS the
safety constraint `||pot_t - p3_t||^2 - d_min^2 >= 0`.

USED ONLY BY THE ROTATING HTML COMPANION. Panel (a) has one fixed camera and
draws `shell_outline` instead, which says the same thing in one curve. A
wireframe is what a viewpoint-independent scene needs.

A FULL sphere, ANCHORED AT `centre[3]`. This used to be a hemisphere pinned to
z = 0, and both halves of that were right only while the child stood on the
floor: nothing could get underneath it, so the lower half was unreachable and
drawing it would have been clutter. At robosuite scale the child rides at
1.10 m -- ABOVE where the grippers start, at 0.96 m -- so the pot genuinely
passes below it and that half of the constraint is live. Left as it was, the
shells rendered as red domes lying on z = 0 with nothing near them, detached
from the scene they constrain.
"""
function safety_shell(centre, radius; n = 72, ring_fracs = (-0.6, 0.0, 0.6),
                      azimuths = (0.0, pi/3, 2pi/3))
    segs = Vector{Vector{Float64}}[]
    for frac in ring_fracs                         # z = centre_z + frac*radius
        h = frac * radius
        r = sqrt(max(radius^2 - h^2, 0.0))         # from the equation itself
        a = range(0, 2pi; length = n)
        push!(segs, [[centre[1] + r*cos(t), centre[2] + r*sin(t), centre[3] + h]
                     for t in a])
    end
    for az in azimuths                             # full meridian circles
        t = range(0, 2pi; length = n)
        push!(segs, [[centre[1] + radius*cos(s)*cos(az),
                      centre[2] + radius*cos(s)*sin(az),
                      centre[3] + radius*sin(s)] for s in t])
    end
    return segs
end

"Round tick positions inside [lo, hi], about `n` of them."
function nice_ticks(lo, hi, n = 4)
    raw = (hi - lo) / n
    mag = 10.0^floor(log10(raw))
    step = mag * (raw/mag < 1.5 ? 1 : raw/mag < 3 ? 2 : raw/mag < 7 ? 5 : 10)
    return filter(t -> lo - 1e-9 <= t <= hi + 1e-9,
                  collect(ceil(lo/step)*step : step : floor(hi/step)*step))
end

"""
    tick_labels(ticks) -> Vector{String}

Tick text carrying as many decimals as the tick STEP needs, not as many as the
values happen to have. An 8 m workspace steps in whole metres and reads "2"; a
robosuite cell about 1 m across steps in 0.2 m and reads "0.2". The previous
`round(Int, .)` was correct only for the first case -- at mocap scale it turned
the z axis `[1.0, 1.2, 1.4]` into a column of `1`s, and `[0, 0.5, 1, 1.5, 2]`
into `0, 0, 1, 2, 2`.
"""
function tick_labels(ticks)
    isempty(ticks) && return String[]
    step = length(ticks) > 1 ? abs(ticks[2] - ticks[1]) : max(abs(first(ticks)), 1.0)
    d    = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3
    return map(ticks) do t
        tt = abs(t) < 1e-12 ? zero(t) : t        # never print "-0.0"
        d == 0 ? string(round(Int, tt)) : string(round(tt; digits = d))
    end
end

"A 3D polyline in screen space, at one stroke weight."
seg3!(pl, a, b; kw...) = plot!(pl, [project(a)[1], project(b)[1]],
                                   [project(a)[2], project(b)[2]]; label = "", kw...)

function trajectory3d_panel(z)
    g1, g2, pot, ch = trajectories(z)
    gaps = [norm(pot[k] .- ch[k]) for k in eachindex(pot)]
    kc   = argmin(gaps)                        # the safety-critical instant

    # Only three shells are drawn, so only those three widen the ground frame --
    # padding for every child position would inflate it by metres of emptiness
    # and shrink the scene inside the panel.
    shell_at = [marker_indices(count = 3)[1], marker_indices(count = 3)[3], kc]
    xs = vcat(coord(g1,1), coord(g2,1), coord(ch,1),
              vec([ch[k][1] + d for k in shell_at, d in (-D_MIN, D_MIN)]))
    ys = vcat(coord(g1,2), coord(g2,2), coord(ch,2),
              vec([ch[k][2] + d for k in shell_at, d in (-D_MIN, D_MIN)]))
    # NO SHELL TERM HERE, unlike x and y. The keep-out is a cylinder spanning
    # whatever vertical frame it is given, so it cannot be clipped by the floor
    # the way a sphere's underside could, and padding z by a radius would only
    # buy empty space above and below the scene.
    zs = vcat(coord(g1,3), coord(g2,3), coord(ch,3), [POT_GOAL[3]])

    # THE FRAME IS A FRACTION OF THE SCENE, NOT A FIXED NUMBER OF METRES.
    # `floor(min - 0.5)` / `ceil(max + 0.5)` was written for an 8 x 12 m
    # workspace, where half a metre is trim and rounding to whole metres is
    # free. At robosuite mocap scale the whole cell is about 1.2 m across and
    # that same rule produced a 3.0 x 2.0 x 2.0 m box: with `aspect_ratio =
    # :equal` -- which is the whole point of projecting by hand -- the arms, the
    # pot and the child collapsed into a knot with two thirds of the panel
    # empty. Every pad, tick length and label offset below is now a fraction of
    # `span`, so the panel frames itself the same way at either scale.
    span = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys))
    pad  = 0.08 * span
    x0, x1 = minimum(xs) - pad, maximum(xs) + pad
    y0, y1 = minimum(ys) - pad, maximum(ys) + pad

    # THE REFERENCE PLANE IS THE FLOOR OF THE FRAME, NOT z = 0, and at this
    # scale it is not the child's height either: `child_ground` pins the child's
    # vertical CONTROL to zero, so it holds whatever height it started at
    # (1.10 m on the robosuite table) -- which is ABOVE where the grippers begin
    # (0.96 m). Anchoring at zero put a metre of empty air under everything and
    # pushed the scene into the top of the panel. The safety spheres still show
    # where the child actually is.
    z0 = minimum(zs) - 0.25 * pad
    z1 = max(POT_GOAL[3], maximum(zs)) + pad

    xt, yt, zt = nice_ticks(x0, x1), nice_ticks(y0, y1), nice_ticks(z0, z1, 3)
    xl, yl, zl = tick_labels(xt), tick_labels(yt), tick_labels(zt)

    # Tick marks, tick labels and axis titles step off `span` for the same
    # reason the padding does. At 8 m these are the old 0.25 / 0.85 / 2.0.
    tk, lab, ttl = 0.030span, 0.105span, 0.24span

    ax_back = "#c4c4c4"          # back frame: a depth cue, well below the axes
    pa = plot(; BASE..., framestyle = :none, title = "(a)  trajectory",
                legend = :topright, aspect_ratio = :equal, grid = false,
                xticks = false, yticks = false)

    # -- weakest layer: the floor plane and its grid ---------------------------
    ground = [[x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0]]
    gu, gv = project_path(ground)
    plot!(pa, Shape(gu, gv); fillcolor = "#f4f4f2", linecolor = :transparent,
          label = "")
    for x in xt
        seg3!(pa, [x,y0,z0], [x,y1,z0]; color = "#dedede", linewidth = 0.5)
    end
    for y in yt
        seg3!(pa, [x0,y,z0], [x1,y,z0]; color = "#dedede", linewidth = 0.5)
    end
    # The back frame: BOTH far edges, each running from the top of the z-axis to
    # its far corner and then down to the ground. Kept a clear step lighter than
    # the ground axes: they are a depth cue, not a measured frame, and the
    # hierarchy is trajectories > shells > axes > these. They close the vertical
    # extent so the elevated pot has something to be elevated
    # AGAINST. Only these two walls -- the front and side edges of a bounding
    # cube stay absent.
    seg3!(pa, [x0,y0,z1], [x1,y0,z1]; color = ax_back, linewidth = 1.0)
    seg3!(pa, [x1,y0,z1], [x1,y0,z0]; color = ax_back, linewidth = 1.0)
    seg3!(pa, [x0,y0,z1], [x0,y1,z1]; color = ax_back, linewidth = 1.0)
    seg3!(pa, [x0,y1,z1], [x0,y1,z0]; color = ax_back, linewidth = 1.0)

    # -- secondary layer: the keep-out cylinders ------------------------------
    # See `keepout_outline`. Weighted so the closest approach carries the
    # constraint and the other two stay context -- that weighting is the whole
    # reason three are drawn rather than one, since the pair on either side show
    # the keep-out travelling with the child.
    #
    # Each is drawn floor-to-ceiling of the FRAME, because the constraint is
    # height-independent: there is no top to this shape, and giving it one would
    # re-draw the vertical escape that horizontal `safety` exists to deny.
    for (k, alpha, lw) in zip(shell_at, (0.16, 0.16, 0.85), (0.7, 0.7, 1.5))
        lo, hi, gens = keepout_outline(ch[k], D_MIN, z0, z1)
        for (gu, gv) in gens
            plot!(pa, gu, gv; color = CRITICAL, alpha, linewidth = lw, label = "")
        end
        # The floor rim solid and the ceiling rim dotted: the pair reads as a
        # standing tube, and which end is which stays legible without a legend.
        plot!(pa, lo[1], lo[2]; color = CRITICAL, alpha, linewidth = lw, label = "")
        plot!(pa, hi[1], hi[2]; color = CRITICAL, alpha = 0.6alpha,
              linewidth = 0.8lw, linestyle = :dot, label = "")
    end

    # -- ground-plane axes, drawn on the two front edges ----------------------
    ax = "#6f6f6f"
    seg3!(pa, [x0,y1,z0], [x1,y1,z0]; color = ax, linewidth = 1.5)   # x axis
    seg3!(pa, [x1,y0,z0], [x1,y1,z0]; color = ax, linewidth = 1.5)   # y axis
    seg3!(pa, [x0,y0,z0], [x0,y0,z1]; color = ax, linewidth = 1.5)   # z at back corner
    for (x, s) in zip(xt, xl)
        seg3!(pa, [x,y1,z0], [x,y1+tk,z0]; color = ax, linewidth = 1.0)
        u, v = project([x, y1 + lab, z0])
        annotate!(pa, u, v, text(s, 7, ax, :center, FONT))
    end
    for (y, s) in zip(yt, yl)
        seg3!(pa, [x1,y,z0], [x1+tk,y,z0]; color = ax, linewidth = 1.0)
        u, v = project([x1 + lab, y, z0])
        annotate!(pa, u, v, text(s, 7, ax, :center, FONT))
    end
    # Screen-space offsets here, not world-space: a world offset along -x swings
    # the labels across the axis at this azimuth, a screen offset never does.
    for (zz, s) in zip(zt, zl)
        abs(zz - z0) < 1e-9 && continue        # would sit on the x/y axes
        seg3!(pa, [x0,y0,zz], [x0-1.2tk,y0,zz]; color = ax, linewidth = 1.0)
        u, v = project([x0, y0, zz])
        annotate!(pa, u - 0.9lab, v, text(s, 7, ax, :right, FONT))
    end
    u, v = project([0.5*(x0+x1), y1 + ttl, z0])
    annotate!(pa, u, v, text(L"x\;(\mathrm{m})", 9, ax, :center, FONT))
    u, v = project([x1 + ttl, 0.5*(y0+y1), z0])
    annotate!(pa, u, v, text(L"y\;(\mathrm{m})", 9, ax, :center, FONT))
    u, v = project([x0, y0, z1])
    annotate!(pa, u, v + 0.9lab, text(L"z\;(\mathrm{m})", 9, ax, :center, FONT))

    # -- the pot's rigid geometry at five instants ----------------------------
    # Endpoints solid and full weight, interior ones faded: the reader should
    # see where the carry starts and ends without counting rungs.
    for (j, k) in enumerate(config_indices())
        strong = j == 1 || j == length(config_indices())
        seg3!(pa, g1[k], g2[k]; color = strong ? "#5a5a5a" : NEUTRAL,
              linewidth = strong ? 2.4 : 1.3, alpha = strong ? 1.0 : 0.45)
    end

    # -- strongest layer: the trajectories ------------------------------------
    for (path, col, lw, ls, lab) in ((g1, GRIP[1], 1.7, :solid, "gripper 1"),
                                     (g2, GRIP[2], 1.7, :solid, "gripper 2"),
                                     (pot, POT,    3.0, :solid, "pot"),
                                     (ch, CHILD,   2.0, :dash,  "child"))
        pu, pv = project_path(path)
        plot!(pa, pu, pv; color = col, linewidth = lw, linestyle = ls, label = lab)
    end
    # The clearance at the critical instant, where it happens.
    seg3!(pa, pot[kc], ch[kc]; color = POT, linewidth = 1.0, linestyle = :dot)

    # circle = start, star = goal (convention stated once, in the caption)
    for (p, col, m, ms) in ((pot[1], POT, :circle, 5.5), (ch[1], CHILD, :circle, 5.5),
                            (POT_GOAL, POT, :star5, 9.0))
        u, v = project(p)
        scatter!(pa, [u], [v]; color = col, marker = m, markersize = ms,
                 markerstrokewidth = 0, label = "")
    end
    return pa
end

# ---------------------------------------------------------------------------
# (b) the coupling inequality: robot-child separation
# ---------------------------------------------------------------------------
function clearance_panel(z)
    _, _, pot, ch = trajectories(z)
    tt  = collect(0:T)
    # HORIZONTAL, because that is what `safety` constrains (robotic_arm.jl,
    # section 4). Plotted as a 3D norm this panel disagreed with every printed
    # table by the child's fixed height -- it read a minimum of 0.44 m against a
    # `d_safe` of 0.35 that the solver had pinned EXACTLY, so the one curve whose
    # job is to show a bound being touched showed it comfortably slack instead.
    gap = [norm(pot[k][1:2] .- ch[k][1:2]) for k in eachindex(tt)]
    k   = argmin(gap)
    # Relative padding, for the same reason panel (a) frames itself relatively:
    # a fixed 0.55 m below and 0.6 m above was trim on an 8 m workspace, but at
    # mocap scale the gap runs about 0.25-0.72 m, so those pads left the curve
    # in the middle third of the panel and pushed the shaded infeasible band
    # below zero, where a distance cannot go.
    lo_raw, hi_raw = min(D_MIN, minimum(gap)), max(D_MIN, maximum(gap))
    pad = 0.22 * max(hi_raw - lo_raw, D_MIN)
    ylo = max(lo_raw - pad, 0.0)
    pc  = plot(; BASE..., title = "(b)  robot--child separation (horizontal)",
                 legend = :topright, xlabel = L"t\;(\mathrm{step})",
                 ylabel = L"d(t)\;(\mathrm{m})",
                 ylims = (ylo, hi_raw + pad), xlims = (-0.5, T + 0.5))
    plot!(pc, Shape([-0.5, T + 0.5, T + 0.5, -0.5], [ylo, ylo, D_MIN, D_MIN]);
          fillcolor = CRITICAL, fillalpha = 0.07, linecolor = :transparent,
          label = "")
    hline!(pc, [D_MIN]; color = CRITICAL, linewidth = 1.3, linestyle = :dash,
           label = latexstring("d_{\\mathrm{safe}} = $(D_MIN)\\;\\mathrm{m}"))
    plot!(pc, tt, gap; color = POT, linewidth = 2.2, label = L"d(t)")
    # d_min gets a drop line to the bound and one number. The margin above
    # d_safe is what the reader wants, and the drop line shows it as a length
    # instead of spending a parenthetical on it.
    plot!(pc, [tt[k], tt[k]], [D_MIN, gap[k]]; color = POT, linewidth = 0.8,
          linestyle = :dot, label = "")
    scatter!(pc, [tt[k]], [gap[k]]; color = POT, markersize = 5,
             markerstrokecolor = :white, markerstrokewidth = 1, label = "")
    # Above the marker, never below: when the constraint is ACTIVE the curve sits
    # on the bound and a label underneath lands inside the shaded band.
    annotate!(pc, tt[k], gap[k] + 0.55pad,
              text(latexstring(@sprintf("d_{\\min} = %.2f\\;\\mathrm{m}", gap[k])),
                   8, POT, :center, FONT))
    return pc
end

# ---------------------------------------------------------------------------
# (c) the private inequalities: actuation limits
# ---------------------------------------------------------------------------
# Each curve is EXACTLY the quantity the solver constrains, not a proxy: the
# arms are bounded on the full 3D control norm per gripper, the child only on
# its HORIZONTAL norm -- its vertical control is pinned to zero by an equality,
# so bounding it too would be meaningless. Both grippers are drawn rather than
# their maximum, since there are only two and they carry separate constraints.
#
# Controls exist for t = 1..T, one per step, so this panel's axis starts at 1
# where (b)'s starts at 0. That is not a mismatch: (b) plots a distance between
# STATES, and there is a state at t = 0.
function velocity_panel(z)
    ts = collect(1:T)
    v1 = [norm(control(z, 1, t)[1:3]) for t in ts]
    v2 = [norm(control(z, 1, t)[4:6]) for t in ts]
    vc = [norm(control(z, 2, t)[1:2]) for t in ts]
    # Headroom for the legend. At 1.18 the panel stopped just above the arm
    # bound, so a legend anywhere inside it sat on the curves -- `:bottomright`
    # covered the child's drop off its limit, which is the one event in the
    # panel. 1.5 opens a clear band above both dashed bounds, where nothing is
    # ever plotted: the speeds cannot exceed the limits they are drawn against.
    top = max(V_ARM, V_CHILD) * 1.5

    pv = plot(; BASE..., title = "(c)  velocity constraints", legend = :topright,
                xlabel = L"t\;(\mathrm{step})", ylabel = L"\|v\|\;(\mathrm{m/s})",
                ylims = (0, top), xlims = (0.5, T + 0.5), legend_columns = 2)
    hline!(pv, [V_ARM]; color = GRIP[1], linewidth = 1.0, linestyle = :dash,
           label = latexstring("v^{\\mathrm{arm}}_{\\max} = $(V_ARM)"))
    hline!(pv, [V_CHILD]; color = CHILD, linewidth = 1.0, linestyle = :dash,
           label = latexstring("v^{\\mathrm{child}}_{\\max} = $(V_CHILD)"))
    # The two grippers carry separate constraints but move together here, so
    # gripper 1 is drawn wide and gripper 2 narrow on top: one two-tone curve
    # when they agree, two visibly different ones when they do not.
    plot!(pv, ts, v1; color = GRIP[1], linewidth = 3.2, label = "gripper 1")
    plot!(pv, ts, v2; color = GRIP[2], linewidth = 1.3, label = "gripper 2")
    plot!(pv, ts, vc; color = CHILD, linewidth = 2.0, linestyle = :dash,
          label = "child (horiz.)")
    # The child rides its bound for the whole horizon -- the one active
    # constraint in the game -- so it is called out rather than left to inference.
    if maximum(vc) > V_CHILD - 1e-3
        annotate!(pv, 0.5 * T, V_CHILD - 0.09top,
                  text("saturated", 6, NEUTRAL, :center, FONT))
    end
    return pv
end

# ---------------------------------------------------------------------------
# (d) the nominal reach inequalities: end-effector workspace bounds
# ---------------------------------------------------------------------------
# Plot the exact distances used by `arm_reach` and `child_reach`, rather than
# the signed slack. The bound lines make an active reach face visible, while
# the three curves retain the identity of the two grippers and the child.
function reach_panel(z)
    ts = collect(1:T)
    r1 = [norm(gripper(z, t, 1) .- BASES[1]) for t in ts]
    r2 = [norm(gripper(z, t, 2) .- BASES[2]) for t in ts]
    rc = [norm(child(z, t) .- BASES[3]) for t in ts]
    top = 1.12 * max(REACH_MAX, CHILD_REACH_MAX, maximum(r1), maximum(r2), maximum(rc))
    pr = plot(; BASE..., title = "(d)  workspace reach", legend = :topright,
                xlabel = L"t\;(\mathrm{step})", ylabel = L"\|p - p_{\mathrm{base}}\|\;(\mathrm{m})",
                ylims = (0, top), xlims = (0.5, T + 0.5), legend_columns = 2)
    hline!(pr, [REACH_MAX]; color = GRIP[1], linewidth = 1.0, linestyle = :dash,
           label = latexstring("r^{\\mathrm{arm}}_{\\max} = $(REACH_MAX)"))
    hline!(pr, [CHILD_REACH_MAX]; color = CHILD, linewidth = 1.0, linestyle = :dash,
           label = latexstring("r^{\\mathrm{child}}_{\\max} = $(CHILD_REACH_MAX)"))
    plot!(pr, ts, r1; color = GRIP[1], linewidth = 2.2, label = "gripper 1")
    plot!(pr, ts, r2; color = GRIP[2], linewidth = 1.3, label = "gripper 2")
    plot!(pr, ts, rc; color = CHILD, linewidth = 2.0, linestyle = :dash,
          label = "child")
    return pr
end

# ---------------------------------------------------------------------------
# (e) the level-3 objective: pot balance
# ---------------------------------------------------------------------------
# `load_balance` (section 5) sums `pot_tilt`'s EXCESS over `BALANCE` across the
# whole horizon into one scalar objective; this panel draws `pot_tilt` itself
# at every step, against the same bound, so a reader can see WHEN the pot is
# closest to tipping rather than only the aggregate the solver reduces.
# Axis starts at t = 0 for the same reason as (b): this is a distance between
# STATES, and there is a state at t = 0.
function balance_panel(z)
    ts  = collect(0:T)
    off = [t == 0 ? abs(X_INIT[1][3] - X_INIT[1][6]) : pot_tilt(z, t) for t in ts]
    yhi = max(BALANCE, maximum(off)) * 1.15
    pb  = plot(; BASE..., title = "(e)  pot balance", legend = :topright,
                 xlabel = L"t\;(\mathrm{step})",
                 ylabel = L"|\Delta z|\;(\mathrm{m})",
                 ylims = (0, yhi), xlims = (-0.5, T + 0.5))
    plot!(pb, Shape([-0.5, T + 0.5, T + 0.5, -0.5], [BALANCE, BALANCE, yhi, yhi]);
          fillcolor = CRITICAL, fillalpha = 0.07, linecolor = :transparent,
          label = "")
    hline!(pb, [BALANCE]; color = CRITICAL, linewidth = 1.3, linestyle = :dash,
           label = latexstring("\\mathrm{balance} = $(BALANCE)\\;\\mathrm{m}"))
    plot!(pb, ts, off; color = POT, linewidth = 2.2, label = L"|\Delta z(t)|")
    return pb
end

# ---------------------------------------------------------------------------
# (f) the level-1 objectives: control effort, per agent
# ---------------------------------------------------------------------------
# `effort(i)` (section 5) sums this quantity over the whole horizon into one
# scalar objective per player; this panel draws it per step instead, so a
# reader can see WHERE each agent works hardest rather than only the total
# each level 1 reduces. The robot's two grippers are summed into one 6D norm
# here -- unlike (c), which keeps them separate because they carry SEPARATE
# speed constraints -- since `effort(1)` itself never splits them.
function effort_panel(z)
    ts  = collect(1:T)
    e1  = [effort_at(1, t, z) for t in ts]
    ec  = [effort_at(2, t, z) for t in ts]
    top = max(maximum(e1), maximum(ec)) * 1.15
    pe  = plot(; BASE..., title = "(f)  control effort", legend = :topright,
                 xlabel = L"t\;(\mathrm{step})", ylabel = L"\|u\|^2",
                 ylims = (0, top), xlims = (0.5, T + 0.5))
    plot!(pe, ts, e1; color = GRIP[1], linewidth = 2.2, label = "robot")
    plot!(pe, ts, ec; color = CHILD, linewidth = 2.0, linestyle = :dash,
          label = "child")
    return pe
end

# --- convergence panels ------------------------------------------------------
#
# Zeros and non-finite values are legal (an exact hit; a diverged step) but have
# no place on a log axis. They drop to the smallest positive value present
# rather than being discarded, which would shift every iteration index after
# them and quietly misplace the curve.
function _logsafe(v)
    pos = filter(x -> x > 0 && isfinite(x), v)
    isempty(pos) && return Float64[]
    lo = minimum(pos)
    return [x > 0 && isfinite(x) ? float(x) : lo for x in v]
end

# One convergence series against its iteration index. The two series come from
# DIFFERENT places -- (d) from the result's `history`, (e) from the solver trace
# -- so they need not be the same length; each is drawn against its own `1:n`
# and they are never zipped. They are two panels rather than one twin-axis plot
# because they are different quantities in different units.
#
# `mark = (level, label)` draws a dashed reference line, which is how panel (d)
# shows the tolerance the run was tested against. `xmax` pins the iteration axis
# to a value the caller shares between the panels, so a series that ends early
# reads as ending early rather than as a shorter run.
# `preserve_gaps` sends a non-positive value to NaN instead of lifting it, so
# the LINE BREAKS there. Two series need it and neither is an error: eta is NaN
# under `:pinv`, where there is no Tikhonov step to report, and alpha is exactly
# 0 when no trial step passed Armijo. `_logsafe` would raise both to the
# smallest positive value present, painting an accepted step where the solver
# gave up -- the one reading the panel exists to rule out.
function _convergence_panel(v, tag, title, ylab, colour; mark = nothing,
                            xmax = nothing, preserve_gaps = false)
    y = preserve_gaps ? [x > 0 && isfinite(x) ? float(x) : NaN for x in v] :
                        _logsafe(v)
    positive = filter(x -> x > 0 && isfinite(x), y)
    k = xmax === nothing ? length(y) : max(xmax, length(y))
    p = plot(; BASE..., title = "($tag)  $title", legend = false,
               xlabel = L"k\;(\mathrm{Newton\ iteration})", ylabel = ylab,
               yscale = :log10,
               xlims = k <= 1 ? :auto : (1 - 0.03k, k + 0.03k))
    isempty(positive) && return p   # nothing to draw; an empty frame is honest
    plot!(p, 1:length(y), y; color = colour, linewidth = 2.0,
          marker = length(y) <= 60 ? :circle : :none, markersize = 3.0,
          markerstrokewidth = 0,
          _decade_axis(positive; include = mark === nothing ? nothing : first(mark),
                       pad = preserve_gaps)...)
    if mark !== nothing
        hline!(p, [first(mark)]; color = CRITICAL, linestyle = :dash, linewidth = 1.0)
        # Left edge: these curves descend, so the space above the line is free
        # there whatever the run did, while the right edge is where a converged
        # curve ends up sitting right on the label.
        annotate!(p, 1, first(mark),
                  text(last(mark), 9, CRITICAL, :left, :bottom, FONT))
    end
    return p
end

# One tick per decade, labelled in LaTeX. NOT cosmetic: left to itself GR sets a
# log tick as "10" plus a UNICODE superscript, and the Computer Modern face has
# no U+2070 -- so every 10^0 tick makes the screen workstation log "glyph
# missing from current font: 8304" (silent in a headless script, noisy in the
# REPL). Going through `latexstring` puts the exponent in GR's math renderer,
# which sets it in CM properly and says nothing.
# `ylims` comes along because the ticks bracket the data OUTWARDS: without it
# the top tick lands past the auto-fitted limit and GR draws its label above the
# frame, on top of the panel title.
# `include` widens the range to cover a reference line: a run that stalls well
# above its tolerance would otherwise fit the axis to the data alone and clip
# the very line the reader is meant to compare against.
# `pad` lifts the frame off the data by a fraction of a decade, for the
# `preserve_gaps` panels: alpha sits ON 1.0 for most of a healthy run and eta on
# its floor, and a curve drawn exactly along `ylims` is half-clipped by GR.
function _decade_axis(y; include = nothing, pad = false)
    lo, hi = extrema(y)
    if include !== nothing && include > 0 && isfinite(include)
        lo, hi = min(lo, include), max(hi, include)
    end
    e0 = floor(Int, log10(lo))
    e1 = ceil(Int, log10(hi))
    pad && lo == hi && (e0 -= 1; e1 += 1)  # a constant series needs a frame
    e1 == e0 && (e1 = e0 + 1)              # a flat series still needs two ticks
    es = e0:max(1, cld(e1 - e0 + 1, 8)):e1  # at most ~8 labels, however wide
    padding = pad ? 0.08 * (e1 - e0) : 0.0
    return (yticks = (10.0 .^ es, [latexstring("10^{$e}") for e in es]),
            ylims  = (10.0^(e0 - padding), 10.0^(e1 + padding)))
end

# The rho-PERTURBED residual the solve actually minimises, NOT `||K_0||`: at a
# fixed rho the complementarity products converge to rho, so the true KKT
# residual has a floor of O(rho) however far this curve falls.
#
# The dashed line is `tol_inner`, the level THIS curve is tested against
# (src/solve.jl), and `stop` names the break that actually fired -- without
# both, a reader cannot tell a solved level from a stalled one, since the two
# look alike on a log axis. The curve must therefore be the solver's
# `history`, whose last entry IS the number the test saw; see the note in
# `robotic_arm.jl` on why the trace cannot supply it.
#
# PLOTTED PER EQUATION, `||R||_2/sqrt(m)`, as `trajectory_game_plot.jl` does.
# A 2-norm sums over all `m = n_nc + 2nc` residual rows, so its height rides on
# the problem size and two horizons cannot be compared; dividing by `sqrt(m)`
# removes that. It also makes the reference line read TRUE: `solve_rho_sweep`
# hands the solver `tol_inner = rho*sqrt(m)`, so on this axis the bar is exactly
# `rho`, which is what the label has always claimed.
residual_panel(resid, tol_inner, stop; xmax = nothing, tol_scale = 1.0) =
    _convergence_panel(resid ./ tol_scale, "g",
                       stop === nothing ? "homotopy residual" :
                                          "homotopy residual  [$stop]",
                       tol_scale == 1 ? L"||R(w;\rho)||_2" :
                                        L"||R(w;\rho)||_2/\sqrt{m}", POT; xmax,
                       mark = tol_inner === nothing ? nothing :
                              (tol_inner, L"\mathrm{tol}_{\mathrm{inner}} = \rho"))

# THE INFINITY NORM -- the largest single coordinate move, not a sum over the
# whole vector. A 2-norm over primals, duals, slacks and `u` together makes the
# same per-coordinate step read bigger on a bigger problem; the max coordinate
# needs no normalisation, which is why (g) divides by `sqrt(m)` and this does
# not. `trajectory_game.jl` reports it the same way, and `solve_rho_sweep`'s
# trace is what takes the norm -- change it there, not here.
#
# One point SHORTER than (g) whenever the level exits on a tolerance: the test
# comes before the step (src/solve.jl), so at the iterate that ENDED the run no
# direction was ever computed. There is no honest value to draw there and none
# is invented; instead the panel shares (g)'s iteration axis and says so in its
# title, which turns the gap from a discrepancy into the thing it means -- the
# solver stopped instead of stepping.
step_panel(dwn; xmax = nothing) =
    _convergence_panel(dwn, "i",
                       xmax !== nothing && length(dwn) == xmax - 1 ?
                           "Newton direction size  [no step at k = $xmax]" :
                           "Newton direction size",
                       L"||\delta w||_\infty", GRIP[1]; xmax)

# (e) THE ONE PART OF `R` THAT DOES NOT CONVERGE TO ZERO: the relaxation pins it
# at rho, so this is where the accuracy `rho` bought is visible as a number.
# Plotted RAW, in the product's own units -- it is not a residual row and there
# is no `sqrt(m)` to divide out.
#
# `s (x) Gam + u` over the RELAXED rows only; `convergence_trace` does the
# filtering. UPPERCASE Gam because this file builds `phi = false`, where the
# inner multipliers are eliminated into the effective multiplier (src/kkt.jl:
# `gam_name = phi ? "gam" : "Gam"`) -- `trajectory_game.jl` runs `phi = true`
# and writes the same panel with a lowercase gamma.
#
# No rho reference line: at a fixed rho this curve SITS at rho, so a bar drawn
# there runs through the data instead of bounding it. `rho` stays in the
# signature so restoring one is a one-line change.
comp_panel(comp, rho; xmax = nothing) =
    _convergence_panel(comp, "h", "complementarity slackness",
                       L"||s \odot \Gamma + u||_\infty", COMP_C; xmax)

# (j) The effective Tikhonov parameter for the direction just taken, AFTER any
# retry escalation. Flat at the floor unless `eta_schedule` is on -- which is a
# fact about the solve, not a broken panel, and worth seeing either way.
eta_panel(eta; xmax = nothing) =
    _convergence_panel(eta, "j", "Tikhonov regularization",
                       L"\eta_k", ETA_C; xmax, preserve_gaps = true)

# (k) The smallest entry of each sign-constrained block, pre-step. These are the
# coordinates the fraction-to-boundary rule is holding off zero, so a run that
# stalls against the boundary shows it here first -- one of them diving while
# the residual flattens is the signature.
#
# Its own axis: `log10` of the value rather than a log scale, because the
# reference these are read against (analysis/boundary_stall.jl) plots them that
# way, and because a zero has to land somewhere rather than vanish -- the
# subnormal floor below is what puts it at about -324 instead of -Inf.
function positive_coordinates_panel(history; xmax = nothing)
    x = [sample.iter for sample in history]
    k = max(something(xmax, 0), isempty(x) ? 0 : maximum(x))
    p = plot(; BASE..., title = "(k)  smallest positive coordinates",
               xlabel = L"k\;(\mathrm{Newton\ iteration})",
               ylabel = L"\log_{10}(\mathrm{value})",
               legend = :bottomleft, legend_columns = 3,
               xlims = k <= 1 ? :auto : (1 - 0.03k, k + 0.03k))
    isempty(history) && return p
    for (key, label, colour) in ((:s, L"s", GRIP[1]),
                                 (:gamma, L"\Gamma", CRITICAL),
                                 (:u, L"u", CHILD))
        values = [sample[key] for sample in history]
        y = [isfinite(v) ? log10(max(v, nextfloat(0.0))) : NaN for v in values]
        plot!(p, x, y; label, color = colour, linewidth = 2.0,
              marker = length(x) <= 60 ? :circle : :none,
              markersize = 3.0, markerstrokewidth = 0)
    end
    return p
end

# (i) A LOG axis, because alpha runs from 1 down past 1e-12 under heavy
# backtracking and a linear one collapses all of that onto the floor.
#
# What a log axis cannot draw is alpha = 0 exactly -- what `solve_goop` reports
# when no trial step passed Armijo. `preserve_gaps` sends those to NaN so the
# LINE BREAKS rather than showing a small accepted step, and each break is
# marked on the axis floor with the count in the title.
function alpha_panel(alpha; xmax = nothing)
    fails = findall(iszero, alpha)
    nf    = length(fails)
    p = _convergence_panel(alpha, "l",
                           nf == 0 ? "Newton step size" :
                           "Newton step size  [" * string(nf) *
                           (nf == 1 ? " line-search stall]" : " line-search stalls]"),
                           L"\alpha_k", ALPHA_C; xmax, preserve_gaps = true)
    positive = filter(a -> a > 0 && isfinite(a), alpha)
    isempty(positive) && return p
    # The full step, for reference: alpha = 1 is Armijo accepting the undamped
    # Newton step, which is what a healthy late iteration looks like. Drawn in
    # NEUTRAL, not CRITICAL -- it is a reference line, not a violated bound.
    hline!(p, [1.0]; color = NEUTRAL, linestyle = :dash, linewidth = 1.0)
    nf == 0 && return p
    # BELOW the data, not inside it: a stall is NO step, not a small one. 2% up
    # the log range, never on `ylims[1]` itself -- GR clips a marker sitting
    # exactly on the boundary, and the panel would then lose a stall its own
    # title is still counting.
    lo, hi  = _decade_axis(positive; pad = true).ylims
    floor_y = lo * (hi / lo)^0.02
    scatter!(p, fails, fill(floor_y, nf); color = CRITICAL,
             markershape = :dtriangle, markersize = 5, markerstrokewidth = 0)
    return p
end

"""
    plot_robotic_arm(z; rho, path, resid, dwn, stop, tol_scale) -> path

The publication figure for one solved primal `z`, written as a vector PDF:
three panels for the scene, plus two more for the solve itself when `resid`
and `dwn` are supplied. `resid` is `GOOPResult.history`, drawn per equation as
`||R||_2/sqrt(m)`; `dwn` comes through `solve_goop`'s `trace` and is an
INFINITY norm, so the two panels need no common scaling. `stop` is a short
phrase naming the break that ended the run, drawn in panel (d)'s title beside
the `tol_inner` line it was tested against; pass `nothing` to leave the panel
unannotated. `tol_scale` is `sqrt(m)` -- pass the same one `solve_rho_sweep`
used, or leave it at 1.0 to plot the raw 2-norm.

CAPTION (for the paper): Two-arm transport of a pot with a rigid handle while a
child pursues it. (a) Trajectories, orthographically projected at true metric
scale: gripper and pot paths, the pot drawn as a rigid bar at five instants
(endpoints solid, interior faded), and the child's collision-avoidance sphere of
radius d_safe as its silhouette -- exactly a circle of the true radius under an
orthographic projection -- emphasized at the closest approach and faint
elsewhere, with the dotted equator marking the critical sphere's horizontal
great circle. Axes lie on the floor of the frame; the child holds a constant
height above it. (b) Robot-child separation against d_safe. (c) Realized gripper
and child speeds (solid) against their limits (dashed). (d) Gripper and child
distances from their nominal reach centers against their workspace bounds.
Circles mark initial positions, stars mark goals. SIX CONVERGENCE PANELS show
what the solver did to arrive at (a), all on log axes and all dimension-agnostic,
so figures at different horizons compare: (g) the homotopy residual per
equation, whose dashed line is the inner tolerance rho and whose bracketed
phrase names the stopping rule that fired; (h) the complementarity slackness the
relaxation pins at rho, the one part of R that does not go to zero; (i) the
Newton direction size as an infinity norm, the largest single coordinate move;
(j) the effective Tikhonov parameter; (k) log10 of the smallest slack, multiplier
and relaxation variable, the coordinates the fraction-to-boundary rule holds off
zero; and (l) the accepted step length, where a failed line search reports alpha
= 0, breaks the line and is marked on the axis floor with the count in the title.
(g) carries one more sample than the other five on a run that exits on a
tolerance: history is pushed at the top of the loop, before the stopping tests,
while both traces fire only once a direction has been computed. The panels share
an iteration axis rather than being zipped, so that gap reads as "the solver
stopped instead of stepping".
"""
function plot_robotic_arm(z; rho, path = joinpath(OUT, "robotic_arm_$(rho_tag(rho)).pdf"),
                          resid = nothing, tr = nothing, stop = nothing,
                          tol_scale = 1.0)
    gr()
    scene  = (trajectory3d_panel(z), clearance_panel(z), velocity_panel(z),
              reach_panel(z), balance_panel(z), effort_panel(z))
    # 10mm on the left, not 4: panel (h) labels its axis `log10(value)` against
    # ticks as wide as "-100", and (d) carries `||R(w;rho)||_2/sqrt(m)`. At 4mm
    # GR ran both guides off the canvas -- the panels drew, the labels did not.
    common = (background_color = :white, plot_title = "",
              left_margin = 10Plots.mm, bottom_margin = 4Plots.mm,
              right_margin = 2Plots.mm, top_margin = 2Plots.mm)
    fig = if resid === nothing && tr === nothing
                plot(scene...; layout = @layout([a{0.58w}; b c; d e; f{0.58w}]),
                         size = (1080, 1400),
             common...)
    else
        t  = something(tr, Core_.new_traces())
        rv = something(resid, Float64[])
        kmax = max(length(rv), length(t.dwn), length(t.eta), length(t.comp),
                   length(t.alpha),
                   isempty(t.positive) ? 0 : maximum(s.iter for s in t.positive))
        plot(scene...,
             residual_panel(rv, rho, stop; xmax = kmax, tol_scale),
             comp_panel(t.comp, rho; xmax = kmax),
             step_panel(t.dwn; xmax = kmax),
             eta_panel(t.eta; xmax = kmax),
             positive_coordinates_panel(t.positive; xmax = kmax),
             alpha_panel(t.alpha; xmax = kmax);
             # The workspace is full width at the top; every other panel is
             # below it in a two-column row. The tall canvas keeps the six
             # convergence panels readable in the PDF rather than compressing
             # them into the workspace row.
             layout = @layout([a; b c; d e; f g; h i; j k; l]),
             size = (1080, 2400), common...)
    end
    mkpath(dirname(path))
    savefig(fig, path)          # vector PDF, the deliverable
    return path
end

"""
    plot_robotic_arm_interactive(z; rho, path) -> path

The workspace as a standalone HTML page: drag to rotate, scroll to zoom,
right-drag to pan. The companion to the PDF, not part of it -- the printed
figure is deliberately a fixed orthographic projection, and this is where a
reader who wants to look around gets to.

It is built from the SAME scene as panel (a) -- same palette, same viewpoint,
same five pot configurations, same three weighted safety shells, same ground grid --
so the two read as one figure in two media rather than as two drawings.

TWO THINGS PLOTLY NEEDS THAT GR DID NOT. Its 3D scene defaults to
`aspectmode: "cube"`, which is exactly the distortion panel (a) exists to avoid,
so the emitted HTML is rewritten to `"data"` -- plotly then scales the axes by
their true ranges and the safety spheres are round in the browser too. And browsers
have no Computer Modern, so the page asks for a Times stack instead; it is the
nearest widely-installed serif and keeps the typography close to the PDF's.

The page links `plotly.js` from a CDN rather than embedding it, which keeps each
file at a few KB but means it wants a network connection the first time.

`PlotlyKaleido` is deliberately not installed: Kaleido exists to export STATIC
images from plotly, and the static figure here comes from GR. The first switch
to the plotly backend would warn about its absence -- harmlessly, since the
fallback plotly writer produces the HTML just fine -- so that switch is made
below with warnings muted.
"""
function plot_robotic_arm_interactive(z; rho,
        path = joinpath(OUT, "robotic_arm_$(rho_tag(rho)).html"),
        title = "pot carry,  rho = $(@sprintf "%.0e" rho)")
    # Drop Warn and below for just this call, keeping genuine Errors visible.
    with_logger(ConsoleLogger(stderr, Logging.Error)) do
        plotly()
    end
    g1, g2, pot, ch = trajectories(z)
    gaps = [norm(pot[k] .- ch[k]) for k in eachindex(pot)]
    kc   = argmin(gaps)
    shell_at = [marker_indices(count = 3)[1], marker_indices(count = 3)[3], kc]

    xs = vcat(coord(g1,1), coord(g2,1), coord(ch,1),
              vec([ch[k][1] + d for k in shell_at, d in (-D_MIN, D_MIN)]))
    ys = vcat(coord(g1,2), coord(g2,2), coord(ch,2),
              vec([ch[k][2] + d for k in shell_at, d in (-D_MIN, D_MIN)]))
    # Framed off the scene span, as panel (a) is -- the ground grid here is the
    # same grid, and rounding it out to whole metres shrank the motion just as
    # badly in the browser.
    pad = 0.08 * max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys))
    x0, x1 = minimum(xs) - pad, maximum(xs) + pad
    y0, y1 = minimum(ys) - pad, maximum(ys) + pad
    zg = minimum(vcat(coord(g1,3), coord(g2,3), coord(ch,3),
                      vec([ch[k][3] - D_MIN for k in shell_at]))) - 0.25pad

    # Plots' plotly azimuth is measured 90 degrees off this file's projector --
    # `camera = (AZIM, ELEV)` opens the scene mirrored in y. `AZIM + 90` puts the
    # eye on the same side panel (a) projects from (verified on the emitted
    # camera: both give an eye ratio y/x = 1.11).
    fig = plot(; legend = :outerright, camera = (AZIM + 90, ELEV), size = (1020, 720),
                 background_color = :white, background_color_inside = :white,
                 fontfamily = "Times New Roman", grid = false,
                 titlefontsize = 13, guidefontsize = 11, legendfontsize = 10,
                 tickfontsize = 9, foreground_color_text = INK,
                 foreground_color_guide = INK,
                 title = title,
                 xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)")

    line3!(a, b; kw...) = plot!(fig, [a[1], b[1]], [a[2], b[2]], [a[3], b[3]];
                                label = "", kw...)

    # weakest: the ground grid at z = 0, so "the child is on the floor" is
    # visible from any angle the reader rotates to.
    for x in nice_ticks(x0, x1)
        line3!([x,y0,zg], [x,y1,zg]; color = "#e2e2e2", linewidth = 1)
    end
    for y in nice_ticks(y0, y1)
        line3!([x0,y,zg], [x1,y,zg]; color = "#e2e2e2", linewidth = 1)
    end

    # secondary: the safety shells, weighted as in the PDF.
    for (j, (k, alpha, lw)) in enumerate(zip(shell_at, (0.13, 0.13, 0.6),
                                                       (1.0, 1.0, 2.0)))
        for (i, seg) in enumerate(safety_shell(ch[k], D_MIN))
            lab = (j == 3 && i == 1) ? "safety, d = $(D_MIN) m" : ""
            plot!(fig, coord(seg,1), coord(seg,2), coord(seg,3);
                  color = CRITICAL, alpha, linewidth = lw, label = lab)
        end
    end

    # the pot's rigid geometry at five instants, endpoints strong.
    for (j, k) in enumerate(config_indices())
        strong = j == 1 || j == length(config_indices())
        line3!(g1[k], g2[k]; color = strong ? "#5a5a5a" : NEUTRAL,
               linewidth = strong ? 5 : 3, alpha = strong ? 1.0 : 0.45)
    end

    # strongest: the trajectories.
    for (path3, col, lw, ls, lab) in ((g1, GRIP[1], 3, :solid, "gripper 1"),
                                      (g2, GRIP[2], 3, :solid, "gripper 2"),
                                      (pot, POT,    5, :solid, "pot"),
                                      (ch, CHILD,   3, :dash,  "child"))
        plot!(fig, coord(path3,1), coord(path3,2), coord(path3,3);
              color = col, linewidth = lw, linestyle = ls, label = lab)
    end
    line3!(pot[kc], ch[kc]; color = POT, linewidth = 2, linestyle = :dot)

    # circle = start, diamond = goal (plotly has no star5 marker).
    for (p, col, m, ms, lab) in ((pot[1], POT, :circle, 7, "pot start"),
                                 (ch[1], CHILD, :circle, 7, "child start"),
                                 (POT_GOAL, POT, :diamond, 9, "pot goal"))
        scatter!(fig, [p[1]], [p[2]], [p[3]]; color = col, marker = m,
                 markersize = ms, markerstrokewidth = 0, label = lab)
    end

    mkpath(dirname(path))
    savefig(fig, path)
    # See the docstring: plotly's default cube aspect is the very distortion
    # panel (a) is projected by hand to avoid, so ask for true data scaling.
    write(path, replace(read(path, String),
                        r"\"aspectmode\"\s*:\s*\"cube\"" => "\"aspectmode\": \"data\""))
    gr()                      # leave the default backend as we found it
    return path
end

# ---------------------------------------------------------------------------
# From the source's examples/robotic_arm.jl: the stopping rule named in panel (g), and
# the loop that writes one PDF and one HTML per guess and per solve.
# ---------------------------------------------------------------------------

# Which break ended a single-level run, in the solver's own order of precedence.
# `scale` affects only how the bar is PRINTED.
function stop_reason(rr, tol_inner, max_inner, tol; scale = 1.0)
    r_name  = scale == 1 ? "||R||_2" : "||R||_2/sqrt(m)"
    k0_name = scale == 1 ? "||K_0||" : "||K_0||/sqrt(m)"
    rr.converged && return @sprintf("%s < %.0e", k0_name, tol / scale)
    isempty(rr.history) && return "no iterations"
    isfinite(rr.history[end]) || return "non-finite residual"
    rr.history[end] < tol_inner &&
        return @sprintf("%s < %.0e", r_name, tol_inner / scale)
    rr.iters >= max_inner && return @sprintf("max_inner = %d", max_inner)
    return "line-search stall"
end

"""
    write_robotic_arm_plots(sc, solutions, warm_solutions, cold_traces; max_inner,
                            tol, tol_scale = 1.0, guesses = (), output_dir = OUT) -> count

The source's `write_robotic_arm_plots`: a PDF and an HTML for every initial guess in
`guesses` (`name => z`), every cold row in `solutions` (`ρ => result`, traces in
`cold_traces[ρ]`) and every warm row in `warm_solutions` (`(tag, ρ, result, trace)`).
Returns the number of files written, as the source does.
"""
function write_robotic_arm_plots(sc, solutions, warm_solutions, cold_traces;
                                 max_inner = Core_.MAX_INNER, tol = Core_.TOL,
                                 tol_scale = 1.0, guesses = (), output_dir = OUT)
    use_scenario!(sc)
    plot_dir = String(output_dir)
    mkpath(plot_dir)

    for (name, z) in guesses
        stem = "robotic_arm_guess_$(name)"
        plot_robotic_arm(z; rho = NaN, path = joinpath(plot_dir, stem * ".pdf"))
        plot_robotic_arm_interactive(z; rho = NaN, path = joinpath(plot_dir, stem * ".html"),
                                     title = "pot carry,  initial guess: $(name)")
    end

    for (rho, rr) in solutions
        stem = "robotic_arm_$(rho_tag(rho))"
        plot_robotic_arm(rr.z; rho, path = joinpath(plot_dir, stem * ".pdf"),
                         resid = rr.history, tr = cold_traces[rho], tol_scale,
                         stop = stop_reason(rr, rho * tol_scale, max_inner,
                                            tol * tol_scale; scale = tol_scale))
        plot_robotic_arm_interactive(rr.z; rho,
                         path = joinpath(plot_dir, stem * ".html"))
    end

    for (tag, rho, rr, tr) in warm_solutions
        stem = "robotic_arm_$(rho_tag(rho))_$(replace(tag, "+" => "-"))"
        plot_robotic_arm(rr.z; rho, path = joinpath(plot_dir, stem * ".pdf"),
                         resid = rr.history, tr, tol_scale,
                         stop = stop_reason(rr, rho * tol_scale, max_inner,
                                            tol * tol_scale; scale = tol_scale))
        plot_robotic_arm_interactive(rr.z; rho,
                         path = joinpath(plot_dir, stem * ".html"))
    end
    return 2 * (length(solutions) + length(warm_solutions) + length(guesses))
end

default(show = false)

end # module RoboticArmPlotting
