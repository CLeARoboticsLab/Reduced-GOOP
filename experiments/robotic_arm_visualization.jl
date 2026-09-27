# Figures for the robotic-arm pot carry (experiments/robotic_arm_core.jl). Included by
# Robotic_arm.jl and Robotic_arm_mpc.jl; needs the `experiments` environment (CairoMakie).

using CairoMakie: CairoMakie
using LinearAlgebra: norm

"""
    save_plan_figure(sc, z, path; title = "")

Four panels of one plan: (a) plan view (xy) of the pot and the child, with the safety
distance `d_min` drawn around the pot at every knot and the pot goal; (b) pot and
child heights; (c) per-arm and child horizontal speeds against their limits; (d) pot tilt
against the balance allowance.
"""
function save_plan_figure(sc, z, path; title = "")
    C = Main.RoboticArmCore
    T = sc.horizon
    ts = 1:T
    pot = reduce(hcat, [C.pot_centre(sc, z, t) for t in ts])
    kid = reduce(hcat, [C.child(sc, z, t) for t in ts])
    fig = CairoMakie.Figure(size = (1100, 800))
    ax = CairoMakie.Axis(fig[1, 1]; title = "(a) plan view", xlabel = "x [m]", ylabel = "y [m]",
                         aspect = CairoMakie.DataAspect())
    θs = range(0, 2π; length = 60)
    for t in ts
        CairoMakie.lines!(ax, pot[1, t] .+ sc.d_min .* cos.(θs), pot[2, t] .+ sc.d_min .* sin.(θs);
                          color = (:orangered, 0.12))
    end
    CairoMakie.lines!(ax, pot[1, :], pot[2, :]; color = :orangered, linewidth = 2, label = "pot")
    CairoMakie.lines!(ax, kid[1, :], kid[2, :]; color = :dodgerblue, linewidth = 2, label = "child")
    CairoMakie.scatter!(ax, [sc.pot_goal[1]], [sc.pot_goal[2]]; marker = :star5, markersize = 16,
                        color = :black, label = "pot goal")
    CairoMakie.axislegend(ax; position = :rb)

    ax = CairoMakie.Axis(fig[1, 2]; title = "(b) height", xlabel = "knot", ylabel = "z [m]")
    CairoMakie.lines!(ax, ts, pot[3, :]; color = :orangered, label = "pot")
    CairoMakie.lines!(ax, ts, kid[3, :]; color = :dodgerblue, label = "child")
    CairoMakie.hlines!(ax, [sc.pot_goal[3]]; color = :black, linestyle = :dash)
    CairoMakie.axislegend(ax; position = :rb)

    ax = CairoMakie.Axis(fig[2, 1]; title = "(c) speeds", xlabel = "knot", ylabel = "‖u‖ [m/s]")
    arm1 = [norm(C.control(sc, z, 1, t)[1:3]) for t in ts]
    arm2 = [norm(C.control(sc, z, 1, t)[4:6]) for t in ts]
    kidv = [norm(C.control(sc, z, 2, t)[1:2]) for t in ts]
    CairoMakie.lines!(ax, ts, arm1; color = :orangered, label = "arm 1")
    CairoMakie.lines!(ax, ts, arm2; color = :darkorange, linestyle = :dash, label = "arm 2")
    CairoMakie.lines!(ax, ts, kidv; color = :dodgerblue, label = "child (xy)")
    CairoMakie.hlines!(ax, [sc.v_arm, sc.v_child]; color = [:orangered, :dodgerblue], linestyle = :dot)
    CairoMakie.axislegend(ax; position = :rt)

    ax = CairoMakie.Axis(fig[2, 2]; title = "(d) pot tilt", xlabel = "knot", ylabel = "|z₁ − z₂| [m]")
    CairoMakie.lines!(ax, ts, [C.pot_tilt(sc, z, t) for t in ts]; color = :purple)
    CairoMakie.hlines!(ax, [sc.balance]; color = :black, linestyle = :dot)

    isempty(title) || CairoMakie.Label(fig[0, :], title; fontsize = 16)
    mkpath(dirname(path))
    CairoMakie.save(path, fig)
    return path
end

"`log10 ‖R(w; ρ)‖` per iteration, one line per solve (`rows`: `(label, result)` pairs)."
function save_convergence_figure(rows, path; title = "Scholtes residual per iteration")
    fig = CairoMakie.Figure(size = (800, 450))
    ax = CairoMakie.Axis(fig[1, 1]; title, xlabel = "iteration", ylabel = "log₁₀ ‖R(w; ρ)‖")
    for (label, r) in rows
        CairoMakie.lines!(ax, 1:length(r.history), log10.(max.(r.history, 1e-300)); label)
    end
    CairoMakie.axislegend(ax; position = :rt, labelsize = 10)
    mkpath(dirname(path))
    CairoMakie.save(path, fig)
    return path
end
