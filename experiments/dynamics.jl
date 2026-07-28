#=
	Common dynamics functions for experiments

	This file contains shared dynamics models used across experiments.
=#

function unflatten_trajectory(z, state_dimension, control_dimension)
    Z = reshape(z, state_dimension + control_dimension, :)
    X = @view Z[1:state_dimension, :]
    U = @view Z[(state_dimension + 1):end, :]
    xs = eachcol(X) .|> collect
    us = eachcol(U) .|> collect
    (; xs, us)
end

"""
	unicycle_dynamics(z, t; Δt, state_dimension=4, control_dimension=2)

Kinematic unicycle dynamics constraint (nonlinear).
State: x = [x, y, v, ψ] (position, speed, heading)
Control: u = [a, ω] (acceleration, yaw rate)

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function unicycle_dynamics(z, t; Δt, state_dimension = 4, control_dimension = 2)
    (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t + 1]

    x_pred = unicycle_step(x_t, u_t; Δt)

    return x_tp1 - x_pred
end

function unicycle_step(x, u; Δt)
    _, _, v, ψ = x
    a, ω = u

    xdot = v * cos(ψ)
    ydot = v * sin(ψ)
    vdot = a
    psidot = ω

    x .+ Δt .* [xdot, ydot, vdot, psidot]
end

"""
	bicycle_dynamics(z, t; Δt, L=1.0, state_dimension=4, control_dimension=2)

Kinematic bicycle dynamics constraint (nonlinear).
State: x = [x, y, v, ψ] (position, speed, heading)
Control: u = [a, δ] (acceleration, steering angle)
L: wheelbase length

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function bicycle_dynamics(z, t; Δt, L = 1.0, state_dimension = 4, control_dimension = 2)
    (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t + 1]

    x_pred = bicycle_step(x_t, u_t; Δt, L)

    return x_tp1 - x_pred
end

function bicycle_step(x, u; Δt, L = 1.0)
    _, _, v, ψ = x
    a, δ = u

    xdot = v * cos(ψ)
    ydot = v * sin(ψ)
    vdot = a
    psidot = (v / L) * tan(δ)

    x .+ Δt .* [xdot, ydot, vdot, psidot]
end

"""
	double_integrator_2d(z, t; Δt, state_dimension=4, control_dimension=2)

2D double integrator dynamics constraint (linear).
State: x = [x, y, vx, vy] (position, velocity)
Control: u = [ax, ay] (acceleration)

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function planar_double_integrator(z, t; Δt, state_dimension = 4, control_dimension = 2)
    (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)

    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t + 1]

    dt2 = 0.5 * Δt * Δt
    # Keep the symbolic residual scalarized; routing it through the dense
    # matrix step changes the generated KKT code enough to affect convergence.
    x_pred = [
        x_t[1] + Δt * x_t[3] + dt2 * u_t[1],
        x_t[2] + Δt * x_t[4] + dt2 * u_t[2],
        x_t[3] + Δt * u_t[1],
        x_t[4] + Δt * u_t[2],
    ]
    return x_tp1 - x_pred
end

function planar_double_integrator_step(x, u; Δt)
    dt2 = 0.5 * Δt * Δt

    A = [
        1.0 0.0 Δt 0.0
        0.0 1.0 0.0 Δt
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ]

    B = [
        dt2 0.0
        0.0 dt2
        Δt 0.0
        0.0 Δt
    ]

    A * x + B * u
end

"""
	single_integrator_2d(z, t; Δt, state_dimension=2, control_dimension=2)

2D single integrator dynamics constraint (linear).
State: x = [x, y] (position)
Control: u = [vx, vy] (velocity)

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function single_integrator_2d(z, t; Δt, state_dimension = 2, control_dimension = 2)
    (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)

    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t + 1]

    x_pred = x_t .+ Δt .* u_t
    return x_tp1 - x_pred
end

"""
	single_integrator_3d(z, t; Δt, state_dimension=3, control_dimension=3)

3D single integrator dynamics constraint (linear).
State: x = [x, y, z] (position)
Control: u = [vx, vy, vz] (velocity)

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function single_integrator_3d(z, t; Δt, state_dimension = 3, control_dimension = 3)
    (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)

    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t + 1]

    x_pred = single_integrator_3d_step(x_t, u_t; Δt)
    # x_pred =  .+ Δt .* u_t
    return x_tp1 - x_pred
end

function single_integrator_3d_step(x, u; Δt)
    x .+ Δt .* u
end
