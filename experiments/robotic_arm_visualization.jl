#=
	Shared visualization support for the robotic-arm experiments.

	This file is included after importing `Main.RoboticArmCore` by both the
	open-loop and receding-horizon entry points. It loads the generic plotting
	helpers and defines the robotic-arm-specific visualization configuration and
	output routines used by both experiments.
=#

using LaTeXStrings: @L_str

include(joinpath(@__DIR__, "plotting.jl"))
include(joinpath(@__DIR__, "3d_plotting.jl"))

"""Output locations and display/file-format choices for experiment plots."""
Base.@kwdef struct VisualizationConfig{D}
	dirs::D
	show_interactive_trajectory::Bool = false
	static_extension::String = "pdf"
	interactive_extension::String = "html"
end

function save_warmstart_visualizations(
	warmstart_solution;
	total_attempts = nothing,
	instance_idx = nothing,
	filename_tag = "attempt_$(total_attempts)_instance_$(instance_idx)",
	primal_dimensions,
	instance_parameters::InstanceParameters,
	scenario_config::ScenarioConfig,
	visualization_config::VisualizationConfig,
)
	(; θ1, θ2, θ3) = instance_parameters
	(;
		dynamics,
		map_end,
		lane_width,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
		dₚ,
		arm_speed_limit,
		child_speed_limit,
		control_bounds,
	) = scenario_config
	(; warmstart_plots_dir) = visualization_config.dirs
	static_extension = visualization_config.static_extension
	# Plots keep the per-arm view: split the combined two-arm strategy back
	# into [arm1, arm2, child].
	warmstart_strategies = split_arm_strategies(
		extract_player_strategies(warmstart_solution, primal_dimensions, dynamics),
	)

	warmstart_fig, _ = plot_single_integrator_3d_trajectories(;
		map_end,
		lane_width,
		strategy = warmstart_strategies,
		θ1,
		θ2,
		θ3,
		goal_position1,
		goal_position2,
		goal_position3,
		collision_avoidance,
	)
	warmstart_speed_fig, _ = speed_plot(;
		strategy = warmstart_strategies,
		speed_limit = arm_speed_limit,
		dynamics_model = dynamics[1].model,
		speed_limit_players = 1:2,
		additional_speed_limits = [(; limit = child_speed_limit, players = 3)],
	)
	warmstart_control_fig, _ = control_plot(;
		strategy = warmstart_strategies,
		control_lb = control_bounds.lb,
		control_ub = control_bounds.ub,
	)
	warmstart_distance_fig, _ = inter_player_distance_plot(;
		strategy = warmstart_strategies,
		reference_distance = dₚ,
		safety_distance = collision_avoidance,
	)

	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_$(filename_tag).$(static_extension)",
		),
		warmstart_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_speed_$(filename_tag).$(static_extension)",
		),
		warmstart_speed_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_control_$(filename_tag).$(static_extension)",
		),
		warmstart_control_fig,
	)
	save_figure(
		joinpath(
			warmstart_plots_dir,
			"warmstart_distance_$(filename_tag).$(static_extension)",
		),
		warmstart_distance_fig,
	)
end

function safe_log10_history(history)
	map(history) do value
		isfinite(value) && value > 0 ? log10(value) : NaN
	end
end

function save_convergence_diagnostics(
	solution_dict,
	convergence_plots_dir,
	instance_idx,
	ϵ₀;
	filename_suffix = "",
)
	kkt_error_history = get(solution_dict, "kkt_error_history", Float64[])
	if !isempty(kkt_error_history)
		convergence_fig, _ = plot_convergence_plot(;
			kkt_error_history = safe_log10_history(kkt_error_history),
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"convergence_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			convergence_fig,
		)
	end

	condition_number_history = get(solution_dict, "condition_number_history", Float64[])
	if !isempty(condition_number_history)
		condition_number_fig, _ = plot_condition_number_plot(;
			condition_number_history = safe_log10_history(condition_number_history),
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"condition_number_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			condition_number_fig,
		)
	end

	eta_history = get(solution_dict, "eta_history", Float64[])
	if !isempty(eta_history)
		eta_fig, _ = plot_eta_plot(;
			eta_history = safe_log10_history(eta_history),
			total_iters = solution_dict["total_iters"],
		)
		save_figure(
			joinpath(
				convergence_plots_dir,
				"eta_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			eta_fig,
		)
	end

	alpha_history = get(solution_dict, "alpha_history", Float64[])
	if !isempty(alpha_history)
		alpha_fig, _ =
			plot_alpha_plot(; alpha_history, total_iters = solution_dict["total_iters"])
		save_figure(
			joinpath(
				convergence_plots_dir,
				"alpha_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			alpha_fig,
		)
	end

	rho_history = get(solution_dict, "rho_history", Float64[])
	if !isempty(rho_history)
		rho_fig, _ =
			plot_rho_plot(; rho_history, total_iters = solution_dict["total_iters"])
		save_figure(
			joinpath(
				convergence_plots_dir,
				"rho_instance_$(instance_idx)_eps$(ϵ₀)$(filename_suffix).pdf",
			),
			rho_fig,
		)
	end
end
