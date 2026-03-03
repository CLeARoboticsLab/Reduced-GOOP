using JLD2
using CairoMakie

include(joinpath(@__DIR__, "Plotting.jl"))

run_id = "run_NQP_2players_4prefs_0.1ρ_10pdim_3mₑ_2mᵢ"
run_dir = joinpath(@__DIR__, "..", "data", "QP_benchmark", run_id)
histories_dir = joinpath(run_dir, "data", "histories")
convergence_plots_dir = joinpath(run_dir, "plots", "convergence")
mkpath(convergence_plots_dir)

cd(histories_dir) do
	reduced_aggregate_file = "kkt_error_histories_reduced.jld2"
	complete_aggregate_file = "kkt_error_histories_complete.jld2"

	if isfile(reduced_aggregate_file) && isfile(complete_aggregate_file)
		reduced_histories = JLD2.load(reduced_aggregate_file, "kkt_error_histories_reduced")
		complete_histories = JLD2.load(complete_aggregate_file, "kkt_error_histories_complete")
	else
		reduced_histories = Vector{Vector{Float64}}()
		complete_histories = Vector{Vector{Float64}}()

		for instance_idx in 1:10
			reduced_file = "kkt_error_history_reduced_instance_$(instance_idx).jld2"
			complete_file = "kkt_error_history_complete_instance_$(instance_idx).jld2"

			isfile(reduced_file) || error("Missing file: $(reduced_file)")
			isfile(complete_file) || error("Missing file: $(complete_file)")

			reduced_history = JLD2.load(reduced_file, "kkt_error_history")
			complete_history = JLD2.load(complete_file, "kkt_error_history")

			push!(reduced_histories, reduced_history)
			push!(complete_histories, complete_history)
		end

		JLD2.save(reduced_aggregate_file, "kkt_error_histories_reduced", reduced_histories)
		JLD2.save(complete_aggregate_file, "kkt_error_histories_complete", complete_histories)
	end

	fig, _ = plot_convergence_plot_aggregate_comparison(
		;
		reduced_kkt_error_histories = reduced_histories,
		complete_kkt_error_histories = complete_histories,
		show_legend = false,
		show_ylabel = false,
	)

	output_path = joinpath(convergence_plots_dir, "temp_convergence_aggregate_reduced_vs_complete.pdf")
	CairoMakie.save(output_path, fig)
	println("Saved: $(output_path)")
end
