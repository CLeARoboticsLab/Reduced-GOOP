using Dates
using GLMakie

# one, two are directory names inside of /data/relaxably_feasible
function compare_two(one, two; name_one = "First", name_two = "Second")

    data_folder = "./data/"
    relaxably_feasible_folder = joinpath(data_folder, "relaxably_feasible")
    folder_one = joinpath(relaxably_feasible_folder, one)
    folder_two = joinpath(relaxably_feasible_folder, two)

    out_folder = joinpath(relaxably_feasible_folder, string("benchmarks_", Dates.format(now(), "yyyymmdd_HHMMSS")))

    # runtime (we suppose both directories run the same thing the same amount of times)
    runtime_out = joinpath(out_folder, "runtime.png")
    println(string("Saving runtime plot to ", runtime_out))
    generic_plot(
        [
            Dict(
                :data => retrieve_runtime_info(joinpath(folder_one, "runtime")), 
                :label => name_one, :color => :red, :marker => :circle
            ),
            Dict(
                :data => retrieve_runtime_info(joinpath(folder_two, "runtime")),
                :label => name_two, :color => :blue, :marker => :star5
            )
        ];
        title="Runtime",
        y_label="Problem",
        x_label="Time (s)",
        save_to=runtime_out,
    )

    relaxation_one = Float64[]
    relaxation_two = Float64[]

    residual_one = Float64[]
    residual_two = Float64[]

    @showprogress desc = "Benchmarking ..." for file_one in readdir(joinpath(folder_one, "solutions"); join=true)
        file_two = joinpath(folder_two, "solutions", basename(file_one))
        
        if endswith(file_one, "rfp_equilibrium.jld2")
            continue
        end

        contents_one = load(file_one, "single_stored_object")
        if !isfile(file_two)
            @warn "File $file_two not available, skipping"
            continue
        end
        contents_two = load(file_two, "single_stored_object")

        ## Speed

        horizontal_speed_data = Vector{Vector{Float64}}[]
        vertical_speed_data = Vector{Vector{Float64}}[]
        push!(
            horizontal_speed_data,
            [
                vcat(contents_one["strategy1"].xs...)[3:4:end],
                vcat(contents_one["strategy2"].xs...)[3:4:end],
                vcat(contents_one["strategy3"].xs...)[3:4:end],
                vcat(contents_two["strategy1"].xs...)[3:4:end],
                vcat(contents_two["strategy2"].xs...)[3:4:end],
                vcat(contents_two["strategy3"].xs...)[3:4:end],
            ],
        )
        push!(
            vertical_speed_data,
            [
                vcat(contents_one["strategy1"].xs...)[4:4:end],
                vcat(contents_one["strategy2"].xs...)[4:4:end],
                vcat(contents_one["strategy3"].xs...)[4:4:end],
                vcat(contents_two["strategy1"].xs...)[4:4:end],
                vcat(contents_two["strategy2"].xs...)[4:4:end],
                vcat(contents_two["strategy3"].xs...)[4:4:end],
            ],
        )

        generic_plot(
            [
                Dict(
                    :data => horizontal_speed_data[1][1], 
                    :label => string("v1 ", name_one), :color => :red, :marker => :circle
                ),
                Dict(
                    :data => horizontal_speed_data[1][2], 
                    :label => string("v2 ", name_one), :color => :green, :marker => :star4
                ),
                Dict(
                    :data => horizontal_speed_data[1][3], 
                    :label => string("v3 ", name_one), :color => :blue, :marker => :star5
                ),

                Dict(
                    :data => horizontal_speed_data[1][4], 
                    :label => string("v1 ", name_two), :color => :red, :marker => :circle
                ),
                Dict(
                    :data => horizontal_speed_data[1][5], 
                    :label => string("v2 ", name_two), :color => :green, :marker => :star4
                ),
                Dict(
                    :data => horizontal_speed_data[1][6], 
                    :label => string("v3 ", name_two), :color => :blue, :marker => :star5
                ),
            ];
            title="Horizontal Speed",
            y_label="Frame",
            x_label="Horizontal Speed",
            save_to=joinpath(out_folder, string(basename(file_one), "_hor_speed.png")),
        )

        generic_plot(
            [
                Dict(
                    :data => vertical_speed_data[1][1], 
                    :label => string("v1 ", name_one), :color => :red, :marker => :circle
                ),
                Dict(
                    :data => vertical_speed_data[1][2], 
                    :label => string("v2 ", name_one), :color => :green, :marker => :star4
                ),
                Dict(
                    :data => vertical_speed_data[1][3], 
                    :label => string("v3 ", name_one), :color => :blue, :marker => :star5
                ),

                Dict(
                    :data => vertical_speed_data[1][4], 
                    :label => string("v1 ", name_two), :color => :red, :marker => :circle
                ),
                Dict(
                    :data => vertical_speed_data[1][5], 
                    :label => string("v2 ", name_two), :color => :green, :marker => :star4
                ),
                Dict(
                    :data => vertical_speed_data[1][6], 
                    :label => string("v3 ", name_two), :color => :blue, :marker => :star5
                ),
            ];
            title="Vertical Speed",
            y_label="Frame",
            x_label="Vertical Speed",
            save_to=joinpath(out_folder, string(basename(file_one), "_ver_speed.png")),
        )

        # openloop

        planning_horizon = 5 # TODO -> un-hardcode

        openloop_distance1_one = Vector{Float64}[]
        openloop_distance2_one = Vector{Float64}[]
        openloop_distance3_one = Vector{Float64}[]
        push!(
            openloop_distance1_one,
            [
                sqrt(sum((contents_one["strategy1"].xs[k][1:2] - contents_one["strategy2"].xs[k][1:2]) .^ 2)) for
                k in 1:planning_horizon
            ],
        )
        push!(
            openloop_distance2_one,
            [
                sqrt(sum((contents_one["strategy1"].xs[k][1:2] - contents_one["strategy3"].xs[k][1:2]) .^ 2)) for
                k in 1:planning_horizon
            ],
        )
        push!(
            openloop_distance3_one,
            [
                sqrt(sum((contents_one["strategy2"].xs[k][1:2] - contents_one["strategy3"].xs[k][1:2]) .^ 2)) for
                k in 1:planning_horizon
            ],
        )
        openloop_distance1_two = Vector{Float64}[]
        openloop_distance2_two = Vector{Float64}[]
        openloop_distance3_two = Vector{Float64}[]
        push!(
            openloop_distance1_two,
            [
                sqrt(sum((contents_two["strategy1"].xs[k][1:2] - contents_two["strategy2"].xs[k][1:2]) .^ 2)) for
                k in 1:planning_horizon
            ],
        )
        push!(
            openloop_distance2_two,
            [
                sqrt(sum((contents_two["strategy1"].xs[k][1:2] - contents_two["strategy3"].xs[k][1:2]) .^ 2)) for
                k in 1:planning_horizon
            ],
        )
        push!(
            openloop_distance3_two,
            [
                sqrt(sum((contents_two["strategy2"].xs[k][1:2] - contents_two["strategy3"].xs[k][1:2]) .^ 2)) for
                k in 1:planning_horizon
            ],
        )

        # plot distance

        generic_plot(
            [
                Dict(
                    :data => openloop_distance1_one[1], 
                    :label => string("B/w Agent 1 & Agent 2 - ", name_one), :color => :red, :marker => :circle
                ),
                Dict(
                    :data => openloop_distance2_one[1], 
                    :label => string("B/w Agent 1 & Agent 3 - ", name_one), :color => :green, :marker => :star4
                ),
                Dict(
                    :data => openloop_distance3_one[1], 
                    :label => string("B/w Agent 2 & Agent 3 ", name_one), :color => :blue, :marker => :star5
                ),

                Dict(
                    :data => openloop_distance1_two[1], 
                    :label => string("B/w Agent 1 & Agent 2 - ", name_two), :color => :red, :marker => :circle
                ),
                Dict(
                    :data => openloop_distance2_two[1], 
                    :label => string("B/w Agent 1 & Agent 3 - ", name_two), :color => :green, :marker => :star4
                ),
                Dict(
                    :data => openloop_distance3_two[1], 
                    :label => string("B/w Agent 2 & Agent 3 ", name_two), :color => :blue, :marker => :star5
                ),

                # Collision limit
                Dict(
                    :data => [0.2 for _ in 0:(planning_horizon - 1)],
                    :label => "Collision distance", :color => :grey, :marker => :hline
                ),

            ];
            title="Vehicle Distance",
            y_label="Frame",
            x_label="Vehicle Distance",
            save_to=joinpath(out_folder, string(basename(file_one), "_distance.png")),
        )

        # End

        push!(relaxation_one, contents_one["relaxation"])
        push!(relaxation_two, contents_two["relaxation"])
        push!(residual_one, contents_one["residual"])
        push!(residual_two, contents_two["residual"])

        ## GIF
        datasets1 = [
            contents_one["strategy1"].xs,
            contents_one["strategy2"].xs,
            contents_one["strategy3"].xs,
        ]
        datasets2 = [
            contents_two["strategy1"].xs,
            contents_two["strategy2"].xs,
            contents_two["strategy3"].xs,
        ]
        colors = [:red, :blue, :green]
        safe_distance_radius = 0.2

        fig = Figure(size=(1000, 500))
        ax1 = Axis(fig[1, 1], title="Vehicle Positions ($name_one)", xlabel="X Position", ylabel="Y Position")
        ax2 = Axis(fig[1, 2], title="Vehicle Positions ($name_two)", xlabel="X Position", ylabel="Y Position")
        limits!(ax1, -1, 1, -1, 1)
        limits!(ax2, -1, 1, -1, 1)

        # Initialize scatter plots, circles, and arrows for both datasets
        scatters1 = [scatter!(ax1, [0.0], [0.0], color=c, markersize=10) for c in colors]
        scatters2 = [scatter!(ax2, [0.0], [0.0], color=c, markersize=10) for c in colors]

        circles1 = [lines!(ax1, cos.(range(0, 2π, length=100)) * safe_distance_radius .+ 0.0,
                        sin.(range(0, 2π, length=100)) * safe_distance_radius .+ 0.0,
                        color=(c, 0.2), linewidth=1.5) for c in colors]
        circles2 = [lines!(ax2, cos.(range(0, 2π, length=100)) * safe_distance_radius .+ 0.0,
                        sin.(range(0, 2π, length=100)) * safe_distance_radius .+ 0.0,
                        color=(c, 0.2), linewidth=1.5) for c in colors]

        arrows1 = [arrows!(ax1, [0.0], [0.0], [0.0], [0.0], color=c, linewidth=1.5, linestyle=(:dot, 1.0)) for c in colors]
        arrows2 = [arrows!(ax2, [0.0], [0.0], [0.0], [0.0], color=c, linewidth=1.5, linestyle=(:dot, 1.0)) for c in colors]

        # Animation function
        record(fig, joinpath(out_folder, string(basename(file_one), "_dual_trajectory.gif")), 1:length(datasets1[1]); framerate=2) do i
            # Update Dataset 1
            for (s, d, c, a) in zip(scatters1, datasets1, circles1, arrows1)
                x, y, vx, vy = d[i]
                
                s[1] = [x]
                s[2] = [y]

                cx, cy = x .+ safe_distance_radius * cos.(range(0, 2π, length=100)), 
                        y .+ safe_distance_radius * sin.(range(0, 2π, length=100))

                c[1] = cx
                c[2] = cy

                a[1] = [x]
                a[2] = [y]
                a[3] = [vx]
                a[4] = [vy]
            end

            # Update Dataset 2
            for (s, d, c, a) in zip(scatters2, datasets2, circles2, arrows2)
                x, y, vx, vy = d[i]
                
                s[1] = [x]
                s[2] = [y]

                cx, cy = x .+ safe_distance_radius * cos.(range(0, 2π, length=100)), 
                        y .+ safe_distance_radius * sin.(range(0, 2π, length=100))

                c[1] = cx
                c[2] = cy

                a[1] = [x]
                a[2] = [y]
                a[3] = [vx]
                a[4] = [vy]
            end
        end
        ## END GIF
    end

    relaxation_out = joinpath(out_folder, "relaxation.png")
    println(string("Saving relaxation plot to ", relaxation_out))
    generic_plot(
        [
            Dict(
                :data => relaxation_one, 
                :label => name_one, :color => :red, :marker => :circle
            ),
            Dict(
                :data => relaxation_two,
                :label => name_two, :color => :blue, :marker => :star5
            )
        ];
        title="Relaxation",
        y_label="Problem",
        x_label="Relaxation",
        save_to=relaxation_out,
    )

    residual_out = joinpath(out_folder, "residual.png")
    println(string("Saving relaxation plot to ", residual_out))
    generic_plot(
        [
            Dict(
                :data => residual_one, 
                :label => name_one, :color => :red, :marker => :circle
            ),
            Dict(
                :data => residual_two,
                :label => name_two, :color => :blue, :marker => :star5
            )
        ];
        title="Residual",
        y_label="Problem",
        x_label="Residual",
        save_to=residual_out,
    )

    println(string("Benchmark done! Check ", out_folder))
end

function retrieve_runtime_info(dir)
    data = Float64[]
    for file in readdir(dir; join=true)
        if endswith(file, ".jld2")
            loaded_data = load(file, "single_stored_object")
            append!(data, loaded_data)
        end
    end

    return data
end

function generic_plot(data; title, x_label, y_label, save_to)
    fig = CairoMakie.Figure()
    ax4 = CairoMakie.Axis(
        fig[1, 1];
        xlabel = y_label,
        ylabel = x_label,
        title = title,
    )

    # Data is an array of dicts
    for element in data
        CairoMakie.scatterlines!(
            ax4,
            0:length(element[:data]) - 1,
            element[:data],
            label = element[:label],
            color = element[:color],
            marker = element[:marker],
            markersize = 10,
        )
    end

    fig[2, 1] = CairoMakie.Legend(fig, ax4, framevisible = false, orientation = :horizontal)
    save_plot(save_to, fig)
end
