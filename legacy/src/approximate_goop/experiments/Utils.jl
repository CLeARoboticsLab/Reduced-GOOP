function save_object(file, x)
    mkpath(dirname(file))
    JLD2.save_object(file, x)
end

function save_plot(file, fig)
    mkpath(dirname(file))
    CairoMakie.save(file, fig)
end
