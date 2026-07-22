To clone the repo in Windows (avoiding NTFS issues), follow these steps:

```bash
git clone ...
cd Reduced-GOOP
git config core.protectNTFS false
git checkout
```

To set up the Julia environment and run the ICRA experiments, follow these steps:

```julia
# pkg
add Revise, Infiltrator, ProgressMeter, JLD2, Dates, BenchmarkTools, JuliaFormatter  # useful packages for development, only need to add once
activate experiments/; resolve; instantiate;
# run
using Revise; using Infiltrator;
includet("experiments/Robotic_arm.jl");
includet("experiments/Robotic_arm_receding.jl");
# run the demo and save the results
Robotic_arm.demo(; save=true, plot=true);
Robotic_arm_receding.demo(;num_mpc_steps=2);
# retrieve the solution dictionary
solution_dict = Robotic_arm.demo();
```

When the corresponding options are enabled,
results will be saved to ``data/robotic_arm_open_loop/runs/Robotic_arm/plots``