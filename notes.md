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
includet("experiments/Robotic_arm_receding.jl");
# run the demo and save the results
Robotic_arm_receding.demo(;num_mpc_steps=50);
```

When the corresponding options are enabled,
results will be saved to ``data/robotic_arm_receding_horizon/runs/``.