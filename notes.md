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
includet("experiments/Robotic_arm_final.jl");
# run the demo: 50 closed-loop (receding-horizon) steps
Robotic_arm_final.demo(receding_horizon = 50);
```

The demo draws its figures (`plot_fig = true` by default) and writes no data files.