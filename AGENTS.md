# Repository Agent Notes

## Julia environments

- Use the root `Project.toml`/`Manifest.toml` for the core `ReducedGOOP` package and its tests.
- The robotic-arm examples have a separate environment in `experiments/`. They import example-only dependencies such as `AppleAccelerate`, `CairoMakie`, and `JLD2` that are intentionally absent from the root environment.
- Run commands that include `experiments/Robotic_arm.jl` or `experiments/Robotic_arm_receding.jl` from the repository root with `julia --project=experiments ...`.
- If a robotic-arm command run with `--project=.` reports that one of those example dependencies is missing, switch to the existing `experiments` environment. Do not add or install the package into the root environment.
