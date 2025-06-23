# Ordered Preferences Playground

This repository is **our private playground** for the research project on Games Of Ordered Preference (GOOP).

The original paper is ["You Can't Always Get What You Want: Games of Ordered Preference"](https://arxiv.org/abs/2410.21447) by Dong Ho Lee, Lasse Peters and David Fridovich-Keil.

This private version of the repository was partially copied from their [GitHub repo](https://github.com/CLeARoboticsLab/ordered-preferences) by [CLeARoboticsLab](https://clearoboticslab.github.io/).

## Setup

### Install Julia

```sh
curl -fsSL https://install.julialang.org | sh
```

### Set up PATH License

Get a (free) license from [the official page](https://pages.cs.wisc.edu/~ferris/path/julia/LICENSE).

```sh
export PATH_LICENSE_STRING="2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0"
# To make it persistent, run
echo 'export PATH_LICENSE_STRING="2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0"' >> ~/.bashrc
```

### Environment

First, make sure you to activate the environment and instantiate it (install missing packages) from a REPL in the root directory of the repository with:

```sh
julia
```

```julia
] activate .
] instantiate
```

There might be compilation errors if you don't have a X11 session (e.g. in ssh).

You might need to add the packet ``OrderedPreferences.jl``, To do so with the local copy do:

```julia
] add .
```

If needed, you can update packages to the latest possible version with:

```julia
] up
```

However, be careful! You might get dependency errors when updating everything!

## Run things

To generate some samples:

```julia
> include("experiments/Experiments.jl")
> Experiments.generate_samples()
```

To run a solver, play with these:

```julia
> Experiments.run(Experiments.ParametricGamePenalty; num_samples=1)
> Experiments.run(Experiments.ParametricOrderedPreferencesMPCCGame;)
```

You can reuse the problem if it is from the same type/solver.

```julia
> (; problem) = Experiments.run(Experiments.ParametricOrderedPreferencesMPCCGame; solver="PATH")
> Experiments.run(Experiments.ParametricOrderedPreferencesMPCCGame; solver="PATH", problem)
```

Finally, the new solver (for now, WIP).

```julia
> Experiments.run(Experiments.ParametricOrderedPreferencesMPCCGame; solver="IP")
> Experiments.run(Experiments.ParametricOrderedPreferencesMPCCGame; solver="IP", num_samples=1, warmstart_samples=2)
```

Each run will store things in a different directory.
You can compare two directories this way:

```sh
> Experiments.compare_two("directory_one", "directory_two")
```

This will store all sorts of overlapped graphs in a new directory.
