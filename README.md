# SurVAEflows.jl
SurVAE Flows in Julia!

Heavily inspired by https://github.com/didriknielsen/survae_flows and, of course, https://arxiv.org/abs/2007.02731.

To setup the environment run the following from this directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then to use:

```julia
using Revise # useful for auto-reloading changes to SurVAEflows
include("src/SurVAEflows.jl")
using .SurVAEflows
vae = SurVAEflows.VAE(enc, dec)
# or
using .SurVAEflows: VAE, Flow
vae = VAE(enc, dec)
```