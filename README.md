# SurVAEflows.jl
SurVAE Flows in Julia!

Heavily inspired by https://github.com/didriknielsen/survae_flows and, of course, https://arxiv.org/abs/2007.02731.

To setup the environment run the following from this directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
# or
] activate
] instantiate
```

Then to use:

```julia
Pkg.activate(".")
using SurVAEflows
vae = SurVAEflows.VAE(enc, dec)
# or
Pkg.activate(".")
using SurVAEflows: VAE
vae = VAE(enc, dec)
```