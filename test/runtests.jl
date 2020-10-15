# include("src/SurVAEflows.jl")
# include("test/runtests.jl")
using SurVAEflows
using Flux
using Flux.Data: DataLoader
using MLDatasets: Iris

using Test

@testset "SurVAEflows.jl" begin
    include(joinpath("transforms", "stochastic", "VAE.jl"))
end