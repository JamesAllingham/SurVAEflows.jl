module SurVAEflows

    using Distributions
    using Flux
    using LinearAlgebra
    using Statistics

    # include distribution defs
    include(joinpath("distributions", "distributions.jl"))
    include(joinpath("distributions", "conditional", "normal.jl"))

    # include transform defs
    include(joinpath("transforms", "transforms.jl"))
    include(joinpath("transforms", "stochastic", "VAE.jl"))
    include(joinpath("transforms", "surjections", "rounding.jl"))

    # include utils
    include(joinpath("tensor_utils.jl"))


    struct Flow # TODO: maybe <: Distribution?
        base_dist::Distribution
        transforms::Array{Transform,1}
    end

    function forward(flow::Flow, z)
        for t in flow.transforms
            x = forward(t, z)
            z = x
        end
        x
    end  # function forward

    function inverse(flow::Flow, x)
        for t in flow.transforms
            z = inverse(t, x)
            x = z
        end
        z
    end  # function inverse

    function sample(flow::Flow, n)
        samples = []
        for _ in 1:n
            z = sample(flow.base_dist)
            push!(samples, forward(flow, z))
        end
    end  # function sample

    function log_prob(flow::Flow, x)
        Vs = []
        for t in flow.transforms
            z = inverse(t, x)
            push!(Vs, V(t, x, z))
            x = z
        end
        logp_z = log_prob(flow.base_dist, z)

        logp_z + sum(Vs)
    end  # function log_prob

end # module