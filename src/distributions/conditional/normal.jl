# 𝒩(x|μ, logσ) = 𝒩(x|nn(z))
struct ConditionalNormal <: ConditionalDistribution
    nn
end

function sample(dist::ConditionalNormal, z)
    μ, logσ = dist.nn(z)
    @assert size(μ) == size(logσ)

    reparam_trick(μ, logσ)
end  # function sample

function logprob(dist::ConditionalNormal, x, z)
    μ, logσ = dist.nn(z)
    @assert size(μ) == size(logσ)

    normal_logprob(x, μ, logσ)
end  # function logprob

function sample_with_logprob(dist::ConditionalNormal, z)
    μ, logσ = dist.nn(z)
    @assert size(μ) == size(logσ)

    x = reparam_trick(μ, logσ)
    logprob = normal_logprob(x, μ, logσ)

    x, logprob
end  # function sample_with_logprob

# TODO add ConditionalNormals with fixed and homoscedastic learned std

function normal_logprob(x, μ, logσ)
    batchwise_sum(@. -0.5log(2π) - logσ - 0.5 * ((x - μ)/exp(logσ))^2)
end  # function normal_logprob

# TODO: maybe we don't need this and we can back-prop through std Distributions?
function reparam_trick(μ, logσ)
    μ + rand(size(logσ)) .* exp.(logσ)
end  # function reparam_trick