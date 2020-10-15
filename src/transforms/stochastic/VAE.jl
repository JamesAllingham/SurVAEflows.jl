

struct VAE <: StochasticTransform
    encoder
    decoder
end

function forward(vae::VAE, z)
    x̂ = σ.(vae.decoder(z))
end  # function forward

function inverse(vae::VAE, x)
    # sample z
    μ_z, logσ_z = vae.encoder(x)
    z = reparam_trick(μ_z, logσ_z)
end  # function inverse

function V(vae::VAE, x, z)
    μ_z, logσ_z = vae.encoder(x)
    log_qz_x = batchwise_sum(normal_logprob.(z, μ_z, logσ_z))
    x̂ = vae.decoder(z)
    log_px_z = batchwise_sum(-Flux.Losses.logitbinarycrossentropy.(x̂, x))
    ldj = log_px_z - log_qz_x
end  # function V

function normal_logprob(x, μ, logσ)
    -0.5log(2π) - logσ - 0.5 * ((x - μ)/exp(logσ))^2
end  # function normal_logprob

function reparam_trick(μ, logσ)
    μ + randn(size(logσ)) .* exp.(logσ)
end  # function reparam_trick