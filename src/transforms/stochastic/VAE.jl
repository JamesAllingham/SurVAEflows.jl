struct VAE <: StochasticTransform
    encoder::ConditionalDistribution
    decoder::ConditionalDistribution
end

function forward(vae::VAE, z)
    x̂ = sample(vae.decoder, z)
end  # function forward

function inverse(vae::VAE, x)
    z = sample(vae.encoder, x)
end  # function inverse

function V(vae::VAE, x, z)
    log_qz_x = logprob(vae.encoder, z, x)
    log_px_z = logprob(vae.decoder, x, z)
    ldj = log_px_z - log_qz_x
end  # function V
