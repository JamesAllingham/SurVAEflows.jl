@testset "VAE" begin
    struct Encoder
        linear
        μ
        logσ
        Encoder(input_dim, latent_dim, hidden_dim, device) = new(
            Dense(input_dim, hidden_dim, tanh) |> device,   # linear
            Dense(hidden_dim, latent_dim) |> device,        # μ
            Dense(hidden_dim, latent_dim) |> device,        # logσ
        )
    end

    function (encoder::Encoder)(x)
        h = encoder.linear(x)
        encoder.μ(h), encoder.logσ(h)
    end

    struct Decoder
        linear
        x̂
        Decoder(input_dim, latent_dim, hidden_dim, device) = new(
            Dense(latent_dim, hidden_dim, tanh) |> device,   # linear
            Dense(hidden_dim, input_dim) |> device,        # x̂
        )
    end

    function (decoder::Decoder)(x)
        h = decoder.linear(x)
        decoder.x̂(h)
    end

    input_dim = 4
    latent_dim =2
    hidden_dim = 8
    vae = SurVAEflows.VAE(
        Encoder(input_dim, latent_dim, hidden_dim, cpu),
        Decoder(input_dim, latent_dim, hidden_dim, cpu)
    )

    loader = collect(DataLoader(Iris.features(), batchsize=5))

    @testset "sizes" begin
        x = popfirst!(loader)
        z = SurVAEflows.inverse(vae, x)
        x̂ = SurVAEflows.forward(vae, z)
        V = SurVAEflows.V(vae, x, z)

        @test size(x̂) == size(x)
        @test size(z)[1] == latent_dim
        @test size(V) == (1, size(x)[end])
    end

    @testset "gradient descent" begin

    end
end