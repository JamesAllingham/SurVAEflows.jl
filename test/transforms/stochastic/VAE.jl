@testset "VAE" begin
    struct NN
        linear
        μ
        logσ
        NN(input_dim, latent_dim, hidden_dim, device) = new(
            Dense(input_dim, hidden_dim, tanh) |> device,   # linear
            Dense(hidden_dim, latent_dim) |> device,        # μ
            Dense(hidden_dim, latent_dim) |> device,        # logσ
        )
    end

    function (nn::NN)(x)
        h = nn.linear(x)
        nn.μ(h), nn.logσ(h)
    end

    input_dim = 4
    latent_dim =2
    hidden_dim = 8
    vae = SurVAEflows.VAE(
        SurVAEflows.ConditionalNormal(NN(input_dim, latent_dim, hidden_dim, cpu)),
        SurVAEflows.ConditionalNormal(NN(latent_dim, input_dim, hidden_dim, cpu))
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

    function normal_logprob(x, μ, logσ)
        SurVAEflows.batchwise_sum(@. -0.5log(2π) - logσ - 0.5 * ((x - μ)/exp(logσ))^2)
    end  # function normal_logprob

    function loss(x)
        z = SurVAEflows.inverse(vae, x)
        V = SurVAEflows.V(vae, x, z)

        sum(normal_logprob(z, 0, 1) + V)
    end  # function loss

    @testset "gradient descent" begin
        opt = Descent()
        x = popfirst!(loader)
        loss1 = loss(x)

        ps = Flux.params(vae)

        gs = gradient(ps) do
            loss(x)
        end

        Flux.update!(opt, ps, gs)

        loss2 = loss(x)

        @test loss1 > loss2
    end
end