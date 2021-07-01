using OptimalTransport

using Distances
using ForwardDiff
using LogExpFunctions
using PythonOT: PythonOT
using Distributions

using LinearAlgebra
using Random
using Test

const POT = PythonOT

Random.seed!(100)

@testset "sinkhorn divergence" begin
    @testset "univariate exmaples" begin
        # create distributions 
        n = 20
        m = 10
        μsupp = [rand(1) for i in 1:n]
        νsupp = [rand(1) for i in 1:m]
        μprobs = normalize!(rand(n), 1)
        μ = OptimalTransport.discretemeasure(μsupp, μprobs)
        ν = OptimalTransport.discretemeasure(νsupp)

        metrics_list = [
            (sqeuclidean, SqEuclidean()),
            (euclidean, Euclidean()),
            (totalvariation, TotalVariation()),
        ]
        for ε in [0.1, 1.0, 10.0], metrics in metrics_list
            for metric in metrics
                @test sinkhorn_divergence(metric, μ, μ, ε) ≈ 0.0
                @test sinkhorn_divergence(metric, ν, ν, ε) ≈ 0.0

                sd_c = sinkhorn_divergence(metric, μ, ν, ε)

                # calculating cost matrices to use in POT.sinkhorn2
                Cμν = pairwise(metric, μ.support, ν.support)
                Cμ = pairwise(metric, μ.support)
                Cν = pairwise(metric, ν.support)

                sd_C = sinkhorn_divergence(Cμν, Cμ, Cν, μ, ν, ε)

                # the empirical_sinkhorn_divergence returns an error if the weights are not all equal
                # so instead, it's more realiable to calculate using sinkhorn2
                sd_pot =
                    POT.sinkhorn2(μ.p, ν.p, Cμν, ε) -
                    (POT.sinkhorn2(μ.p, μ.p, Cμ, ε) + POT.sinkhorn2(ν.p, ν.p, Cν, ε)) / 2

                @test sd_c ≈ sd_pot[1]
                @test sd_C ≈ sd_pot[1]
            end
        end
    end
    @testset "multivariate exmaples" begin
        # create distributions 
        n = 20
        m = 10
        μsupp = [rand(3) for i in 1:n]
        νsupp = [rand(3) for i in 1:m]
        μprobs = normalize!(rand(n), 1)
        μ = OptimalTransport.discretemeasure(μsupp, μprobs)
        ν = OptimalTransport.discretemeasure(νsupp)

        metrics_list = [
            (sqeuclidean, SqEuclidean()),
            (euclidean, Euclidean()),
            (totalvariation, TotalVariation()),
        ]
        for ε in [0.1, 1.0, 10.0], metrics in metrics_list
            for metric in metrics
                @test sinkhorn_divergence(metric, μ, μ, ε) ≈ 0.0
                @test sinkhorn_divergence(metric, ν, ν, ε) ≈ 0.0

                sd_c = sinkhorn_divergence(metric, μ, ν, ε)

                # calculating cost matrices to use in POT.sinkhorn2
                Cμν = pairwise(metric, μ.support, ν.support)
                Cμ = pairwise(metric, μ.support)
                Cν = pairwise(metric, ν.support)

                sd_C = sinkhorn_divergence(Cμν, Cμ, Cν, μ, ν, ε)

                # the empirical_sinkhorn_divergence returns an error if the weights are not all equal
                # so instead, it's more realiable to calculate using sinkhorn2
                sd_pot =
                    POT.sinkhorn2(μ.p, ν.p, Cμν, ε) -
                    (POT.sinkhorn2(μ.p, μ.p, Cμ, ε) + POT.sinkhorn2(ν.p, ν.p, Cν, ε)) / 2

                @test sd_c ≈ sd_pot[1]
                @test sd_C ≈ sd_pot[1]
            end
        end
    end
end
