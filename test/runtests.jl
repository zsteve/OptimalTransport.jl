using OptimalTransport

using CUDA
using Distances
using PyCall
using Tulip
using MathOptInterface
using Distributions
using SparseArrays

using LinearAlgebra
using Random
using Test

const MOI = MathOptInterface

Random.seed!(100)

@testset "Earth-Movers Distance" begin
    M = 200
    N = 250
    μ = rand(M)
    ν = rand(N)
    μ ./= sum(μ)
    ν ./= sum(ν)

    # create random cost matrix
    C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

    # compute optimal transport map and cost with POT
    pot_P = POT.emd(μ, ν, C)
    pot_cost = POT.emd2(μ, ν, C)

    # compute optimal transport map and cost with Tulip
    lp = Tulip.Optimizer()
    P = emd(μ, ν, C, lp)
    @test size(C) == size(P)
    @test MOI.get(lp, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test maximum(abs, P .- pot_P) < 1e-2

    lp = Tulip.Optimizer()
    cost = emd2(μ, ν, C, lp)
    @test dot(C, P) ≈ cost atol = 1e-5
    @test MOI.get(lp, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test cost ≈ pot_cost atol = 1e-5

    # ensure that provided map is used
    cost2 = emd2(similar(μ), similar(ν), C, lp; map=P)
    @test cost2 ≈ cost
end

@testset "1D Optimal Transport for Convex Cost" begin
    # Continuous Case
    μ = Normal(0,2)
    ν = Normal(10,2)
    c(x,y) = abs(x-y)

    @test otCost1d(c,μ,ν) ≈ 10 atol=1e-5

    # Discrete Case
    n,m = 100, 150

    μ = rand(n)
    ν = rand(m) .+ 0.5;
    μ_n = rand(n)
    ν_m = rand(m)
    μ_n = μ_n/sum(μ_n)
    ν_m = ν_m/sum(ν_m);

    c(x,y) = (x-y)^2
    C = Distances.pairwise(Distances.SqEuclidean(), μ', ν');

    lp = Tulip.Optimizer()
    cost = emd2(μ_n, ν_m, C, lp)
    cost1 = otCost1d(c,μ,μ_n,ν,ν_m)

    P = emd(μ_n, ν_m, C, lp)
    γ = otPlan1d(c,μ,μ_n,ν,ν_m)

    @test cost ≈ cost1 atol=1e-5
    @test sum(γ .>=0) == n*m
    @test dot(C,γ) ≈ dot(C,P) atol=1e-5

    μ = DiscreteNonParametric(μ, μ_n);
    ν = DiscreteNonParametric(ν, ν_m);

    u = μ.support
    u_n = μ.p
    v = ν.support
    v_m = ν.p

    γ1 = otPlan1d(c,u,u_n,v,v_m)
    γ2 = otPlan1d(c,μ,ν)
    cost2 = otCost1d(c,μ,ν)

    @test sum(γ2 .>=0) == n*m
    @test γ2 ≈ γ1 atol = 1e-5
    @test cost2 ≈ cost1

end


@testset "entropically regularized transport" begin
    M = 250
    N = 200

    @testset "example" begin
        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ = sinkhorn(μ, ν, C, eps)
        γ_pot = POT.sinkhorn(μ, ν, C, eps)
        @test norm(γ - γ_pot, Inf) < 1e-9

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps)
        c_pot = POT.sinkhorn2(μ, ν, C, eps)
        @test c ≈ c_pot atol = 1e-9

        # ensure that provided map is used
        c2 = sinkhorn2(similar(μ), similar(ν), C, rand(); map=γ)
        @test c2 ≈ c
    end

    # different element type
    @testset "Float32" begin
        # create two uniform histograms
        μ = fill(Float32(1 / M), M)
        ν = fill(Float32(1 / N), N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(Float32, 1, M), rand(Float32, 1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01f0
        γ = sinkhorn(μ, ν, C, eps)
        @test eltype(γ) === Float32

        γ_pot = POT.sinkhorn(μ, ν, C, eps)
        @test eltype(γ_pot) === Float64 # POT does not respect input type
        @test norm(γ - γ_pot, Inf) < Base.eps(Float32)

        # compute optimal transport cost (Julia implementation + POT)
        c = sinkhorn2(μ, ν, C, eps)
        @test c isa Float32

        c_pot = POT.sinkhorn2(μ, ν, C, eps)
        @test c_pot isa Float64 # POT does not respect input types
        @test c ≈ c_pot atol = Base.eps(Float32)
    end

    # computation on the GPU
    if CUDA.functional()
        @testset "CUDA" begin
            # create two uniform histograms
            μ = CUDA.fill(Float32(1 / M), M)
            ν = CUDA.fill(Float32(1 / N), N)

            # create random cost matrix
            C = abs2.(CUDA.rand(M) .- CUDA.rand(1, N))

            # compute optimal transport map
            eps = 0.01f0
            γ = sinkhorn(μ, ν, C, eps)
            @test γ isa CuArray{Float32}

            # compute optimal transport cost
            c = sinkhorn2(μ, ν, C, eps)
            @test c isa Float32
        end
    end
end

@testset "unbalanced transport" begin
    M = 250
    N = 200
    @testset "example" begin
        μ = fill(1 / N, M)
        ν = fill(1 / N, N)

        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        eps = 0.01
        lambda = 1
        γ = sinkhorn_unbalanced(μ, ν, C, lambda, lambda, eps)
        γ_pot = POT.sinkhorn_unbalanced(μ, ν, C, eps, lambda)

        # compute optimal transport map
        @test norm(γ - γ_pot, Inf) < 1e-9

        c = sinkhorn_unbalanced2(μ, ν, C, lambda, lambda, eps)
        c_pot = POT.sinkhorn_unbalanced2(μ, ν, C, eps, lambda)

        @test c ≈ c_pot atol = 1e-9

        # ensure that provided map is used
        c2 = sinkhorn_unbalanced2(similar(μ), similar(ν), C, rand(), rand(), rand(); map=γ)
        @test c2 ≈ c
    end
end

@testset "stabilized sinkhorn" begin
    M = 250
    N = 200

    @testset "example" begin
        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.01
        γ = sinkhorn_stabilized(μ, ν, C, eps)
        γ_pot = POT.sinkhorn(μ, ν, C, eps; method="sinkhorn_stabilized")
        @test norm(γ - γ_pot, Inf) < 1e-9
    end
end

@testset "quadratic optimal transport" begin
    M = 250
    N = 200
    @testset "example" begin
        # create two uniform histograms
        μ = fill(1 / M, M)
        ν = fill(1 / N, N)

        # create random cost matrix
        C = pairwise(SqEuclidean(), rand(1, M), rand(1, N); dims=2)

        # compute optimal transport map (Julia implementation + POT)
        eps = 0.25
        γ = quadreg(μ, ν, C, eps)
        γ_pot = sparse(POT.smooth_ot_dual(μ, ν, C, eps; max_iter=5000))
        # need to use a larger tolerance here because of a quirk with the POT solver 
        @test norm(γ - γ_pot, Inf) < 1e-4
    end
end

@testset "sinkhorn barycenter" begin
    @testset "example" begin
        # set up support
        support = range(-1, 1; length=250)
        μ1 = exp.(-(support .+ 0.5) .^ 2 ./ 0.1^2)
        μ1 ./= sum(μ1)
        μ2 = exp.(-(support .- 0.5) .^ 2 ./ 0.1^2)
        μ2 ./= sum(μ2)
        μ_all = hcat(μ1, μ2)'
        # create cost matrix
        C = pairwise(SqEuclidean(), support')
        # compute Sinkhorn barycenter (Julia implementation + POT)
        eps = 0.01
        μ_interp = sinkhorn_barycenter(μ_all, [C, C], eps, [0.5, 0.5])
        μ_interp_pot = POT.barycenter(μ_all, C, eps; weights=[0.5, 0.5])
        # need to use a larger tolerance here because of a quirk with the POT solver 
        @test norm(μ_interp - μ_interp_pot, Inf) < 1e-9
    end
end
