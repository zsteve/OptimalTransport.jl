using OptimalTransport
using Pkg: Pkg
using SafeTestsets

using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "OptimalTransport" begin
    if GROUP == "All" || GROUP == "OptimalTransport"
        @safetestset "Utilities" begin
            include("utils.jl")
        end

        @safetestset "Exact OT" begin
            include("exact.jl")
        end

        @testset "Entropically regularized OT" begin
            @safetestset "Sinkhorn Gibbs" begin
                include(joinpath("entropic", "sinkhorn_gibbs.jl"))
            end
            @safetestset "Stabilized Sinkhorn" begin
                include(joinpath("entropic", "sinkhorn_stabilized.jl"))
            end
            @safetestset "Sinkhorn with ε-scaling" begin
                include(joinpath("entropic", "sinkhorn_epsscaling.jl"))
            end
            @safetestset "Unbalanced Sinkhorn" begin
                include(joinpath("entropic", "sinkhorn_unbalanced.jl"))
            end
            @safetestset "Sinkhorn barycenter" begin
                include(joinpath("entropic", "sinkhorn_barycenter.jl"))
            end
            @safetestset "Sinkhorn Divergence" begin
                include(joinpath("entropic", "sinkhorn_divergence.jl"))
            end
        end

        @safetestset "Quadratically regularized OT" begin
            include("quadratic.jl")
        end

        @safetestset "Wasserstein distance" begin
            include("wasserstein.jl")
        end

        @safetestset "Bures distance" begin
            include("bures.jl")
        end
    end

    # CUDA requires Julia >= 1.6
    if (GROUP == "All" || GROUP == "GPU") && VERSION >= v"1.6"
        # activate separate environment: CUDA can't be added to test/Project.toml since it
        # is not available on older Julia versions
        Pkg.activate("gpu")
        Pkg.instantiate()

        @safetestset "Simple GPU" begin
            include(joinpath("gpu/simple_gpu.jl"))
        end
    end
end
