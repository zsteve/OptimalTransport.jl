using OptimalTransport
using Pkg: Pkg
using SafeTestsets

using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "OptimalTransport" begin
    if GROUP == "All" || GROUP == "OptimalTransport"
        @safetestset "Exact OT" begin
            include("exact.jl")
        end
        @safetestset "Entropically regularized OT" begin
            include("entropic.jl")
        end
        @safetestset "Quadratically regularized OT" begin
            include("quadratic.jl")
        end
        @safetestset "Unbalanced OT" begin
            include("unbalanced.jl")
        end
        @safetestset "Wasserstein distance" begin
            include("wasserstein.jl")
        end
        @safetestset "Finite Discrete Measure" begin
            include("finitediscretemeasure.jl")
        end
    end

    # CUDA requires Julia >= 1.6
    if (GROUP == "All" || GROUP == "GPU") && VERSION >= v"1.6"
        # activate separate environment: CUDA can't be added to test/Project.toml since it
        # is not available on older Julia versions
        pkgdir = dirname(dirname(pathof(OptimalTransport)))
        Pkg.activate("gpu")
        Pkg.develop(Pkg.PackageSpec(; path=pkgdir))
        Pkg.instantiate()

        @safetestset "Simple GPU" begin
            include(joinpath("gpu/simple_gpu.jl"))
        end
    end
end
