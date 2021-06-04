# OptimalTransport.jl -- optimal transportation algorithms for Julia
# See prettyprinted documentation at https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/
#

module OptimalTransport

using LinearAlgebra: AbstractMatrix
using Distances
using LinearAlgebra
using IterativeSolvers, SparseArrays
using LogExpFunctions: LogExpFunctions
using MathOptInterface
using Distributions
using QuadGK
using StatsBase: StatsBase

export sinkhorn, sinkhorn2
export emd, emd2
export sinkhorn_stabilized, sinkhorn_stabilized_epsscaling, sinkhorn_barycenter
export sinkhorn_unbalanced, sinkhorn_unbalanced2
export sinkhorn_divergence
export quadreg
export ot_cost, ot_plan, wasserstein, squared2wasserstein
export FiniteDiscreteMeasure
export cost_matrix

const MOI = MathOptInterface

include("utils.jl")
include("exact.jl")
include("wasserstein.jl")
include("entropic.jl")
include("quadratic.jl")

end
