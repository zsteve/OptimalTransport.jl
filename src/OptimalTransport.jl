# OptimalTransport.jl -- optimal transportation algorithms for Julia
# See prettyprinted documentation at https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/
#

module OptimalTransport

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
export quadreg
export ot_cost, ot_plan, wasserstein, squared2wasserstein

const MOI = MathOptInterface

include("exact.jl")
include("wasserstein.jl")

dot_matwise(x::AbstractMatrix, y::AbstractMatrix) = dot(x, y)
function dot_matwise(x::AbstractArray, y::AbstractMatrix)
    xmat = reshape(x, size(x, 1) * size(x, 2), :)
    return reshape(reshape(y, 1, :) * xmat, size(x)[3:end])
end

"""
    sinkhorn_gibbs(
        μ, ν, K; atol=0, rtol=atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000
    )

Compute the dual potentials for the entropically regularized optimal transport problem
with source and target marginals `μ` and `ν` and Gibbs kernel `K` using the Sinkhorn
algorithm.

The Gibbs kernel `K` is defined as
```math
K = \\exp(-C / \\varepsilon),
```
where ``C`` is the cost matrix and ``\\varepsilon`` the entropic regularization parameter.
The corresponding optimal transport plan can be computed from the dual potentials ``u``
and ``v`` as
```math
\\gamma = \\operatorname{diag}(u) K \\operatorname{diag}(v).
```
Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterate of the transport plan `G` satisfies
```julia
isapprox(sum(G; dims=2), μ; atol=atol, rtol=rtol, norm=x -> norm(x, 1))
```
The default `rtol` depends on the types of `μ`, `ν`, and `K`. After `maxiter` iterations,
the computation is stopped.

Note that for a common kernel `K`, multiple histograms may be provided for a batch computation by passing `μ` and `ν`
as matrices whose columns `μ[:, i]` and `ν[:, i]` correspond to pairs of histograms. 
The output are then matrices `u` and `v` such that `u[:, i]` and `v[:, i]` are the dual variables for `μ[:, i]` and `ν[:, i]`.

In addition, the case where one of `μ` or `ν` is a single histogram and the other a matrix of histograms is supported.
"""
function sinkhorn_gibbs(
    μ,
    ν,
    K;
    tol=nothing,
    atol=tol,
    rtol=nothing,
    check_marginal_step=nothing,
    check_convergence=check_marginal_step,
    maxiter::Int=1_000,
)
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_gibbs,
        )
    end
    if check_marginal_step !== nothing
        Base.depwarn(
            "keyword argument `check_marginal_step` is deprecated, please use `check_convergence`",
            :sinkhorn_gibbs,
        )
    end
    if (size(μ, 2) != size(ν, 2)) && (min(size(μ, 2), size(ν, 2)) > 1)
        throw(
            DimensionMismatch(
                "Error: number of columns in μ and ν must coincide, if both are matrix valued",
            ),
        )
    end
    all(sum(μ; dims=1) .≈ sum(ν; dims=1)) ||
        throw(ArgumentError("source and target marginals must have the same mass"))

    # set default values of tolerances
    T = float(Base.promote_eltype(μ, ν, K))
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # initial iteration
    u = if isequal(size(μ, 2), size(ν, 2))
        similar(μ)
    else
        repeat(similar(μ[:, 1]); outer=(1, max(size(μ, 2), size(ν, 2))))
    end
    u .= μ ./ vec(sum(K; dims=2))
    v = ν ./ (K' * u)
    tmp1 = K * v
    tmp2 = similar(u)

    norm_μ = sum(abs, μ; dims=1) # for convergence check
    isconverged = false
    check_step = check_convergence === nothing ? 10 : check_convergence
    for iter in 0:maxiter
        if iter % check_step == 0
            # check source marginal
            # do not overwrite `tmp1` but reuse it for computing `u` if not converged
            @. tmp2 = u * tmp1
            norm_uKv = sum(abs, tmp2; dims=1)
            @. tmp2 = μ - tmp2
            norm_diff = sum(abs, tmp2; dims=1)

            @debug "Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": absolute error of source marginal = " *
                   string(maximum(norm_diff))

            # check stopping criterion
            if all(@. norm_diff < max(_atol, _rtol * max(norm_μ, norm_uKv)))
                @debug "Sinkhorn algorithm ($iter/$maxiter): converged"
                isconverged = true
                break
            end
        end

        # perform next iteration
        if iter < maxiter
            @. u = μ / tmp1
            mul!(v, K', u)
            @. v = ν / v
            mul!(tmp1, K, v)
        end
    end

    if !isconverged
        @warn "Sinkhorn algorithm ($maxiter/$maxiter): not converged"
    end

    return u, v
end

function add_singleton(x::AbstractArray, ::Val{dim}) where {dim}
    shape = ntuple(ndims(x) + 1) do i
        return i < dim ? size(x, i) : (i > dim ? size(x, i - 1) : 1)
    end
    return reshape(x, shape)
end

"""
    sinkhorn(
        μ, ν, C, ε; atol=0, rtol=atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000
    )

Compute the optimal transport plan for the entropically regularized optimal transport
problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularization parameter `ε`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term.

Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterate of the transport plan `G` satisfies
```julia
isapprox(sum(G; dims=2), μ; atol=atol, rtol=rtol, norm=x -> norm(x, 1))
```
The default `rtol` depends on the types of `μ`, `ν`, and `C`. After `maxiter` iterations,
the computation is stopped.

Note that for a common cost `C`, multiple histograms may be provided for a batch computation by passing `μ` and `ν`
as matrices whose columns `μ[:, i]` and `ν[:, i]` correspond to pairs of histograms. 

The output in this case is an `Array` `γ` of coupling matrices such that `γ[:, :, i]` is a coupling of `μ[:, i]` and `ν[:, i]`.

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn(μ, ν, C, ε; kwargs...)
    # compute Gibbs kernel
    K = @. exp(-C / ε)

    # compute dual potentials
    u, v = sinkhorn_gibbs(μ, ν, K; kwargs...)
    return K .* add_singleton(u, Val(2)) .* add_singleton(v, Val(1))
end

"""
    sinkhorn2(μ, ν, C, ε; regularization=false, plan=nothing, kwargs...)

Solve the entropically regularized optimal transport problem with source and target
marginals `μ` and `ν`, cost matrix `C` of size `(length(μ), length(ν))`, and entropic
regularization parameter `ε`, and return the optimal cost.

A pre-computed optimal transport `plan` may be provided. The other keyword arguments
supported here are the same as those in the [`sinkhorn`](@ref) function.

!!! note
    As the `sinkhorn2` function in the Python Optimal Transport package, this function
    returns the optimal transport cost without the regularization term. The cost
    with the regularization term can be computed by setting `regularization=true`.

See also: [`sinkhorn`](@ref)
"""
function sinkhorn2(μ, ν, C, ε; regularization=false, plan=nothing, kwargs...)
    γ = if plan === nothing
        sinkhorn(μ, ν, C, ε; kwargs...)
    else
        # check dimensions
        size(C) == (size(μ, 1), size(ν, 1)) || error(
            "cost matrix `C` must be of size `(size(μ, dims = 1), size(ν, dims = 1))`",
        )
        (size(plan, 1), size(plan, 2)) == size(C) || error(
            "optimal transport plan `plan` and cost matrix `C` must be of the same size",
        )
        plan
    end
    cost = if regularization
        dot_matwise(γ, C) .+
        ε * reshape(sum(LogExpFunctions.xlogx, γ; dims=(1, 2)), size(γ)[3:end])
    else
        dot_matwise(γ, C)
    end

    return cost
end

"""
    sinkhorn_unbalanced(μ, ν, C, λ1::Real, λ2::Real, ε; kwargs...)

Compute the optimal transport plan for the unbalanced entropically regularized optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
terms `λ1` and `λ2`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ \\lambda_1 \\operatorname{KL}(\\gamma 1 | \\mu)
+ \\lambda_2 \\operatorname{KL}(\\gamma^{\\mathsf{T}} 1 | \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``\\operatorname{KL}`` is the Kullback-Leibler divergence.

The keyword arguments supported here are the same as those in the `sinkhorn_unbalanced`
for unbalanced optimal transport problems with general soft marginal constraints.
"""
function sinkhorn_unbalanced(
    μ, ν, C, λ1::Real, λ2::Real, ε; proxdiv_F1=nothing, proxdiv_F2=nothing, kwargs...
)
    if proxdiv_F1 !== nothing && proxdiv_F2 !== nothing
        Base.depwarn(
            "keyword arguments `proxdiv_F1` and `proxdiv_F2` are deprecated",
            :sinkhorn_unbalanced,
        )

        # have to wrap the "proxdiv" functions since the signature changed
        # ε was fixed in the function, so we ignore it
        proxdiv_F1_wrapper(s, p, _) = copyto!(s, proxdiv_F1(s, p))
        proxdiv_F2_wrapper(s, p, _) = copyto!(s, proxdiv_F2(s, p))

        return sinkhorn_unbalanced(
            μ, ν, C, proxdiv_F1_wrapper, proxdiv_F2_wrapper, ε; kwargs...
        )
    end

    # define "proxdiv" functions for the unbalanced OT problem
    proxdivF!(s, p, ε, λ) = (s .= (p ./ s) .^ (λ / (ε + λ)))
    proxdivF1!(s, p, ε) = proxdivF!(s, p, ε, λ1)
    proxdivF2!(s, p, ε) = proxdivF!(s, p, ε, λ2)

    return sinkhorn_unbalanced(μ, ν, C, proxdivF1!, proxdivF2!, ε; kwargs...)
end

"""
    sinkhorn_unbalanced(
        μ, ν, C, proxdivF1!, proxdivF2!, ε;
        atol=0, rtol=atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000,
    )

Compute the optimal transport plan for the unbalanced entropically regularized optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and soft marginal
constraints ``F_1`` and ``F_2`` with "proxdiv" functions `proxdivF!` and `proxdivG!`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ F_1(\\gamma 1, \\mu)
+ F_2(\\gamma^{\\mathsf{T}} 1, \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``F_1(\\cdot, \\mu)`` and ``F_2(\\cdot, \\nu)`` are soft marginal
constraints for the source and target marginals.

The functions `proxdivF1!(s, p, ε)` and `proxdivF2!(s, p, ε)` evaluate the "proxdiv"
functions of ``F_1(\\cdot, p)`` and ``F_2(\\cdot, p)`` at ``s`` for the entropic
regularization parameter ``\\varepsilon``. They have to be mutating and overwrite the first
argument `s` with the result of their computations.

Mathematically, the "proxdiv" functions are defined as
```math
\\operatorname{proxdiv}_{F_i}(s, p, \\varepsilon)
= \\operatorname{prox}^{\\operatorname{KL}}_{F_i(\\cdot, p)/\\varepsilon}(s) \\oslash s
```
where ``\\oslash`` denotes element-wise division and
``\\operatorname{prox}_{F_i(\\cdot, p)/\\varepsilon}^{\\operatorname{KL}}`` is the proximal
operator of ``F_i(\\cdot, p)/\\varepsilon`` for the Kullback-Leibler
(``\\operatorname{KL}``) divergence.  It is defined as
```math
\\operatorname{prox}_{F}^{\\operatorname{KL}}(x)
= \\operatorname{argmin}_{y} F(y) + \\operatorname{KL}(y|x)
```
and can be computed in closed-form for specific choices of ``F``. For instance, if
``F(\\cdot, p) = \\lambda \\operatorname{KL}(\\cdot | p)`` (``\\lambda > 0``), then
```math
\\operatorname{prox}_{F(\\cdot, p)/\\varepsilon}^{\\operatorname{KL}}(x)
= x^{\\frac{\\varepsilon}{\\varepsilon + \\lambda}} p^{\\frac{\\lambda}{\\varepsilon + \\lambda}},
```
where all operators are acting pointwise.[^CPSV18]

Every `check_convergence` steps it is assessed if the algorithm is converged by checking if
the iterates of the scaling factor in the current and previous iteration satisfy
`isapprox(vcat(a, b), vcat(aprev, bprev); atol=atol, rtol=rtol)` where `a` and `b` are the
current iterates and `aprev` and `bprev` the previous ones. The default `rtol` depends on
the types of `μ`, `ν`, and `C`. After `maxiter` iterations, the computation is stopped.

[^CPSV18]: Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F.-X. (2018). [Scaling algorithms for unbalanced optimal transport problems](https://doi.org/10.1090/mcom/3303). Mathematics of Computation, 87(314), 2563–2609.

See also: [`sinkhorn_unbalanced2`](@ref)
"""
function sinkhorn_unbalanced(
    μ,
    ν,
    C,
    proxdivF1!,
    proxdivF2!,
    ε;
    tol=nothing,
    atol=tol,
    rtol=nothing,
    max_iter=nothing,
    maxiter=max_iter,
    check_convergence::Int=10,
)
    # deprecations
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_unbalanced,
        )
    end
    if max_iter !== nothing
        Base.depwarn(
            "keyword argument `max_iter` is deprecated, please use `maxiter`",
            :sinkhorn_unbalanced,
        )
    end

    # compute Gibbs kernel
    K = @. exp(-C / ε)

    # set default values of squared tolerances
    T = float(Base.promote_eltype(μ, ν, K))
    sqatol = atol === nothing ? 0 : atol^2
    sqrtol = rtol === nothing ? (sqatol > zero(sqatol) ? zero(T) : eps(T)) : rtol^2

    # initialize iterates
    a = similar(μ, T)
    sum!(a, K)
    proxdivF1!(a, μ, ε)
    b = similar(ν, T)
    mul!(b, K', a)
    proxdivF2!(b, ν, ε)

    # caches for convergence checks
    a_old = similar(a)
    b_old = similar(b)

    isconverged = false
    _maxiter = maxiter === nothing ? 1_000 : maxiter
    for iter in 1:_maxiter
        # update cache if necessary
        ischeck = iter % check_convergence == 0
        if ischeck
            copyto!(a_old, a)
            copyto!(b_old, b)
        end

        # compute next iterates
        mul!(a, K, b)
        proxdivF1!(a, μ, ε)
        mul!(b, K', a)
        proxdivF2!(b, ν, ε)

        # check convergence of the scaling factors
        if ischeck
            # compute norm of current and previous scaling factors and their difference
            sqnorm_a_b = sum(abs2, a) + sum(abs2, b)
            sqnorm_a_b_old = sum(abs2, a_old) + sum(abs2, b_old)
            a_old .-= a
            b_old .-= b
            sqeuclidean_a_b = sum(abs2, a_old) + sum(abs2, b_old)
            @debug "Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(_maxiter) *
                   ": squared Euclidean distance of iterates = " *
                   string(sqeuclidean_a_b)

            # check convergence of `a`
            if sqeuclidean_a_b < max(sqatol, sqrtol * max(sqnorm_a_b, sqnorm_a_b_old))
                @debug "Sinkhorn algorithm ($iter/$_maxiter): converged"
                isconverged = true
                break
            end
        end
    end

    if !isconverged
        @warn "Sinkhorn algorithm ($_maxiter/$_maxiter): not converged"
    end

    return K .* a .* b'
end

"""
    sinkhorn_unbalanced2(μ, ν, C, λ1, λ2, ε; plan=nothing, kwargs...)
    sinkhorn_unbalanced2(μ, ν, C, proxdivF1!, proxdivF2!, ε; plan=nothing, kwargs...)

Compute the optimal transport plan for the unbalanced entropically regularized optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
terms `λ1` and `λ2` or soft marginal constraints with "proxdiv" functions `proxdivF1!` and
`proxdivF2!`.

A pre-computed optimal transport `plan` may be provided. The other keyword arguments
supported here are the same as those in the [`sinkhorn_unbalanced`](@ref) for unbalanced
optimal transport problems with general soft marginal constraints.

See also: [`sinkhorn_unbalanced`](@ref)
"""
function sinkhorn_unbalanced2(
    μ, ν, C, λ1_or_proxdivF1, λ2_or_proxdivF2, ε; plan=nothing, kwargs...
)
    γ = if plan === nothing
        sinkhorn_unbalanced(μ, ν, C, λ1_or_proxdivF1, λ2_or_proxdivF2, ε; kwargs...)
    else
        # check dimensions
        size(C) == (length(μ), length(ν)) ||
            error("cost matrix `C` must be of size `(length(μ), length(ν))`")
        size(plan) == size(C) || error(
            "optimal transport plan `plan` and cost matrix `C` must be of the same size",
        )
        plan
    end
    return dot(γ, C)
end

"""
    sinkhorn_stabilized_epsscaling(μ, ν, C, ε; lambda = 0.5, k = 5, kwargs...)

Compute the optimal transport plan for the entropically regularized optimal transport problem 
with source and target marginals `μ` and `ν`, cost matrix `C` of size `(length(μ), length(ν))`, and entropic regularisation parameter `ε`. Employs the log-domain stabilized algorithm of Schmitzer et al. [^S19] with ε-scaling. 

`k` ε-scaling steps are used with scaling factor `lambda`, i.e. sequentially solve Sinkhorn using `sinkhorn_stabilized` with regularisation parameters  
``ε_i \\in [λ^{1-k}, \\ldots, λ^{-1}, 1] \\times ε``.

See also: [`sinkhorn_stabilized`](@ref), [`sinkhorn`](@ref)
"""
function sinkhorn_stabilized_epsscaling(μ, ν, C, ε; lambda=0.5, k=5, kwargs...)
    α = zero(μ)
    β = zero(ν)
    for ε_i in (ε * lambda^(1 - j) for j in k:-1:1)
        @debug "Epsilon-scaling Sinkhorn algorithm: ε = $ε_i"
        α, β = sinkhorn_stabilized(
            μ, ν, C, ε_i; alpha=α, beta=β, return_duals=true, kwargs...
        )
    end
    gamma = similar(C)
    getK!(gamma, C, α, β, ε, μ, ν)
    return gamma
end

function getK!(K, C, α, β, ε, μ, ν)
    @. K = exp(-(C - α - β') / ε) * μ * ν'
    return K
end

"""
    sinkhorn_stabilized(μ, ν, C, ε; absorb_tol = 1e3, alpha_0 = zero(μ), beta = zero(ν), maxiter = 1_000, atol = tol, rtol=nothing, return_duals = false)

Compute the optimal transport plan for the entropically regularized optimal transport problem 
with source and target marginals `μ` and `ν`, cost matrix `C` of size `(length(μ), length(ν))`, and entropic regularisation parameter `ε`. Employs the log-domain stabilized algorithm of Schmitzer et al. [^S19] 

`alpha` and `beta` are initial scalings for the stabilized Gibbs kernel. If not specified, `alpha` and `beta` are initialised to zero. 

If `return_duals = true`, then the optimal dual variables `(u, v)` corresponding to `(μ, ν)` are returned. Otherwise, the coupling `γ` is returned. 

[^S19]: Schmitzer, B., 2019. Stabilized sparse scaling algorithms for entropy regularized transport problems. SIAM Journal on Scientific Computing, 41(3), pp.A1443-A1481.

See also: [`sinkhorn`](@ref)
"""
function sinkhorn_stabilized(
    μ,
    ν,
    C,
    ε;
    absorb_tol=1e3,
    maxiter=1_000,
    tol=nothing,
    atol=tol,
    rtol=nothing,
    check_convergence=10,
    alpha=zero(μ),
    beta=zero(ν),
    return_duals=false,
)
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_stabilized,
        )
    end
    sum(μ) ≈ sum(ν) ||
        throw(ArgumentError("source and target marginals must have the same mass"))

    T = float(Base.promote_eltype(μ, ν, C))
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    norm_μ = sum(abs, μ)
    isconverged = false

    K = similar(C)
    gamma = similar(C)

    getK!(K, C, alpha, beta, ε, μ, ν)
    u = μ ./ sum(K; dims=2)
    v = ν ./ (K' * u)
    tmp_u = similar(u)
    for iter in 0:maxiter
        if (max(norm(u, Inf), norm(v, Inf)) > absorb_tol)
            @debug "Absorbing (u, v) into (alpha, beta)"
            # absorb into α, β
            alpha += ε * log.(u)
            beta += ε * log.(v)
            u .= 1
            v .= 1
            getK!(K, C, alpha, beta, ε, μ, ν)
        end
        if iter % check_convergence == 0
            # check marginal
            getK!(gamma, C, alpha, beta, ε, μ, ν)
            @. gamma *= u * v'
            norm_diff = sum(abs, gamma * ones(size(ν)) - μ)
            norm_uKv = sum(abs, gamma)
            @debug "Stabilized Sinkhorn algorithm (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": error of source marginal = " *
                   string(norm_diff)

            if norm_diff < max(_atol, _rtol * max(norm_μ, norm_uKv))
                @debug "Stabilized Sinkhorn algorithm ($iter/$maxiter): converged"
                isconverged = true
                break
            end
        end
        mul!(tmp_u, K, v)
        u = μ ./ tmp_u
        mul!(v, K', u)
        v = ν ./ v
    end

    if !isconverged
        @warn "Stabilized Sinkhorn algorithm ($maxiter/$maxiter): not converged"
    end

    alpha = alpha + ε * log.(u)
    beta = beta + ε * log.(v)
    if return_duals
        return alpha, beta
    end
    getK!(gamma, C, alpha, beta, ε, μ, ν)
    return gamma
end

"""
    sinkhorn_barycenter(μ, C, ε, w; atol = 0, rtol = atol > 0 ? 0 : √eps, check_convergence=10, maxiter=1_000)

Compute the Sinkhorn barycenter for a collection of `N` histograms contained in the columns of `μ`, for a cost matrix `C` of size `(size(μ, 1), size(μ, 1))`, relative weights `w` of size `N`, and entropic regularisation parameter `ε`. 
Returns the entropically regularised barycenter of the `μ`, i.e. the histogram `ρ` of length `size(μ, 1)` that solves 

```math
\\min_{\\rho \\in \\Sigma} \\sum_{i = 1}^N w_i \\operatorname{OT}_{\\varepsilon}(\\mu_i, \\rho)
```

where ``\\operatorname{OT}_{ε}(\\mu, \\nu) = \\inf_{\\gamma \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle + \\varepsilon \\Omega(\\gamma)`` 
is the entropic optimal transport loss with cost ``C`` and regularisation ``\\epsilon``.
"""
function sinkhorn_barycenter(
    μ,
    C,
    ε,
    w;
    tol=nothing,
    atol=tol,
    rtol=nothing,
    check_marginal_step=nothing,
    check_convergence=10,
    max_iter=nothing,
    maxiter=1_000,
)
    if tol !== nothing
        Base.depwarn(
            "keyword argument `tol` is deprecated, please use `atol` and `rtol`",
            :sinkhorn_barycenter,
        )
    end
    if check_marginal_step !== nothing
        Base.depwarn(
            "keyword argument `check_marginal_step` is deprecated, please use `check_convergence`",
            :sinkhorn_barycenter,
        )
    end
    if max_iter !== nothing
        Base.depwarn(
            "keyword argument `max_iter` is deprecated, please use `maxiter`",
            :sinkhorn_barycenter,
        )
    end
    if !isapprox(extrema(sum(μ; dims=1))...)
        throw(ArgumentError("Error: input marginals must have the same mass"))
    end

    T = float(Base.promote_eltype(μ, C))
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    K = exp.(-C / ε)
    v = similar(μ, T)
    u = similar(μ, T)
    fill!(u, one(T))
    a = similar(u, T, size(u, 1), 1)
    tmp = similar(u)
    N = size(μ, 2)
    # norm of inputs for convergence check
    norm_μ = sum(abs, μ; dims=1)
    isconverged = false
    for iter in 1:maxiter
        mul!(tmp, K', u)
        @. v = μ / tmp
        mul!(tmp, K, v)
        a .= prod(tmp' .^ w; dims=1)'
        @. u = a / tmp
        if iter % check_convergence == 0
            # check marginal errors
            mul!(tmp, K', u)
            norm_diff = sum(abs, @. μ - v * tmp; dims=1)
            mul!(tmp, K, v)
            norm_uKv = sum(abs, @. u * tmp; dims=1)
            @debug "Sinkhorn barycenter algorithm (" *
                   string(iter) *
                   "/" *
                   string(maxiter) *
                   ": absolute error of source marginal = " *
                   string(maximum(norm_diff))
            if all(@. norm_diff < max(_atol, _rtol * max(norm_μ, norm_uKv)))
                @debug "Sinkhorn barycenter algorithm ($iter/$maxiter): converged"
                isconverged = true
                break
            end
        end
    end
    if !isconverged
        @warn "Sinkhorn barycenter algorithm ($maxiter/$maxiter): not converged"
    end
    return u[:, 1] .* (K * v[:, 1])
end

"""
    quadreg(mu, nu, C, ϵ; θ = 0.1, tol = 1e-5,maxiter = 50,κ = 0.5,δ = 1e-5)

Computes the optimal transport plan of histograms `mu` and `nu` with cost matrix `C` and quadratic regularization parameter `ϵ`, 
using the semismooth Newton algorithm [Lorenz 2016].

This implementation makes use of IterativeSolvers.jl and SparseArrays.jl.

Parameters:\n
θ: starting Armijo parameter.\n
tol: tolerance of marginal error.\n
maxiter: maximum interation number.\n
κ: control parameter of Armijo.\n
δ: small constant for the numerical stability of conjugate gradient iterative solver.\n

Tips:
If the algorithm does not converge, try some different values of θ.

Reference:
Lorenz, D.A., Manns, P. and Meyer, C., 2019. Quadratically regularized optimal transport. arXiv preprint arXiv:1903.01112v4.
"""
function quadreg(mu, nu, C, ϵ; θ=0.1, tol=1e-5, maxiter=50, κ=0.5, δ=1e-5)
    if !(sum(mu) ≈ sum(nu))
        throw(ArgumentError("Error: mu and nu must lie in the simplex"))
    end

    N = length(mu)
    M = length(nu)

    # initialize dual potentials as uniforms
    a = ones(M) ./ M
    b = ones(N) ./ N
    γ = spzeros(M, N)

    da = spzeros(M)
    db = spzeros(N)

    converged = false

    function DualObjective(a, b)
        A = a .* ones(N)' + ones(M) .* b' - C'

        return 0.5 * norm(A[A .> 0], 2)^2 - ϵ * (dot(nu, a) + dot(mu, b))
    end

    # computes minimizing directions, update γ
    function search_dir!(a, b, da, db)
        P = a * ones(N)' .+ ones(M) * b' .- C'

        σ = 1.0 * sparse(P .>= 0)
        γ = sparse(max.(P, 0) ./ ϵ)

        G = vcat(
            hcat(spdiagm(0 => σ * ones(N)), σ),
            hcat(sparse(σ'), spdiagm(0 => sparse(σ') * ones(M))),
        )

        h = vcat(γ * ones(N) - nu, sparse(γ') * ones(M) - mu)

        x = cg(G + δ * I, -ϵ * h)

        da = x[1:M]
        return db = x[(M + 1):end]
    end

    function search_dir(a, b)
        P = a * ones(N)' .+ ones(M) * b' .- C'

        σ = 1.0 * sparse(P .>= 0)
        γ = sparse(max.(P, 0) ./ ϵ)

        G = vcat(
            hcat(spdiagm(0 => σ * ones(N)), σ),
            hcat(sparse(σ'), spdiagm(0 => sparse(σ') * ones(M))),
        )

        h = vcat(γ * ones(N) - nu, sparse(γ') * ones(M) - mu)

        x = cg(G + δ * I, -ϵ * h)

        return x[1:M], x[(M + 1):end]
    end

    # computes optimal maginitude in the minimizing directions
    function search_t(a, b, da, db, θ)
        d = ϵ * dot(γ, (da .* ones(N)' .+ ones(M) .* db')) - ϵ * (dot(da, nu) + dot(db, mu))

        ϕ₀ = DualObjective(a, b)
        t = 1

        while DualObjective(a + t * da, b + t * db) >= ϕ₀ + t * θ * d
            t *= κ

            if t < 1e-15
                # @warn "@ i = $i, t = $t , armijo did not converge"
                break
            end
        end
        return t
    end

    for i in 1:maxiter

        # search_dir!(a, b, da, db)
        da, db = search_dir(a, b)

        t = search_t(a, b, da, db, θ)

        a += t * da
        b += t * db

        err1 = norm(γ * ones(N) - nu, Inf)
        err2 = norm(sparse(γ') * ones(M) - mu, Inf)

        if err1 <= tol && err2 <= tol
            converged = true
            @warn "Converged @ i = $i with marginal errors: \n err1 = $err1, err2 = $err2 \n"
            break
        elseif i == maxiter
            @warn "Not Converged with errors:\n err1 = $err1, err2 = $err2 \n"
        end

        @debug " t = $t"
        @debug "marginal @ i = $i: err1 = $err1, err2 = $err2 "
    end

    if !converged
        @warn "SemiSmooth Newton algorithm did not converge"
    end

    return sparse(γ')
end

end
