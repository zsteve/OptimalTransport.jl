"""
    sinkhorn_divergence(
        c,
        μ::Union{FiniteDiscreteMeasure, DiscreteNonParametric},
        ν::Union{FiniteDiscreteMeasure, DiscreteNonParametric},
        ε,
        alg::Sinkhorn
        ; plan=nothing, kwargs...
    )


Compute the Sinkhorn Divergence between finite discrete
measures `μ` and `ν` with respect to a cost function `c`,
entropic regularization parameter `ε` and algorithm `alg`.
The default algorithm is the `SinkhornGibbs`.

A pre-computed optimal transport `plan` between `μ` and `ν` may be provided.

The Sinkhorn Divergence is computed as:
```math
\\operatorname{S}_{c,ε}(μ,ν) := \\operatorname{OT}_{c,ε}(μ,ν)
- \\frac{1}{2}(\\operatorname{OT}_{c,ε}(μ,μ) + \\operatorname{OT}_{c,ε}(ν,ν)),
```
where ``\\operatorname{OT}_{c,ε}(μ,ν)``, ``\\operatorname{OT}_{c,ε}(μ,μ)`` and
``\\operatorname{OT}_{c,ε}(ν,ν)`` are the entropically regularized optimal transport cost
between `(μ,ν)`, `(μ,μ)` and `(ν,ν)`, respectively.

The formulation for the Sinkhorn Divergence may have slight variations depending on the paper consulted.
The Sinkhorn Divergence was initially proposed by [^GPC18], although, this package uses the formulation given by
[^FeydyP19], which is also the one used on the Python Optimal Transport package.

[^GPC18]: Aude Genevay, Gabriel Peyré, Marco Cuturi, Learning Generative Models with Sinkhorn Divergences,
Proceedings of the Twenty-First International Conference on Artficial Intelligence and Statistics, (AISTATS) 21, 2018

[^FeydyP19]: Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi
Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between op-
timal transport and mmd using sinkhorn divergences. In The 22nd In-
ternational Conference on Artificial Intelligence and Statistics, pages
2681–2690. PMLR, 2019.

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn_divergence(
    c,
    μ::T,
    ν::T,
    ε,
    alg::Sinkhorn=SinkhornGibbs();
    regularization=nothing,
    plan=nothing,
    kwargs...,
) where {T<:Union{FiniteDiscreteMeasure,DiscreteNonParametric}}
    return sinkhorn_divergence(
        pairwise(c, μ.support, ν.support),
        pairwise(c, μ.support),
        pairwise(c, ν.support),
        μ,
        ν,
        ε;
        regularization=regularization,
        kwargs...,
    )
end

"""
    sinkhorn_divergence(
        Cμν, Cμμ, Cνν,
        μ::Union{FiniteDiscreteMeasure, DiscreteNonParametric},
        ν::Union{FiniteDiscreteMeasure, DiscreteNonParametric},
        ε,
        alg::Sinkhorn; plan=nothing, kwargs...
    )

Compute the Sinkhorn Divergence between finite discrete
measures `μ` and `ν` with respect to the precomputed cost matrices `Cμν`,
`Cμμ` and `Cνν`, entropic regularization parameter `ε` and algorithm `alg`.
The default algorithm is the `SinkhornGibbs`.

A pre-computed optimal transport `plan` between `μ` and `ν` may be provided.

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn_divergence(
    Cμν,
    Cμ,
    Cν,
    μ::T,
    ν::T,
    ε,
    alg::Sinkhorn=SinkhornGibbs();
    regularization=nothing,
    plan=nothing,
    kwargs...,
) where {T<:Union{FiniteDiscreteMeasure,DiscreteNonParametric}}
    if regularization !== nothing
        @warn "`sinkhorn_divergence` does not support the `regularization` keyword argument"
    end

    OTμν = sinkhorn2(μ.p, ν.p, Cμν, ε, alg; plan=plan, regularization=false, kwargs...)
    OTμ = sinkhorn2(μ.p, μ.p, Cμ, ε, alg; plan=nothing, regularization=false, kwargs...)
    OTν = sinkhorn2(ν.p, ν.p, Cν, ε, alg; plan=nothing, regularization=false, kwargs...)
    return max(0, OTμν - (OTμ + OTν) / 2)
end
