#=
Update hyperparameters
=#

using Distributions
using LinearAlgebra
using EllipsisNotation
using TensorOperations
using Einsum

include("./type.jl")
include("./utils.jl")
include("./tn_tools.jl")
include("./tensor_basics.jl")


function computeγ(δ)
    γ = cumprod.(δ)
    return γ
end


# Hyper parameters
"""
Update latent factor η
isSample = true for GS;
isSample = false for EM;
"""
function updateη!(
    YEst :: TRLF{T},
    Y,
    τ;
    isSample = true) where {T <: Real}

    rank = YEst.rank
    sz = YEst.shape
    D = YEst.ndim
    k = rank[1] * rank[D]  # factor number

    Yvec = matricization(Y, D, false)

    W = computeCoef(YEst)
    W = reshape(W, size(W, 1), :, size(W, ndims(W)))
    W = matricization(W, 2, true)
    VInv = τ * W' * W + Matrix{T}(I, k, k)
    V = inv(VInv)
    μ = τ * V * W' * Yvec'  # TODO: This cannot be reduced

    η = zeros(T, rank[D], sz[D], rank[1])
    for i in 1:sz[D]
        if isSample
            ηi = safeMvNormal(μ[:, i], VInv)
        else
            ηi = μ[:, i]
        end
        η[:, i, :] = reshape(ηi, rank[D], rank[1])
    end
    YEst.coresQ[D] = η

    k1 = rank[D]
    k2 = rank[1]

    return reshape((V + V') / 2, k1, k2, k1, k2)
end


function updateη!(
    YEst :: TRLF{T},
    Y, mask, U,
    τ; isSample = true) where {T <: Real}
    rank = YEst.rank
    sz = YEst.shape
    D = YEst.ndim
    k = rank[1] * rank[D]  # factor number

    Yd = matricization(Y, D, true)
    Od = matricization(mask, D, true)
    Qd = zeros(rank[D], sz[D], rank[1])

    GNeq = subchain(YEst, D)
    GNeqVec = reshape(
        permutedims(GNeq, [2, 3, 1]), size(GNeq, 2), :
    )
    UKron = kron(Diagonal(U[1]), Diagonal(U[D]))
    VTot = Array{Array, 1}(undef, sz[D])
    k1 = rank[D]
    k2 = rank[1]
    for i in 1:sz[D]
        Γ = broadcast(*, vec(Yd[i, :]), GNeqVec)
        Γ = vec(sum(Γ, dims=1))

        EvG2 = GNeqVec[Od[i, :] .== 1.0, :]
        EvG2 = EvG2' * EvG2

        VInv = τ * EvG2 + I
        V = inv(VInv)
            
        μ = τ * V * Γ
        if isSample
            Qdi = safeMvNormal(μ, VInv)
        else
            Qdi = μ
        end
        Qdi = reshape(Qdi, rank[D], rank[1])
        Qd[:, i, :] = Qdi
        VTot[i] = (V + V') / 2
    end
    YEst.coresQ[D] = Qd

    return VTot
end


function updateδ!(
    δ, YEst, ϕ,
    U, a_δ1, a_δ2; isSample = true
    )
    sz = YEst.shape
    D = YEst.ndim
    # update δ1
    foo = dropdims(sum(ϕ[1] .* YEst.coresQ[1] .^ 2, dims = 2), dims = 2)
    foo = foo * vec(U[2])
    for h in 1:length(δ[1])
        a_δ_post = sz[1] * length(δ[2]) * (length(δ[1]) - h + 1) / 2
        a_δ_post += a_δ1

        b_δ_post = 1 + 0.5 * sum(foo[h:end] .* (U[1][h:end] / δ[1][h]))[1]
        if isSample
            δ[1][h] = rand(Gamma(a_δ_post, 1 / b_δ_post))
        else
            δ[1][h] = a_δ_post / b_δ_post
        end
    end
    # update δ2, ..., δD
    for d in 2:D-1
        dp, dn = round_index(d, D)
        foo = dropdims(sum(ϕ[d] .* YEst.coresQ[d] .^ 2, dims = 2), dims = 2)
        foo = foo * vec(U[dn])
        bar = dropdims(sum(ϕ[dp] .* YEst.coresQ[dp] .^ 2, dims = 2), dims =2)
        bar = vec(vec(U[dp])' * bar)
        foo += bar
        for h in 1:length(δ[d])
            a_δ_post = (sz[d] * length(δ[dn]) + sz[dp] * length(δ[d])) * (length(δ[d]) - h + 1) / 2
            a_δ_post += a_δ2

            b_δ_post = 1 + 0.5 * sum(foo[h:end] .* (U[d][h:end] / δ[d][h]))[1]
            if isSample
                δ[d][h] = rand(Gamma(a_δ_post, 1 / b_δ_post))
            else
                δ[d][h] = a_δ_post / b_δ_post
            end
        end
    end
    # update δD+1
    bar = dropdims(sum(ϕ[D-1] .* YEst.coresQ[D-1] .^ 2, dims = 2), dims =2)
    bar = vec(vec(U[D-1])' * bar)
    for h in 1:length(δ[D])
        a_δ_post = sz[D-1] * length(δ[D]) * (length(δ[D]) - h + 1) / 2
        a_δ_post += a_δ2

        b_δ_post = 1 + 0.5 * sum(bar[h:end] .* (U[D][h:end] / δ[D][h]))[1]
        if isSample
            δ[D][h] = rand(Gamma(a_δ_post, 1 / b_δ_post))
        else
            δ[D][h] = a_δ_post / b_δ_post
        end
    end
end


function updateϕ!(ϕ, YEst, U, ν; isSample = true)
    sz = YEst.shape
    D = YEst.ndim
    for d in 1:D-1
        _, dn = round_index(d, D)
        for i in 1:sz[d]
            G = YEst.coresQ[d][:, i, :] .^ 2
            a = 0.5 + ν
            b = ν .+ Diagonal(U[d]) * G * Diagonal(U[dn])
            if isSample
                ϕ[d][:, i, :] .= rand.(Gamma.(a, 1 ./ b))
            else
                ϕ[d][:, i, :] .= a ./ b
            end
        end
    end
end