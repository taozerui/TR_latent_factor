using Distributions
using LinearAlgebra
using Images
using Random

using Plots

import Base: circshift

include("./tn_tools.jl")
include("./tn_latent.jl")
include("./tensor_basics.jl")


"""
Log of joint distribution p(Y, Θ).
"""
function logJointDist(
    Y, YFull, YEst, ϕ,
    U, δ, τ, hyperParam
    )

    nObs = prod(YEst.shape)
    nDim = YEst.ndim
    sz = YEst.shape

    # log likelihood
    err = Y - YFull
    prob = 0.5 * nObs * sum(log.(τ)) - 0.5 * sum(τ) * norm(err) ^ 2

    # prior
    for d in 1:nDim-1
        _, dn = round_index(d, nDim)
        # prior of Q
        rd = length(U[d])
        rn = length(U[dn])
        prob += 0.5 * sum(log.(ϕ[d]))
        prob += 0.5 * sz[d] * rd * sum(log.(U[d])) + 0.5 * sz[d] * rn * sum(log.(U[dn]))
        for i in 1:sz[d]
            prob -= 0.5 * tr(
                Diagonal(U[dn]) * transpose(YEst.coresQ[d][:, i, :])
                * Diagonal(U[d]) * (ϕ[d][:, i, :] .* YEst.coresQ[d][:, i, :])
            )
        end
        # prior of ϕ
        prob += (hyperParam["ν"] - 1) * sum(log.(ϕ[d])) - hyperParam["ν"] * sum(ϕ[d])
    end

    for d in 1:nDim
        # prior of δ
        if d == 1
            prob += (hyperParam["a_δ1"] - 1) * sum(log.(δ[d])) - sum(δ[d])
        else
            prob += (hyperParam["a_δ2"] - 1) * sum(log.(δ[d])) - sum(δ[d])
        end
        # prior of η
        for i in 1:sz[d]
            prob -= 0.5 * sum(YEst.coresQ[nDim][:, i, :] .^ 2)
        end
    end
    # noise part
    prob += (hyperParam["a_τ"] - 1) * sum(log.(τ)) - hyperParam["b_τ"] * sum(τ)
    return prob
end


function logJointDist(
    Y, YFull, YEst, mask, ϕ,
    U, δ, τ, hyperParam
    )

    nObs = prod(YEst.shape)
    nDim = YEst.ndim
    sz = YEst.shape

    # log likelihood
    err = (Y - YFull) .* mask
    prob = 0.5 * sum(mask) * log(τ) - 0.5 * τ * norm(err) ^ 2

    # prior
    for d in 1:nDim-1
        _, dn = round_index(d, nDim)
        # prior of Q
        rd = length(U[d])
        rn = length(U[dn])
        prob += 0.5 * sum(log.(ϕ[d]))
        prob += 0.5 * sz[d] * rd * sum(log.(U[d])) + 0.5 * sz[d] * rn * sum(log.(U[dn]))
        for i in 1:sz[d]
            prob -= 0.5 * tr(
                Diagonal(U[dn]) * transpose(YEst.coresQ[d][:, i, :])
                * Diagonal(U[d]) * (ϕ[d][:, i, :] .* YEst.coresQ[d][:, i, :])
            )
        end
        # prior of ϕ
        prob += (hyperParam["ν"] - 1) * sum(log.(ϕ[d])) - hyperParam["ν"] * sum(ϕ[d])
    end

    for d in 1:nDim
        # prior of δ
        if d == 1
            prob += (hyperParam["a_δ1"] - 1) * sum(log.(δ[d])) - sum(δ[d])
        else
            prob += (hyperParam["a_δ2"] - 1) * sum(log.(δ[d])) - sum(δ[d])
        end
        # prior of η
        for i in 1:sz[d]
            prob -= 0.5 * sum(YEst.coresQ[nDim][:, i, :] .^ 2)
        end
    end
    # noise part
    prob += (hyperParam["a_τ"] - 1) * log(τ) - hyperParam["b_τ"] * τ
    return prob
end


function findPruneIndex(
    YEst :: TRLF{T},
    trun :: Float64;
    method = "absolute",  # absolute or relative
    isPruneFactor = true
    ) where {T <: Real}

    nDim = YEst.ndim
    pruneIndex = Array{Array{Bool}}(undef, nDim)
    # get the truncation index
    for d in 1:nDim
        dp, _ = round_index(d, nDim)
        Gd = matricization(YEst.coresQ[d], 1)
        Gp = matricization(YEst.coresQ[dp], 3)
        if d ≡ 1
            Gtemp = Gd
        elseif d ≡ nDim
            Gtemp = Gp
        else
            Gtemp = [Gd Gp]
            # flen = YEst.rank[dp] * YEst.shape[dp] + YEst.rank[d] * YEst.shape[d]
        end
        flen = size(Gtemp, 2)
        comPower = sqrt.(diag(Gtemp * Gtemp'))
        if method == "absolute"
            # trun method 1
            comPower /= flen
            pruneIndex[d] = comPower .> trun
        elseif method == "relative"
            # trun method 2
            trun_ = trun * maximum(comPower)
            pruneIndex[d] = comPower .> trun_
        end
        if d ≡ nDim && isPruneFactor ≡ false
            pruneIndex[d] = convert(BitArray, ones(Bool, length(comPower)))
        end
    end

    return pruneIndex
end


function pruneFactors!(
    YEst, ϕ, δ, trun;
    method = "absolute", isPruneFactor = true
    )
    # prune for EM and MAP

    nDim = length(YEst.coresQ)
    rankest = zeros(Int64, nDim)

    pruneIndex = findPruneIndex(YEst, trun; method = method, isPruneFactor = isPruneFactor)
    # prune the factors
    for d in 1:nDim
        dp, _ = round_index(d, nDim)
        rankest[d] = sum(pruneIndex[d])
        # Q
        YEst.coresQ[dp] = YEst.coresQ[dp][:, :, pruneIndex[d]]
        YEst.coresQ[d] = YEst.coresQ[d][pruneIndex[d], :, :]

        # ϕ
        if d ≡ 1
            ϕ[d] = ϕ[d][pruneIndex[d], :, :]
        elseif d ≡ nDim
            ϕ[dp] = ϕ[dp][:, :, pruneIndex[d]]
        else
            ϕ[dp] = ϕ[dp][:, :, pruneIndex[d]]
            ϕ[d] = ϕ[d][pruneIndex[d], :, :]
        end

        # δ
        δ[d] = δ[d][pruneIndex[d]]
    end
    YEst.rank = rankest
    return pruneIndex
end


function pruneFactors!(
    YEst, coresQPlus, ϕ, ϕPlus,
    δ, δPlus, trun; method = "absolute"
    )
    # prune for MCMC

    nDim = length(YEst.coresQ)
    rankest = zeros(Int64, nDim)

    pruneIndex = findPruneIndex(YEst, trun; method = method)
    # prune the factors
    for d in 1:nDim
        dp, _ = round_index(d, nDim)
        rankest[d] = sum(pruneIndex[d])
        # Q
        YEst.coresQ[dp] = YEst.coresQ[dp][:, :, pruneIndex[d]]
        YEst.coresQ[d] = YEst.coresQ[d][pruneIndex[d], :, :]
        coresQPlus[dp] = coresQPlus[dp][:, :, pruneIndex[d]]
        coresQPlus[d] = coresQPlus[d][pruneIndex[d], :, :]


        # ϕ
        if d ≡ 1
            ϕ[d] = ϕ[d][pruneIndex[d], :, :]
            ϕPlus[d] = ϕPlus[d][pruneIndex[d], :, :]
        elseif d ≡ nDim
            ϕ[dp] = ϕ[dp][:, :, pruneIndex[d]]
            ϕPlus[dp] = ϕPlus[dp][:, :, pruneIndex[d]]
        else
            ϕ[dp] = ϕ[dp][:, :, pruneIndex[d]]
            ϕ[d] = ϕ[d][pruneIndex[d], :, :]
            ϕPlus[dp] = ϕPlus[dp][:, :, pruneIndex[d]]
            ϕPlus[d] = ϕPlus[d][pruneIndex[d], :, :]
        end

        # δ
        δ[d] = δ[d][pruneIndex[d]]
        δPlus[d] = δPlus[d][pruneIndex[d]]
    end
    YEst.rank = rankest
end


function genMask(sz, mis; tp = Float64, rng = MersenneTwister())
    # sz: mask size
    # mis: missing ratio
    mask = rand(rng, tp, sz...)
    mask[mask .>= mis] .= 1
    mask[mask .< mis] .= 0
    return mask
end

function safeMvNormal(μ, ΣInv; N = 1)
    ΣInv_ = (ΣInv' + ΣInv) / 2
    d = length(μ)
    R = cholesky(ΣInv_)
    zi = rand(MvNormal(zeros(d), I), N)
    x = inv(R.U) * zi
    x = broadcast(+, x, μ)
    return x
end

function image2Array(img)
    x = channelview(img)
    x = convert(Array{Float64}, x)
    x = permutedims(x, [2, 3, 1])
    return x
end

function array2Image(x)
    img = permutedims(x, [3, 1, 2])
    img = colorview(RGB, img)
    return img
end

function circshift(A :: CartesianIndex, shifts :: Int64)
    AShift = zeros(Int64, length(A))
    for i in 1:length(A)
        AShift[i] = A[i]
    end
    AShift = circshift(AShift, shifts)
    # return CartesianIndex(tuple(AShift...))
    return AShift
end

function tensorize(
    X :: Array{T},
    shape :: Array{Int64, 1},
    isProduct :: Bool = false) where {T <: Real}

    D = length(shape)
    newShape = copy(shape)
    newShape = [newShape..., newShape...]
    append!(newShape, 3)  # add channel
    XTen = reshape(X, newShape...)
    perm = zeros(Int64, ndims(XTen))
    for d in 1:D
        perm[2*(d-1)+1] = d
        perm[2*d] = D + d
    end
    perm[end] = 2 * D + 1
    XTen = permutedims(XTen, perm)
    if isProduct
        productShape = copy(shape)
        productShape = productShape .^ 2
        append!(productShape, 3)  # add channel
        XTen = reshape(XTen, productShape...)
    end

    return XTen
end

function unTensorize(
    X :: Array{T},
    shape :: Array{Int64, 1},
    isProduct :: Bool = false) where {T <: Real}

    if isProduct
        productShape = collect(size(X))
        productShape = productShape[1:end-1]
        productShape = convert(Array{Int64}, sqrt.(productShape))
        productShape = [productShape..., productShape...]
        append!(productShape, 3)
        X = reshape(X, productShape...)
    end 

    D = ndims(X) - 1
    perm = [collect(1:2:D)..., collect(2:2:D)..., ndims(X)]
    X_ = permutedims(X, perm)
    return reshape(X_, shape...)
end


function array2Patch(x :: Array{T}, patchSize) where {T <: Real}
    k1, k2, c = size(x)
    nrow = k1 ÷ patchSize[1]
    ncol = k2 ÷ patchSize[2]
    N = prod(patchSize)
    out = zeros(T, nrow, ncol, c, N)
    count = 1
    for i = 1:patchSize[1], j = 1:patchSize[2]
        row1 = (i-1) * nrow + 1
        row2 = i * nrow
        col1 = (j-1) * ncol + 1
        col2 = j * ncol
        out[:, :, :, count] = x[row1:row2, col1:col2, :]
        count += 1
    end
    return out
end


function patch2Array(x :: Array{T}, patchSize) where {T <: Real}
    nrow, ncol, c, N = size(x)
    k1 = nrow * patchSize[1]
    k2 = nrow * patchSize[2]
    out = zeros(T, k1, k2, c)
    count = 1
    for i = 1:patchSize[1], j = 1:patchSize[2]
        row1 = (i-1) * nrow + 1
        row2 = i * nrow
        col1 = (j-1) * ncol + 1
        col2 = j * ncol
        out[row1:row2, col1:col2, :] = x[:, :, :, count]
        count += 1
    end
    return out
end


RSE(X, XHat) = norm(XHat - X) / norm(X)

MyPSNR(X, XHat) = 10 * log(10, maximum(X) ^ 2 / (norm(X - XHat) ^ 2 / length(X)))
