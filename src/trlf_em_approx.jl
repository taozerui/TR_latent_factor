#=
Toy model for Multiplicative Gamma Process TR latent model.
Gibbs sampler algorithm.
=#


using Distributions
using LinearAlgebra
using EllipsisNotation
using TensorOperations
using BenchmarkTools
using OMEinsum

include("./type.jl")
include("./utils.jl")
include("./tn_tools.jl")
include("./tensor_basics.jl")
include("./update_hyper.jl")


"""
The Multiplicative Gamma Process Tensor Ring Latent Factor (MGP-TRLF) model.
Infer with the PX-EM algorithm.

Input:
    - Y: An N-way array of the oberserved tensor.
    - rank: An array of `Int` of length `ndims(Y)`. The TR rank.
    - opt: Some options about the model. See `./type.jl` for details.

Output:
    - YEst: An `TR` struct of the estimated tensor. Use `full(YEst)` to check the full tensor.
    - lossTot: Store the log joint distribution values.
    - post: Hyper-parameters
    - computeΣ(YEst): Estimated covariance matrix.
"""
function TRLF_EM(
    Y :: TRLF{T},  # shape (p1, \dots, p_D, N)
    rank :: Array{Int, 1},
    opt :: Optim
    ) where {T <: Real}

#     @info("EM Algorithm for Multiplicative Gamma Process Tensor Ring Latent Factor Model.")

    D = ndims(Y)
    sz = collect(size(Y))
    N = size(Y, D)  # sampler number
    flen = Int(prod(sz) / N)

    # hyperparameters
    hyperParam = Dict{String, T}(
        "a_τ"  => 1e-6,  # hyperparameter for noise level
        "b_τ"  => 1e-6,  # hyperparameter for noise level
        "a_δ1" => 2.1,  # hyperparameter for factor R variance
        "a_δ2" => 3.1,  # hyperparameter for factor R variance, must > 1
        "ν"    => 1.5,  # ϕ
    )

    # init estimator
    if opt.init in ("randn", "rand")
        YEst = TRLF{T}(shape = sz, rank = rank)
        init!(YEst; method = opt.init, scale = opt.initScale)
    else
        @error "Wrong initialize method!"
    end
    rank = YEst.rank

    # init local shrinkage ϕ
    ϕ = Array{Array{T}, 1}(undef, D-1)
    for d in 1:D-1
        _, dn = round_index(d, D)
        ϕ[d] = ones(T, rank[d], sz[d], rank[dn])
    end
    # init global shrinkage U and δ
    δ = Array{Array{T}, 1}(undef, D)
    for d in 1:D
        δ[d] = ones(T, rank[d])
    end
    U = computeγ(δ)
    # init noise
    τ = one(T)
    # rotate
    A = Matrix{T}(I, rank[D], rank[D])
    B = Matrix{T}(I, rank[1], rank[1])
    V = kron(B, A)

    # inference
    rankest = copy(rank)
    lossTot = []
    # ΣEst = computeΣ(YEst)
    # ΣOld = zero(ΣEst)
    for epoch in 1:opt.iter

        # E-step
        V = updateη!(YEst, Y, τ)
        Eη2 = factor_expectation(YEst, V)

        # M-step
        # update core tensor Q
        emupdateCoresQ!(YEst, Eη2, Y, ϕ, U, τ)
        # update hyperparameter δ
        updateδ!(
            δ, YEst, ϕ, U, hyperParam["a_δ1"],
            hyperParam["a_δ2"]; isSample = false
        )
        # update u
        U = computeγ(δ)
        # update ϕ
        updateϕ!(ϕ, YEst, U, hyperParam["ν"]; isSample = false)
        # update noise level
        a_τ_post = hyperParam["a_τ"] + 0.5 * prod(sz)
        eTR = inner_product(Y, Y) - 2 * inner_product(Y, YEst) + inner_product(YEst, Eη2)
        b_τ_post = hyperParam["b_τ"] .+ 0.5 * eTR
        τ = a_τ_post ./ b_τ_post

        # compute loglikelihood
        # loglike = logJointDist(Y, YFull, YEst, ϕ, U, δ, τ, hyperParam)
        # append!(lossTot, loglike)

        # compute diff
        # ΣEst = computeΣ(YEst)
        # diff = norm(ΣEst - ΣOld)  # TODO: compute by cores.
        # println("Epoch $epoch -- Diff is $diff.")
        # ΣOld = ΣEst


        # rotate
        if epoch < opt.rotate
            # left rotate
            η = YEst.coresQ[D]
            k1, k2 = size(η[:, 1, :])
            meanη = @ncon(
                [η, inv(B), η], [[-1, 1, 2], [2, 3], [-2, 1, 3]]
            )
            SXXB = meanη + @ncon([inv(B), V], [[1, 2], [-1, 1, -2, 2]]) * N # not sure
            A = SXXB / (N * k2)
            A = (A + A') / 2
            C = cholesky(A)
            for i in 1:size(YEst.coresQ[D-1], 2)
                YEst.coresQ[D-1][:, i, :] = YEst.coresQ[D-1][:, i, :] * C.L
            end
            # right rotate
            meanη = @ncon(
                [η, inv(A), η], [[1, 2, -1], [1, 3], [3, 2, -2]]
            )
            SXXA = meanη + @ncon([inv(A), V], [[1, 2], [1, -1, 2, -2]]) * N
            B = SXXA / (N * k1)
            B = (B + B') / 2
            # println(B)
            C = cholesky(B)
            for i in 1:size(YEst.coresQ[1], 2)
                YEst.coresQ[1][:, i, :] = C.U * YEst.coresQ[1][:, i, :]
            end
        end

        # prune factors
        if opt.isPrune && epoch < opt.burnin
            pIndex = pruneFactors!(
                YEst, ϕ, δ, opt.trun;
                method = opt.pruneMethod, isPruneFactor = opt.isPruneFactor
            )
            rank = YEst.rank
            rankest = copy(rank)
            if epoch < opt.rotate
                A = A[pIndex[D], pIndex[D]]
                B = B[pIndex[1], pIndex[1]]
            end
        else
            rankest = sum.(findPruneIndex(YEst, opt.trun; method = opt.pruneMethod))
        end
        U = computeγ(δ)

        if opt.isPrint > 0 && mod(epoch, opt.isPrint) == 0
            println("Epoch $epoch, loglikelihood is $loglike, rank estimate is $rankest.")
        end
    end

    post = Dict(
        "τ" => τ,
        "δ" => δ,
        "ϕ" => ϕ,
    )

    return YEst, lossTot, post
end


function inner_y_factor(Y :: TRLF{T}, YEst :: TRLF{T}) where T
    dim = YEst.ndim
    N = size(Y, dim)
    out = @ncon([Y.coresQ[1], YEst.coresQ[1]], [[-1, 1, -2], [-3, 1, -4]])
    for d in 2:dim-1
        cache = @ncon([Y.coresQ[d], YEst.coresQ[d]], [[-1, 1, -2], [-3, 1, -4]])
        out = @ncon([out, cache], [[-1, 1, -3, 2], [1, -2, 2, -4]])
    end
    out = @ncon([Y.coresQ[dim], out], [[1, -1, 2], [2, 1, -3, -2]])
    # out = permutedims(out, [1, 3, 2])
    out = reshape(out, N, :)
    return out'
end


function inner_y_factor(Y:: TRLF{T}, YEst :: TRLF, d :: Int) where T
    dim = YEst.ndim
    N = size(Y, d)

    cores1 = circshift(Y.coresQ, -d)
    cores2 = circshift(YEst.coresQ, -d)
    out = @ncon([cores1[1], cores2[1]], [[-1, 1, -2], [-3, 1, -4]])
    for d in 2:dim-1
        cache = @ncon([cores1[d], cores2[d]], [[-1, 1, -2], [-3, 1, -4]])
        out = @ncon([out, cache], [[-1, 1, -3, 2], [1, -2, 2, -4]])
    end
    out = @ncon([cores1[dim], out], [[1, -1, 2], [2, 1, -3, -2]])
    out = reshape(out, N, :)
    return out
end


function factor_expectation(YEst, Vη)
    D = YEst.ndim
    N = YEst.shape[D]
    # compute expectation term
    η = YEst.coresQ[D]
    k2, _, k4 = size(η)
    Eη2 = @ncon([η, η], [[-1, 1, -2], [-3, 1, -4]]) + Vη * N
    Eη2 = reshape(
        permutedims(Eη2, [2, 1, 4, 3]),
        k2*k4, k2*k4
    )
    Eη2 = reshape(Eη2, k4, k2, k4, k2)
    return Eη2
end


function outer_factors(YEst :: TRLF{T}) where T
    dim = YEst.ndim
    k = YEst.rank[1] * YEst.rank[dim]
    out = @ncon([YEst.coresQ[dim-1], YEst.coresQ[dim-1]], [[-2, 1, -1], [-4, 1, -3]])
    for d = dim-2:-1:1
        cache = @ncon([YEst.coresQ[d], YEst.coresQ[d]], [[-2, 1, -1], [-4, 1, -3]])
        out = @ncon([out, cache], [[-1, 1, -3, 2], [1, -2, 2, -4]])
    end
    out = reshape(out, k, k)
    return out
end


function outer_factors(YEst :: TRLF{T}, Eη2, d) where T
    D = YEst.ndim
    N = YEst.shape[D]

    cores = circshift(YEst.coresQ, -d)

    k = size(YEst.coresQ[d], 1) * size(YEst.coresQ[d], 3)
    if d ≡ 1
        out = Eη2
    else
        out = @ncon([cores[D-1], cores[D-1]], [[-2, 1, -1], [-4, 1, -3]])
    end
    for i = D-2:-1:1
        if i ≡ D-d
            cache = Eη2
        else
            cache = @ncon([cores[i], cores[i]], [[-2, 1, -1], [-4, 1, -3]])
        end
        out = @ncon([out, cache], [[-1, 1, -3, 2], [1, -2, 2, -4]])
    end
    out = reshape(out, k, k)
    return out
end


function updateη!(
    YEst :: TRLF{T},
    Y :: TRLF{T},
    τ) where {T <: Real}

    rank = YEst.rank
    sz = YEst.shape
    D = YEst.ndim
    k = rank[1] * rank[D]  # factor number

    W_Yvec = inner_y_factor(Y, YEst)
    VInv = τ * outer_factors(YEst) + Matrix{T}(I, k, k)
    V = inv(VInv)
    μ = τ * V * W_Yvec

    η = zeros(T, rank[D], sz[D], rank[1])
    for i in 1:sz[D]
        ηi = μ[:, i]
        η[:, i, :] = reshape(ηi, rank[D], rank[1])
    end
    YEst.coresQ[D] = η

    k1 = rank[D]
    k2 = rank[1]

    return reshape((V + V') / 2, k1, k2, k1, k2)
end


function inner_product(Y1 :: TRLF{T}, Y2 :: TRLF{T}) where T
    D = Y1.ndim
    out = @ncon([Y1.coresQ[1], Y2.coresQ[1]], [[-1, 1, -3], [-2, 1, -4]])
    for d = 2:D
        cache = @ncon([Y1.coresQ[d], Y2.coresQ[d]], [[-1, 1, -3], [-2, 1, -4]])
        out = @ncon([out, cache], [[-1, -2, 1, 2], [1, 2, -3, -4]])
    end
    out_ = ein"ijij->"(out)
    return out_[1]
end


function inner_product(Y :: TRLF{T}, Eη2 :: Array) where T
    D = Y.ndim
    out = @ncon([Y.coresQ[1], Y.coresQ[1]], [[-1, 1, -3], [-2, 1, -4]])
    for d = 2:D
        if d == D
            cache = permutedims(Eη2, [4, 2, 3, 1])
        else
            cache = @ncon([Y.coresQ[d], Y.coresQ[d]], [[-1, 1, -3], [-2, 1, -4]])
        end
        out = @ncon([out, cache], [[-1, -2, 1, 2], [1, 2, -3, -4]])
    end
    out_ = ein"ijij->"(out)
    return out_[1]
end


function emupdateCoresQ!(
    YEst :: TRLF{T},
    Eη2 :: Array,
    Y :: TRLF{T},
    ϕ, U, τ) where {T <: Real}

    rank = YEst.rank
    sz = YEst.shape
    D = YEst.ndim
    for d in 1:D-1
        _, dn = round_index(d, D)

        UKron = kron(Diagonal(U[dn]), Diagonal(U[d]))
        W_Yvec = inner_y_factor(Y, YEst, d)
        EvG2 = outer_factors(YEst, Eη2, d)

        Qd = zeros(rank[d], sz[d], rank[dn])
        for i in 1:sz[d]

            VInv = τ * EvG2 + Diagonal(vec(ϕ[d][:, i, :])) .* UKron
            V = inv(VInv)
            
            μ = τ * V * vec(W_Yvec[i, :])
            Qdi = reshape(μ, rank[d], rank[dn])
            Qd[:, i, :] = Qdi
        end
        YEst.coresQ[d] = Qd
    end
end


function cov_diff(Y :: TRLF{T}, cores :: Array) where T
    return 0
end
