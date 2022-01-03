#=
Toy model for Multiplicative Gamma Process TR latent model.
Gibbs sampler algorithm.
=#


using Distributions
using LinearAlgebra
using EllipsisNotation
using TensorOperations
using BenchmarkTools

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
    Y :: Array{T},  # shape (p1, \dots, p_D, N)
    rank :: Array{Int, 1},
    opt :: Optim
    ) where {T <: Real}

    @info("EM Algorithm for Multiplicative Gamma Process Tensor Ring Latent Factor Model.")

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
    τ = oneunit(T)
    # rotate
    A = Matrix{T}(I, rank[D], rank[D])
    B = Matrix{T}(I, rank[1], rank[1])
    V = kron(B, A)

    # inference
    rankest = copy(rank)
    lossTot = []
    lossOld = oneunit(T)
    for epoch in 1:opt.iter

        # E-step
        V = updateη!(YEst, Y, τ; isSample = false)

        # M-step
        # update core tensor Q
        emupdateCoresQ!(YEst, V, Y, ϕ, U, τ)
        # update hyperparameter δ
        updateδ!(
            δ, YEst, ϕ, U, hyperParam["a_δ1"],
            hyperParam["a_δ2"]; isSample = false)
        # update u
        U = computeγ(δ)
        # update ϕ
        updateϕ!(ϕ, YEst, U, hyperParam["ν"]; isSample = false)
        # update estimates
        YFull = fullData(YEst)
        # update noise level
        a_τ_post = hyperParam["a_τ"] + 0.5 * prod(sz)
        eTR = sum(Y .^ 2) - 2 * vec(Y)' * vec(YFull)
        eTR += expectationTR(YEst, V)
        b_τ_post = hyperParam["b_τ"] .+ 0.5 * eTR
        τ = a_τ_post / b_τ_post

        # compute loglikelihood
        loglike = logJointDist(Y, YFull, YEst, ϕ, U, δ, τ, hyperParam)
        append!(lossTot, loglike)

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

        lossChange = abs(loglike - lossOld) / abs(loglike)
        if opt.isPrint > 0 && mod(epoch, opt.isPrint) == 0
            println("Epoch $epoch: Rank estimate is $rankest, Change is $lossChange.")
        end

        if lossChange < opt.ε
            println("Converge in $epoch epochs.")
            break
        end
        lossOld = loglike
    end

    post = Dict(
        "τ" => τ,
        "δ" => δ,
        "ϕ" => ϕ,
    )

    return YEst, lossTot, post
end


"""
Expectation of the inner product of subchain.
"""
function expectationSubchain(YEst :: TRLF{T}, V, dSkip :: Int) where {T <: Real}
    D = YEst.ndim
    N = YEst.shape[D]
    rank = YEst.rank

    if dSkip ≡ D-1
        Gr = zeros(T, rank[D], 1, rank[D])
        Gr[:, 1, :] = Matrix{T}(I, rank[D], rank[D])
    else
        Gr = subchainRight(YEst.coresQ[1:D-1], dSkip+1)
    end
    Gr = permutedims(Gr, [1, 3, 2])
    k1 = size(Gr, 1)
    k2 = size(Gr, 2)
    if dSkip ≡ 1
        Gl = zeros(T, rank[1], 1, rank[1])
        Gl[:, 1, :] = Matrix{T}(I, rank[1], rank[1])
    else
        Gl = subchainLeft(YEst.coresQ[1:D-1], dSkip-1)
    end
    Gl = permutedims(Gl, [3, 1, 2])
    k3 = size(Gl, 1)
    k4 = size(Gl, 2)
    η = YEst.coresQ[D]
    Eη2 = @ncon([η, η], [[-1, 1, -2], [-3, 1, -4]]) + V * N
    Eη2 = reshape(
        permutedims(Eη2, [2, 1, 4, 3]),
        k2*k4, k2*k4
    )
    Eη2 = reshape(Eη2, k4, k2, k4, k2)

    out = @ncon(
        [Gr, Gl, Eη2, Gr, Gl],
        [[-2, 1, 5], [-1, 2, 6], [2, 1, 4, 3], [-4, 3, 5], [-3, 4, 6]];
        order = [1, 3, 2, 6, 5, 4]
    )
    out = reshape(out, k1*k3, k1*k3)

    return out
end


raw"""
# About
Expectation of the sqaure of a TR tensor, i.e.,
`` E || O \ast << G^{(1)}, \dots, G^{(D)} >> ||^2_F ``,
Given each `` E [G^{(1)} G^{(1)}] ``.
"""
function expectationTR(YEst, V)
    D = YEst.ndim
    N = YEst.shape[D]
    k1 = YEst.rank[D]
    k2 = YEst.rank[1]

    η = permutedims(YEst.coresQ[D], [1, 3, 2])
    η = reshape(η, :, N)
    W = computeCoef(YEst)
    W = reshape(W, size(W, 1), :, size(W, ndims(W)))
    W = permutedims(W, [2, 3, 1])
    W = reshape(W, :, k1*k2)

    out = tr(W * (η * η' + reshape(V, k1 * k2, :) * N) * W')

    return out
end


function emupdateCoresQ!(YEst :: TRLF{T}, Vη, Y, ϕ, U, τ) where {T <: Real}
    rank = YEst.rank
    sz = YEst.shape
    D = YEst.ndim
    for d in 1:D-1
        _, dn = round_index(d, D)

        Yd = matricization(Y, d, true)
        Qd = zeros(rank[d], sz[d], rank[dn])

        GNeq = subchain(YEst, d)
        GNeqVec = reshape(
            permutedims(GNeq, [2, 3, 1]), size(GNeq, 2), :
        )
        UKron = kron(Diagonal(U[dn]), Diagonal(U[d]))
        Γ = Yd * GNeqVec
        EvG2 = expectationSubchain(YEst, Vη, d)
        for i in 1:sz[d]

            VInv = τ * EvG2 + Diagonal(vec(ϕ[d][:, i, :])) .* UKron
            V = inv(VInv)
            
            μ = τ * V * vec(Γ[i, :])
            Qdi = reshape(μ, rank[d], rank[dn])
            Qd[:, i, :] = Qdi
        end
        YEst.coresQ[d] = Qd
    end
end
