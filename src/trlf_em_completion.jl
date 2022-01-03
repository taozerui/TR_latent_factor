#=
Toy model for Multiplicative Gamma Process TR latent model,
without any constraints on the factor tensor.
Gibbs sampler algorithm.
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
include("./update_hyper.jl")


"""
The MGP-TRLF model for missing data.
Infer with the PX-EM algorithm.

Input:
    - Y: An N-way array of the oberserved tensor.
    - mask: Mask array of the same size of Y. 0 represents missing entries and 1 otherwise.
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
    mask :: Array{T},
    rank :: Array{Int, 1},
    opt :: Optim; optNoise = true
    ) where {T <: Real}
    @info("EM Algorithm for Multiplicative Gamma Process Tensor Ring Latent Factor Model.")

    D = ndims(Y)
    sz = collect(size(Y))
    N = size(Y, D)  # sampler number

    # hyperparameters
    if optNoise
        hyperParam = Dict{String, T}(
            "a_τ"  => 1e-6,  # hyperparameter for noise level
            "b_τ"  => 1e-6,  # hyperparameter for noise level
            "a_δ1" => 2.1,  # hyperparameter for factor R variance
            "a_δ2" => 3.1,  # hyperparameter for factor R variance, must > 1
            "ν"    => 1.5,  # ϕ
        )
    else
        hyperParam = Dict{String, T}(
            "a_τ"  => 1e-1,  # hyperparameter for noise level
            "b_τ"  => 1e-5,  # hyperparameter for noise level
            "a_δ1" => 2.1,  # hyperparameter for factor R variance
            "a_δ2" => 3.1,  # hyperparameter for factor R variance, must > 1
            "ν"    => 1.5,  # ϕ
        )
    end

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
    τ = T(hyperParam["a_τ"] / hyperParam["b_τ"])  # todo: noise level, this can be set different
    # rotate
    A = Matrix{T}(I, rank[D], rank[D])
    B = Matrix{T}(I, rank[1], rank[1])
    V = kron(B, A)

    # inference
    rankest = copy(rank)
    lossTot = []
    YOld = zero(Y)
    lossOld = oneunit(T)
    for epoch in 1:opt.iter

        # E-step
        V = updateη!(YEst, Y, mask, U, τ; isSample = false)

        # M-step
        # update core tensor Q
        emupdateCoresQMask!(YEst, Y, mask, V, ϕ, U, τ)
        # update hyperparameter δ
        updateδ!(
            δ, YEst, ϕ, U, hyperParam["a_δ1"],
            hyperParam["a_δ2"]; isSample = false
        )
        # update u
        U = computeγ(δ)
        # update ϕ
        updateϕ!(ϕ, YEst, U, hyperParam["ν"]; isSample = false)
        # update estimates
        YFull = fullData(YEst)
        # update noise level
        a_τ_post = hyperParam["a_τ"]
        b_τ_post = hyperParam["b_τ"]
        if optNoise
            a_τ_post += 0.5 * sum(mask)
            eTR = sum((mask .* Y) .^ 2) - 2 * vec(mask .* Y)' * vec(YFull)
            eTR += expectationTR(YEst, mask, V)
            b_τ_post += 0.5 * eTR
        end
        τ = a_τ_post / b_τ_post

        # compute loglikelihood
        loglike = logJointDist(Y, YFull, YEst, mask, ϕ, U, δ, τ, hyperParam)
        append!(lossTot, loglike)

        # rotate
        if epoch < opt.rotate
            k1 = YEst.rank[D]
            k2 = YEst.rank[1]
            V = cat(V..., dims = 3)
            V = permutedims(V, [3, 1, 2])
            V = reshape(V, N, k1, k2, k1, k2)
            # left rotate
            η = YEst.coresQ[D]
            meanη = @ncon(
                [η, inv(B), η], [[-1, 1, 2], [2, 3], [-2, 1, 3]]
            )
            SXXB = @ncon([inv(B), V], [[1, 2], [-1, -2, 1, -3, 2]])
            SXXB = dropdims(sum(SXXB, dims = 1), dims = 1)
            SXXB += meanη
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
            SXXA = @ncon([inv(A), V], [[1, 2], [-1, 1, -2, 2, -3]])
            SXXA = dropdims(sum(SXXA, dims = 1), dims = 1)
            SXXA += meanη
            B = SXXA / (N * k1)
            B = (B + B') / 2
            C = cholesky(B)
            for i in 1:size(YEst.coresQ[1], 2)
                YEst.coresQ[1][:, i, :] = C.U * YEst.coresQ[1][:, i, :]
            end
        end

        # prune factors
        if opt.isPrune
            pIndex = pruneFactors!(
                YEst, ϕ, δ, opt.trun;
                method = opt.pruneMethod
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

        # if opt.isPrint > 0 && mod(epoch, opt.isPrint) == 0
        #     println("Epoch $epoch, loglikelihood is $loglike, rank estimate is $rankest.")
        # end
        
        # converge
        # println("Epoch $epoch, diff $(norm(YFull - YOld) / length(YFull)).")
        change = norm(YFull - YOld) / length(YFull)
        if opt.isPrint > 0 && mod(epoch, opt.isPrint) == 0
            println("Epoch $epoch: Rank estimate is $rankest, Change is $change.")
        end
        if change < opt.ϵ
            println("Converge in $epoch iterations.")
            break
        end
        YOld = YFull
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
function expectationSubchain(YEst :: TRLF{T}, mask, V, dSkip :: Int) where {T <: Real}
    D = YEst.ndim
    sz = YEst.shape
    N = YEst.shape[D]
    rank = YEst.rank

    EvG2 = Array{Array, 1}(undef, D)
    for d in 1:D
        _, dn = round_index(d, D)
        EvG2_d = zeros(T, sz[d], rank[d], rank[dn], rank[d], rank[dn])
        for i in 1:sz[d]
            mu = YEst.coresQ[d][:, i, :]
            EvG2Mean = @ncon([mu, mu], [[-1, -2], [-3, -4]])
            if d == D
                EvG2_d[i, ..] = EvG2Mean + reshape(V[i], rank[d], rank[dn], rank[d], rank[dn])
            else
                EvG2_d[i, ..] = EvG2Mean
            end
        end
        EvG2[d] = EvG2_d
    end
    szNew = size(EvG2[dSkip], 2) * size(EvG2[dSkip], 3)

    EvG2 = reverse(circshift(EvG2, -(dSkip-1)))  # todo: remove reverse operation
    pop!(EvG2)
    maskPerm = permutedims(mask, circshift(1:D, -(dSkip-1)))

    einsumIndex = Array{Array{Int}, 1}(undef, D)
    einsumIndex[1] = [-1, collect(D-1:-1:1)...]
    einsumIndex[2] = [1, D-1+1, -2, D-1+2, -4]
    for d in 2:D-2
        einsumIndex[d+1] = [d, D-1+2(d-1)+1, D-1+2(d-1)-1, D-1+2d, D-1+2(d-1)]
    end
    einsumIndex[D-1+1] = [D-1, -3, D-1+2(D-2)-1, -5, D-1+2(D-2)]

    pushfirst!(EvG2, maskPerm)
    out = @ncon(EvG2, einsumIndex)    
    out = reshape(out, :, szNew, szNew)
    return out
end


raw"""
# About
Expectation of the sqaure of a TR tensor, i.e.,
`` E || O \ast << G^{(1)}, \dots, G^{(D)} >> ||^2_F ``,
Given each `` E [G^{(1)} G^{(1)}] ``.
"""
function expectationTR(YEst, mask, V)
    D = YEst.ndim
    N = YEst.shape[D]
    rank = YEst.rank
    k1 = rank[D]
    k2 = rank[1]

    η = permutedims(YEst.coresQ[D], [1, 3, 2])
    η = reshape(η, :, N)
    W = computeCoef(YEst)
    W = reshape(W, size(W, 1), :, size(W, ndims(W)))
    W = permutedims(W, [2, 3, 1])
    W = reshape(W, :, k1*k2)
    Od = matricization(mask, D)

    out = 0.0
    for i in 1:N
        ηi = vec(η[:, i])
        foo = diag(W * (ηi * ηi' + V[i]) * W')
        out += sum(Od[i, :] .* foo)
    end
    return out
end


function emupdateCoresQMask!(YEst :: TRLF{T}, Y, mask, Vη, ϕ, U, τ) where {T <: Real}
    rank = YEst.rank
    sz = YEst.shape
    D = YEst.ndim
    for d in 1:D-1
        _, dn = round_index(d, D)

        Yd = matricization(Y, d, true)
        Od = matricization(Y, d, true)
        Qd = zeros(rank[d], sz[d], rank[dn])

        GNeq = subchain(YEst, d)
        GNeqVec = reshape(
            permutedims(GNeq, [2, 3, 1]), size(GNeq, 2), :
        )
        UKron = kron(Diagonal(U[dn]), Diagonal(U[d]))
        EvG2 = expectationSubchain(YEst, mask, Vη, d)
        for i in 1:sz[d]
            Γ = broadcast(*, vec(Yd[i, :]), GNeqVec)
            Γ = vec(sum(Γ, dims = 1))

            VInv = τ * EvG2[i, ..] + Diagonal(vec(ϕ[d][:, i, :])) .* UKron
            V = inv(VInv)
            
            μ = τ * V * Γ
            Qdi = reshape(μ, rank[d], rank[dn])
            Qd[:, i, :] = Qdi
        end
        YEst.coresQ[d] = Qd
    end
end
