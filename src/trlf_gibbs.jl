#=
Toy model for Multiplicative Gamma Process TR latent model,
without any constraints on the factor tensor.
Gibbs sampler algorithm.
=#


using Distributions
using LinearAlgebra
using EllipsisNotation
using TensorOperations

include("./type.jl")
include("./utils.jl")
include("./tn_tools.jl")
include("./tensor_basics.jl")
include("./update_hyper.jl")


"""
The Multiplicative Gamma Process Tensor Ring (MGP-TR) model.
Infer with the MCMC algorithm.

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
function TRLF_GS(
    Y :: Array{T},  # shape (p1, \dots, p_D, N)
    rank :: Array{Int, 1},
    opt :: Optim
    ) where {T <: Real}
    @info("Gibbs Sampler for Multiplicative Gamma Process Tensor Ring")

    D = ndims(Y)
    sz = collect(size(Y))
    N = size(Y, D)  # sampler number

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

    # init core
    coresQPlus = Array{Array{T}, 1}(undef, D)
    for d in 1:D
        _, dn = round_index(d, D)
        coresQPlus[d] = zeros(T, rank[d], sz[d], rank[dn])
    end
    # init local shrinkage ϕ
    ϕ = Array{Array{T}, 1}(undef, D-1)
    ϕPlus = Array{Array{T}, 1}(undef, D-1)
    for d in 1:D-1
        _, dn = round_index(d, D)
        ϕ[d] = ones(T, rank[d], sz[d], rank[dn])
        ϕPlus[d] = zeros(T, rank[d], sz[d], rank[dn])
    end
    # init global shrinkage U and δ
    δ = Array{Array{T}, 1}(undef, D)
    δPlus = copy(δ)
    for d in 1:D
        δ[d] = ones(T, rank[d])
        δPlus[d] = zeros(T, rank[d])
    end
    U = computeγ(δ)
    # init noise
    τ = oneunit(T)  # todo: noise level, this can be set different
    τPlus = zero(T)
    ΣPlus = zeros(T, prod(sz) ÷ N, prod(sz) ÷ N)

    # inference
    rankest = copy(rank)
    lossTot = []
    gsCount = 0
    for epoch in 1:opt.iter
        isCount = epoch ≥ opt.burnin && mod(epoch - opt.burnin, opt.thin) ≡ 0
        # sample core tensor Q
        gsUpdateCoresQ!(YEst, Y, ϕ, U, τ)
        updateη!(YEst, Y, τ; isSample = true)
        if isCount
            coresQPlus .+= YEst.coresQ
            gsCount += 1
        end

        # sample hyperparameter δ
        updateδ!(
            δ, YEst, ϕ, U, hyperParam["a_δ1"],
            hyperParam["a_δ2"]; isSample = true
        )
        if isCount
            δPlus .+= δ
        end

        # update u
        U = computeγ(δ)

        # sample ϕ
        updateϕ!(ϕ, YEst, U, hyperParam["ν"]; isSample = true)
        if isCount
            ϕPlus .+= ϕ
        end

        # update estimates
        YFull = fullData(YEst)

        # update noise level
        a_τ_post = hyperParam["a_τ"] + 0.5 * prod(sz)
        estError = Y - YFull
        b_τ_post = hyperParam["b_τ"] + 0.5 * sum(estError .^ 2)
        τ = T(rand(Gamma(a_τ_post, 1 / b_τ_post), 1)[1])
        if isCount
            τPlus += τ
        end

        # compute loglikelihood
        loglike = logJointDist(Y, YFull, YEst, ϕ, U, δ, τ, hyperParam)
        append!(lossTot, loglike)

        if isCount
            ΣPlus += computeΣ(YEst)
        end

        # prune factors
        if opt.isPrune
            pruneFactors!(
                YEst, coresQPlus, ϕ, ϕPlus, δ, δPlus, 
                opt.trun; method = opt.pruneMethod
            )
            rank = YEst.rank
            rankest = copy(rank)
        else
            rankest = sum.(findPruneIndex(YEst, opt.trun; method = opt.pruneMethod))
        end
        U = computeγ(δ)

        if opt.isPrint > 0 && mod(epoch, opt.isPrint) == 0
            println("Epoch $epoch, loglikelihood is $loglike, rank estimate is $rankest.")
        end
    end

    YEst.coresQ = coresQPlus ./ gsCount

    post = Dict(
        "τ" => τPlus / gsCount,
        "δ" => δPlus / gsCount,
        "ϕ" => ϕPlus / gsCount,
    )

    return YEst, lossTot, post, ΣPlus / gsCount
end


"""
The Multiplicative Gamma Process Tensor Ring (MGP-TR) model.
Infer with the MCMC algorithm.

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
function TRLF_GS(
    Y :: Array{T},  # shape (p1, \dots, p_D, N)
    mask :: Array{T},
    rank :: Array{Int, 1},
    opt :: Optim
    ) where {T <: Real}
    @info("Gibbs Sampler for Multiplicative Gamma Process Tensor Ring")

    D = ndims(Y)
    sz = collect(size(Y))
    N = size(Y, D)  # sampler number

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

    # init core
    coresQPlus = Array{Array{T}, 1}(undef, D)
    for d in 1:D
        _, dn = round_index(d, D)
        coresQPlus[d] = zeros(T, rank[d], sz[d], rank[dn])
    end
    # init local shrinkage ϕ
    ϕ = Array{Array{T}, 1}(undef, D-1)
    ϕPlus = Array{Array{T}, 1}(undef, D-1)
    for d in 1:D-1
        _, dn = round_index(d, D)
        ϕ[d] = ones(T, rank[d], sz[d], rank[dn])
        ϕPlus[d] = zeros(T, rank[d], sz[d], rank[dn])
    end
    # init global shrinkage U and δ
    δ = Array{Array{T}, 1}(undef, D)
    δPlus = copy(δ)
    for d in 1:D
        δ[d] = ones(T, rank[d])
        δPlus[d] = zeros(T, rank[d])
    end
    U = computeγ(δ)
    # init noise
    τ = oneunit(T)  # todo: noise level, this can be set different
    τPlus = zero(T)
    ΣPlus = zeros(T, prod(sz) ÷ N, prod(sz) ÷ N)

    # inference
    rankest = copy(rank)
    lossTot = []
    gsCount = 0
    for epoch in 1:opt.iter
        isCount = epoch ≥ opt.burnin && mod(epoch - opt.burnin, opt.thin) ≡ 0
        # sample core tensor Q
        gsUpdateCoresQ!(YEst, Y, mask, ϕ, U, τ)
        # gsUpdateη!(YEst, Y, τ)
        if isCount
            coresQPlus .+= YEst.coresQ
            gsCount += 1
        end

        # sample hyperparameter δ
        updateδ!(
            δ, YEst, ϕ, U, hyperParam["a_δ1"],
            hyperParam["a_δ2"]; isSample = true
        )
        if isCount
            δPlus .+= δ
        end

        # update u
        U = computeγ(δ)

        # sample ϕ
        updateϕ!(ϕ, YEst, U, hyperParam["ν"]; isSample = true)
        if isCount
            ϕPlus .+= ϕ
        end

        # update estimates
        YFull = fullData(YEst)

        # update noise level
        a_τ_post = hyperParam["a_τ"] + 0.5 * sum(mask)
        estError = (Y - YFull) .* mask
        b_τ_post = hyperParam["b_τ"] + 0.5 * sum(estError .^ 2)
        τ = T(rand(Gamma(a_τ_post, 1 / b_τ_post), 1)[1])
        if isCount
            τPlus += τ
        end

        # compute loglikelihood
        loglike = logJointDist(Y, YFull, YEst, mask, ϕ, U, δ, τ, hyperParam)
        append!(lossTot, loglike)

        if isCount
            ΣPlus += computeΣ(YEst)
        end

        # prune factors
        if opt.isPrune
            pruneFactors!(
                YEst, coresQPlus, ϕ, ϕPlus, δ, δPlus, 
                opt.trun; method = opt.pruneMethod
            )
            rank = YEst.rank
            rankest = copy(rank)
        else
            rankest = sum.(findPruneIndex(YEst, opt.trun; method = opt.pruneMethod))
        end
        U = computeγ(δ)

        if opt.isPrint > 0 && mod(epoch, opt.isPrint) == 0
            println("Epoch $epoch, loglikelihood is $loglike, rank estimate is $rankest.")
        end
    end

    YEst.coresQ = coresQPlus ./ gsCount

    post = Dict(
        "τ" => τPlus / gsCount,
        "δ" => δPlus / gsCount,
        "ϕ" => ϕPlus / gsCount,
    )

    return YEst, lossTot, post, ΣPlus / gsCount
end


function gsUpdateCoresQ!(YEst :: TRLF{T}, Y, ϕ, U, τ) where {T <: Real}
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
        EvG2 = GNeqVec' * GNeqVec
        for i in 1:sz[d]
            VInv = τ * EvG2 + Diagonal(vec(ϕ[d][:, i, :])) .* UKron
            V = inv(VInv)
            
            μ = τ * V * vec(Γ[i, :])
            Qdi = safeMvNormal(μ, VInv)
            Qdi = reshape(Qdi, rank[d], rank[dn])
            Qd[:, i, :] = Qdi
        end
        YEst.coresQ[d] = Qd
    end
end


function gsUpdateCoresQ!(YEst :: TRLF{T}, Y, mask, ϕ, U, τ) where {T <: Real}
    rank = YEst.rank
    sz = YEst.shape
    D = YEst.ndim
    for d in 1:D
        _, dn = round_index(d, D)

        Yd = matricization(Y, d, true)
        Od = matricization(mask, d, true)
        Qd = zeros(rank[d], sz[d], rank[dn])

        GNeq = subchain(YEst, d)
        GNeqVec = reshape(
            permutedims(GNeq, [2, 3, 1]), size(GNeq, 2), :
        )
        UKron = kron(Diagonal(U[dn]), Diagonal(U[d]))
        for i in 1:sz[d]
            Γ = broadcast(*, vec(Yd[i, :]), GNeqVec)
            Γ = vec(sum(Γ, dims=1))

            EvG2 = GNeqVec[Od[i, :] .== 1.0, :]
            EvG2 = EvG2' * EvG2

            if d == D
                VInv = τ * EvG2 + I
            else
                VInv = τ * EvG2 + Diagonal(vec(ϕ[d][:, i, :])) .* UKron
            end
            V = inv(VInv)
            
            μ = τ * V * Γ
            Qdi = safeMvNormal(μ, VInv)
            Qdi = reshape(Qdi, rank[d], rank[dn])
            Qd[:, i, :] = Qdi
        end
        YEst.coresQ[d] = Qd
    end
end
