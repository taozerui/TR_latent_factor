using Parameters


# Tensor Ring Latent Factor
@with_kw mutable struct TRLF{T}

    shape :: Array{Int}  # shape of features, p1, ..., pD, N
    rank :: Array{Int}  # TR-rank

    @assert length(shape) == length(rank)

    ndim :: Int = length(shape)
    # core tensors, typical shape (R_d, I_d, R_{d+1}), the last core tensor is η
    coresQ :: Array{Array{T}, 1} = Array{Array{T}, 1}(undef, ndim)

end


@with_kw mutable struct Optim{T}
    iter :: Int  # main iteration steps
    burnin :: Int = Int(round(0.6 * iter))  # iteration steps for mcmc averaging
    thin :: Int = 5 # thin step
    init :: String = "rand"   # init method, SVD, rand or randn
    initScale :: T = 0.5  # init scale
    ϵ :: T = 1e-3  # iteration tolerance
    trun :: T = eps(T)  # truncation level for factors
    isPrune :: Bool = true  # prune irrelevant factors or not
    pruneMethod :: String = "absolute"  # prune method, absolute or relative
    isPruneFactor :: Bool = true  # prune factor or not
    isPrint :: Int = 100  # print loss
    rotate :: Int = 10
    ε :: T = 1e-6
end
