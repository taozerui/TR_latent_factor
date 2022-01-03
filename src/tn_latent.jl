#=
Tensor Ring Latent Factor
=#

using LinearAlgebra
using TensorOperations
using OMEinsum

include("type.jl")
include("tn_tools.jl")
include("tensor_basics.jl")

import Base.length
import Base.ndims
import Base.size


"""
Compute parameter number
"""
function length(tn :: TRLF{T}) where {T <: Real}
    return sum(length.(tn.coresQ))
end


"""
Dimension of TRLF
"""
function ndims(tn :: TRLF{T}) where {T <: Real}
    return tn.ndim
end


"""
Size of TRLF
"""
function size(tn :: TRLF{T}) where {T <: Real}
    return tn.shape
end

function size(tn :: TRLF{T}, d :: Int) where {T <: Real}
    return tn.shape[d]
end


"""
Init the core tensors and latent factors.
"""
function init!(
    tn :: TRLF{T};
    method :: String = "randn",
    scale :: T = T(0.5)
    ) where {T <: Real}

    for d in 1:tn.ndim
        _, dn = round_index(d, tn.ndim)
        if method == "randn"
            Gi = randn(T, tn.rank[d], tn.shape[d], tn.rank[dn])
        elseif method == "rand"
            Gi = rand(T, tn.rank[d], tn.shape[d], tn.rank[dn])
        else
            @error "Wrong initialize method!"
        end

        tn.coresQ[d] = Gi * scale
    end

end


# todo
function fromList!(tn :: TRLF{T}, coresQ :: Array) where {T <: Real}
    tn.coresQ = coresQ
end


"""
Compute the full array of TR format.
Using `ncon` function to constract nearby indices.
"""
function fullData(tn :: TRLF{T}) where {T <: Real}
    return fullCores(tn.coresQ)
end


"""
Compute the coeffients,
shape ` p_1 \\times \\dots p_D \\times k `
"""
function computeCoef(tn :: TRLF{T}) where {T <: Real}

    D = tn.ndim
    return fullCores(tn.coresQ[1:D-1]; isConnect = false)
end


"""
Contract subchain
"""
function subchain(tn :: TRLF{T}, d :: Int) where {T <: Real}
    return subchainSkip(tn.coresQ, d)
end


"""
Contract left chain X = G^{d ⩽ d}
"""
function leftchain(tn :: TRLF{T}, d :: Int) where {T <: Real}
    return subchainLeft(tn.coresQ, d)
end


"""
Contract right chain X = G^{d ⩾ d}
"""
function rightchain(tn :: TRLF{T}, d :: Int) where {T <: Real}
    # cores = computeCores(tn.coresQ, tn.coresR)
    return subchainRight(tn.coresQ, d)
end


"""
Compute the data variance
"""
function computeΣ(tn :: TRLF{T}) where {T <: Real}
    W = computeCoef(tn)
    W = reshape(W, size(W, 1), :, size(W, ndims(W)))
    W2 = matricization(W, 2, false)
    Σ = W2 * W2'

    return Σ
end
