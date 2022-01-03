#=
Functions about TR format, constracting the core tensors
=#

using LinearAlgebra
using TensorOperations

include("tensor_basics.jl")


"""
Compute the full array given core tensors.
If isConnect, then constracting the last and fist indices,
which corresponds to TR format.
If not, equals to TT format. This extensively appears in the computing of subchains.
"""
function fullCores(cores; isConnect :: Bool = true)

    D = length(cores)
    if isConnect
        einsumIndex = Array{Array{Int}, 1}(undef, D)
        einsumIndex[1] = [1, -1, 2]
        for d in 2:D-1
            einsumIndex[d] = [d, -d, d+1]
        end
        einsumIndex[D] = [D, -D, 1]
    else
        einsumIndex = Array{Array{Int}, 1}(undef, D)
        einsumIndex[1] = [-1, -2, 1]
        for d in 2:D-1
            einsumIndex[d] = [d-1, -(d+1), d]
        end
        einsumIndex[D] = [D-1, -(D+1), -(D+2)]
    end

    X = @ncon(cores, einsumIndex)
    return X
end


raw"""
Compute the left subchain of MPS.
For TR format, `X = G^{\leq d}`. If you want to compute `G^{< d}`, just set `d := d + 1`.
"""
function subchainLeft(cores, d :: Int; squeeze :: Bool = true)

    if d == 0
        return [one(cores[1][1])]
    elseif d == 1
        return cores[1]
    else
        cores_ = cores[1:d]
        X = fullCores(cores_; isConnect = false)

        sz = 1
        for i in 1:d
            sz *= size(cores[i], 2)
        end
        r1 = size(cores[1], 1)
        r2 = size(cores[d], 3)
        if squeeze
            new_sz = [r1, sz, r2]
            X = reshape(X, new_sz...)
        end
        return X
    end
end

raw"""
Compute the right subchain of MPS.
For TR format, `X = G^{\geq d}`. If you want to compute `G^{> d}`, just set d := d + 1.
If squeeze is true, then return an order-3 tensor of shape `r_{k} \times prod(sz) r_{D}`,
else, return an order-(D - d + 1) tensor of shape `r_{k} \times sz_{k} ... \times r_{D}`.
"""
function subchainRight(cores, d :: Int; squeeze :: Bool = true)

    nDim = length(cores)
    if d == nDim + 1
        return [one(cores[1][1])]
    elseif d == nDim
        return cores[end]
    else
        cores_ = cores[d:end]
        X = fullCores(cores_; isConnect = false)

        sz = 1
        for i in d:nDim
            sz *= size(cores[i], 2)
        end
        r1 = size(cores[d], 1)
        r2 = size(cores[end], 3)
        if squeeze
            new_sz = [r1, sz, r2]
            X = reshape(X, new_sz...)
        end
        return X
    end
end

function subchainSkip(cores, d :: Int; squeeze :: Bool = true)

    nDim = length(cores)
    sz = zeros(Int, nDim)
    rank = zeros(Int, nDim)
    for i in 1:nDim
        sz[i] = size(cores[i], 2)
        rank[i] = size(cores[i], 1)
    end

    cores_ = circshift(cores, -d)
    pop!(cores_)
    X = fullCores(cores_; isConnect = false)
    if squeeze
        _, dn = round_index(d, nDim)
        new_sz = [rank[dn], prod(sz) ÷ sz[d], rank[d]]
        X = reshape(X, new_sz...)
    end
    return X
end
