# Define some basic operations in tensor

function matricization(X :: Array{T}, d :: Int, isPermute :: Bool = false) where {T <: Real}
    # Input: X of shape N_1 \times \cdots \times N_D
    # Output: X_d of shape N_d \times numel(X) / N_d
    nd = ndims(X)
    if isPermute == false
        perm = Array(1:nd)
        splice!(perm, d)
        insert!(perm, 1, d)

        Xd = permutedims(X, perm)
        Xd = reshape(Xd, size(X, d), :)
        return Xd
    else
        # perm = [Array(d : nd)..., Array(1 : d - 1)...]
        perm = circshift(1:nd, -(d - 1))
        Xd = permutedims(X, perm)
        Xd = reshape(Xd, size(X, d), :)
        return Xd
    end
end

function unfolding(X :: Array{T}, d :: Int) where {T <: Real}
    sz = size(X)
    sz1 = Int(prod(sz[1:d]))
    sz2 = Int(prod(sz) / sz1)
    Xd = reshape(X, sz1, sz2)
    return Xd
end

function unMatricization(X :: Array{T}, d :: Int,
    sz :: Array{Int}, isPermute :: Bool = false
    ) where {T <: Real}
    nd = length(sz)
    if isPermute == false
        sz_d = splice!(sz, d)
        insert!(sz, 1, sz_d)

        X_ = reshape(X, sz...)
        perm = [Array(2:d)..., 1, Array(d+1:nd)...]
        X_ = permutedims(X_, perm)
        return X_
    else
        sz_ = copy(sz)
        sz = [sz_[d:end]..., sz_[1:d-1]...]
        X_ = reshape(X, sz...)
        perm = [Array(d:nd)..., Array(1:d-1)...]
        X_ = permutedims(X_, perm)
        return X_
    end
end

function folding(X :: Array{T}, sz :: Array{Int}) where {T <: Real}
    x_fold = reshape(X, sz...)
    return x_fold
end

function round_index(d :: Int, D :: Int)
    if d == 1
        dp = D
    else
        dp = d - 1
    end

    if d == D
        dn = 1
    else
        dn = d + 1
    end
    return [dp, dn]
end
