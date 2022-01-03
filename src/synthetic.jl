# generate binary shapes


function band end


function band(T, width)
    Σ = zeros(T, width, width)
    for i = 1:width, j = 1:width
        Σ[i, j] = T(exp(- (i - j) ^ 2 / (width ^ 2 / 10)))
    end

    return Σ * Σ' / width
end


function band(width)
    Σ = zeros(width, width)
    for i = 1:width, j = 1:width
        Σ[i, j] = exp(- (i - j) ^ 2 / (width ^ 2 / 10))
    end

    return Σ * Σ' / width
end


function ped end


function ped(T, width)
    Σ = zeros(T, width, width)
    for i = 1:width, j = 1:width
        Σ[i, j] = T(exp(- sin(π * abs(i - j) / (width / 4))^2))
    end

    return Σ * Σ' / width
end


function ped(width)
    Σ = zeros(width, width)
    for i = 1:width, j = 1:width
        Σ[i, j] = exp(- sin(π * abs(i - j) / (width / 4))^2)
    end

    return Σ * Σ' / width
end


function lin_band(T, FL)
    ΣTrue = band(T, Int(FL/4))
    ΣTrue = ΣTrue * ΣTrue' / FL
    Lin = zeros(T, 4, 4)
    for i = 1:4, j = 1:4
        Lin[i, j] = i * j / T(4.0)
    end
    ΣTrue = kron(Lin, ΣTrue)
    return ΣTrue
end


function lin_ped(T, FL)
    ΣTrue = ped(T, Int(FL/4))
    ΣTrue = ΣTrue * ΣTrue' / FL
    Lin = zeros(T, 4, 4)
    for i = 1:4, j = 1:4
        Lin[i, j] = i * j / T(4.0)
    end
    ΣTrue = kron(Lin, ΣTrue)
    return ΣTrue
end
