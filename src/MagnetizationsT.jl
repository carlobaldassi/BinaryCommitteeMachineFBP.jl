# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

using StatsFuns

primitive type MagT64 <: Mag64 64 end

f2mT(a::Float64) = f2m(MagT64, a)

include("AtanhErf.jl")
using .AtanhErf

const mInf = 30.0

magformat(::Type{MagT64}) = :tanh
parseinner(::Type{Val{:tanh}}, s::AbstractString) = mtanh(MagT64, parse(Float64, s))

convert(::Type{MagT64}, y::Float64) = f2mT(clamp(atanh(y), -mInf, mInf))
convert(::Type{Float64}, y::MagT64) = tanh(m2f(y))

MagT64(y::Float64) = convert(MagT64, y)

forcedmag(::Type{MagT64}, y::Float64) = f2mT(atanh(y))

mtanh(::Type{MagT64}, x::Float64) = f2mT(x)
atanh(x::MagT64) = m2f(x)

MagT64(pp::Real, pm::Real) = f2mT(clamp((log(pp) - log(pm)) / 2, -mInf, mInf))

isfinite(a::MagT64) = !isnan(m2f(a))

⊗(a::MagT64, b::MagT64) = f2mT(m2f(a) + m2f(b))
function ⊘(a::MagT64, b::MagT64)
    xa = m2f(a)
    xb = m2f(b)
    return f2mT(ifelse(xa == xb, 0.0, xa - xb)) # NOTE: the ifelse is for the Inf case
end

reinforce(m0::MagT64, γ::Float64) = f2mT(m2f(m0) * γ)

damp(newx::MagT64, oldx::MagT64, λ::Float64) = f2mT(m2f(newx) * (1 - λ) + m2f(oldx) * λ)

lr(x::Float64) = log1p(exp(-2abs(x)))
log2cosh(x::Float64) = abs(x) + lr(x)

function (*)(x::MagT64, y::MagT64)
    ax = m2f(x)
    ay = m2f(y)

    if ax ≥ ay && ax ≥ -ay
        t1 = 2ay
    elseif ax ≥ ay && ax < -ay
        t1 = -2ax
    elseif ax < ay && ax ≥ -ay
        t1 = 2ax
    else # ax < ay && ax < -ay
        t1 = -2ay
    end

    t2 = isinf(ax) || isinf(ay) ?
         0.0 : lr(ax + ay) - lr(ax - ay)

    return f2mT((t1 + t2) / 2)
end

merf(::Type{MagT64}, x::Float64) = f2mT(atanherf(x))

function auxmix(H::MagT64, a₊::Float64, a₋::Float64)
    aH = m2f(H)

    aH == 0.0 && return f2mT(0.0)

    xH₊ = aH + a₊
    xH₋ = aH + a₋

    # we need to compute
    #   t1 = abs(xH₊) - abs(a₊) - abs(xH₋) + abs(a₋)
    #   t2 = lr(xH₊) - lr(a₊) - lr(xH₋) + lr(a₋)
    # but we also need to take into account infinities
    if isinf(aH)
        if !isinf(a₊) && !isinf(a₋)
            t1 = sign(aH) * (a₊ - a₋) - abs(a₊) + abs(a₋)
            t2 = -lr(a₊) + lr(a₋)
        elseif isinf(a₊) && !isinf(a₋)
            if sign(a₊) == sign(aH)
                t1 = -sign(aH) * (a₋) + abs(a₋)
                t2 = lr(a₋)
            else
                t1 = -2mInf
                t2 = 0.0
            end
        elseif !isinf(a₊) && isinf(a₋)
            if sign(a₋) == sign(aH)
                t1 = sign(aH) * (a₊) - abs(a₊)
                t2 = -lr(a₊)
            else
                t1 = 2mInf
                t2 = 0.0
            end
        else # isinf(a₊) && isinf(a₋)
            if (sign(a₊) == sign(aH) && sign(a₋) == sign(aH)) || (sign(a₊) ≠ sign(aH) && sign(a₋) ≠ sign(aH))
                t1 = 0.0
                t2 = 0.0
            elseif sign(a₊) == sign(aH) # && sign(a₋) ≠ sign(aH)
                t1 = 2mInf
                t2 = 0.0
            else # sign(a₋) == sign(aH) && sign(a₊) ≠ sign(aH)
                t1 = -2mInf
                t2 = 0.0
            end
        end
    else # !isinf(aH)
        t1 = 0.0
        t1 += isinf(a₊) ? 0.0 : abs(xH₊) - abs(a₊)
        t1 -= isinf(a₋) ? 0.0 : abs(xH₋) - abs(a₋)
        t2 = lr(xH₊) - lr(a₊) - lr(xH₋) + lr(a₋)
    end

    return f2mT((t1 + t2) / 2)
end

exactmix(H::MagT64, p₊::MagT64, p₋::MagT64) = auxmix(H, m2f(p₊), m2f(p₋))

function erfmix(H::MagT64, m₊::Float64, m₋::Float64)
    aerf₊ = atanherf(m₊)
    aerf₋ = atanherf(m₋)
    return auxmix(H, aerf₊, aerf₋)
end

# log((1 + x * y) / 2)
function log1pxy(x::MagT64, y::MagT64)
    ax = m2f(x)
    ay = m2f(y)

    return !isinf(ax) && !isinf(ay) ? abs(ax + ay) - abs(ax) - abs(ay) + lr(ax + ay) - lr(ax) - lr(ay) :
            isinf(ax) && !isinf(ay) ? sign(ax) * ay - abs(ay) - lr(ay) :
           !isinf(ax) &&  isinf(ay) ? sign(ay) * ax - abs(ax) - lr(ax) :
           sign(ax) == sign(ay)     ? 0.0 : -Inf # isinf(ax) && isinf(ay)
end

# cross entropy with magnetizations:
#
# -(1 + x) / 2 * log((1 + y) / 2) - (1 - x) / 2 * log((1 - y) / 2)
#
# == -x * atanh(y) - log(1 - y^2) / 2 + log(2)
#
# with atanh's:
#
# == -ay * tanh(ax) + log(2cosh(ay))
function mcrossentropy(x::MagT64, y::MagT64)
    tx = tanh(m2f(x))
    ay = m2f(y)
    return !isinf(ay)          ? -abs(ay) * (sign0(ay) * tx - 1) + lr(ay) :
           sign(tx) ≠ sign(ay) ? Inf : 0.0
end

function logZ(u0::MagT64, u::Vector{MagT64})
    a0 = m2f(u0)
    if !isinf(a0)
        s1 = a0
        s2 = abs(a0)
        s3 = lr(a0)
        hasinf = 0
    else
        s1 = s2 = s3 = 0.0
        hasinf = sign(a0)
    end
    for ui in u
        ai = m2f(ui)
        if !isinf(ai)
            s1 += ai
            s2 += abs(ai)
            s3 += lr(ai)
        elseif hasinf == 0
            hasinf = sign(ai)
        elseif hasinf ≠ sign(ai)
            return -Inf
        end
    end
    return abs(s1) - s2 + lr(s1) - s3
end
