# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

@compat primitive type MagP64 <: Mag64 64 end

f2mP(a::Float64) = f2m(MagP64, a)

magformat(::Type{MagP64}) = :plain
parseinner(::Type{Val{:plain}}, s::AbstractString) = MagP64(parse(Float64, s))

convert(::Type{MagP64}, y::Float64) = f2mP(y)
convert(::Type{Float64}, y::MagP64) = m2f(y)

forcedmag(::Type{MagP64}, y::Float64) = MagP64(y)

mtanh(::Type{MagP64}, x::Float64) = f2mP(tanh(x))
atanh(x::MagP64) = atanh(m2f(x))

MagP64(pp::Real, pm::Real) = MagP64((pp - pm) / (pp + pm))

isfinite(a::MagP64) = isfinite(m2f(a))

function ⊗(a::MagP64, b::MagP64)
    xa = m2f(a)
    xb = m2f(b)
    return f2mP(clamp((xa + xb) / (1 + xa * xb), -1, 1))
end
function ⊘(a::MagP64, b::MagP64)
    xa = m2f(a)
    xb = m2f(b)
    return f2mP(xa == xb ? 0.0 : clamp((xa - xb) / (1 - xa * xb), -1, 1))
end

reinforce(m0::MagP64, γ::Float64) = MagP64(tanh(atanh(m2f(m0)) * γ))

damp(newx::MagP64, oldx::MagP64, λ::Float64) = f2mP(m2f(newx) * (1 - λ) + m2f(oldx) * λ)

(*)(x::MagP64, y::MagP64) = MagP64(Float64(x) * Float64(y))

merf(::Type{MagP64}, x::Float64) = f2mP(erf(x))

function exactmix(H::MagP64, p₊::MagP64, p₋::MagP64)
    vH = m2f(H)
    pd = (m2f(p₊) + m2f(p₋)) / 2
    pz = (m2f(p₊) - m2f(p₋)) / 2

    return f2mP(pz * vH / (1 + pd * vH))
end

erfmix(H::MagP64, m₊::Float64, m₋::Float64) = MagP64(erfmix(Float64(H), m₊, m₋))

log1pxy(x::MagP64, y::MagP64) = log((1 + Float64(x) * Float64(y)) / 2)

# cross entropy with magnetizations:
#
# -(1 + x) / 2 * log((1 + y) / 2) + -(1 - x) / 2 * log((1 - y) / 2)
#
# == -x * atanh(y) - log(1 - y^2) / 2 + log(2)
function mcrossentropy(x::MagP64, y::MagP64)
    fx = m2f(x)
    fy = m2f(y)
    return -fx * atanh(fy) - log(1 - fy^2) / 2 + log(2)
end

logmag2pp(x::MagP64) = log((1 + m2f(x)) / 2)
logmag2pm(x::MagP64) = log((1 - m2f(x)) / 2)

function logZ(u0::MagP64, u::Vector{MagP64})
    zkip = logmag2pp(u0)
    zkim = logmag2pm(u0)
    for ui in u
        zkip += logmag2pp(ui)
        zkim += logmag2pm(ui)
    end
    zki = exp(zkip) + exp(zkim)
    return log(zki)
end
