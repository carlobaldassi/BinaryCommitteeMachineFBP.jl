# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

module Magnetizations

export Mag64, MagT64, MagP64, mfill, mflatp, mrand, damp, reinforce, ⊗, ⊘, ↑, sign0,
       merf, exactmix, erfmix, mtanh, log1pxy, mcrossentropy,
       logZ, forcedmag, showinner, parseinner, magformat,
       conv_diff

using SpecialFunctions
using Compat

import Base: convert, promote_rule, *, /, +, -, sign, signbit, isnan,
             show, showcompact, abs, isfinite, isless, copysign,
             atanh, zero

abstract type Mag64 end

@inline m2f(a::Mag64) = reinterpret(Float64, a)
@inline f2m(::Type{F}, a::Float64) where {F<:Mag64} = reinterpret(F, a)

convert(::Type{T}, y::Mag64) where {T<:Real} = convert(T, Float64(y))
convert(::Type{F}, y::Real) where {F<:Mag64} = convert(F, Float64(y))

convert(::Type{F}, x::F) where {F<:Mag64} = x
convert(::Type{F}, x::Mag64) where {F<:Mag64} = F(Float64(x))

Mag64(::Type{F}, pp::Real, pm::Real) where {F<:Mag64} = F(pp, pm)

promote_rule(::Type{<:Mag64}, ::Type{Float64}) = Float64

zero(::Type{F}) where {F<:Mag64} = f2m(F, 0.0)

isnan(a::Mag64) = isnan(m2f(a))

abs(a::F) where {F<:Mag64} = f2m(F, abs(m2f(a)))
copysign(x::F, y::Float64) where {F<:Mag64} = f2m(F, copysign(m2f(x), y))
copysign(x::F, y::Mag64) where {F<:Mag64} = f2m(F, copysign(m2f(x), m2f(y)))

⊗(a::F, b::Float64) where {F<:Mag64} = a ⊗ F(b)
⊘(a::F, b::Float64) where {F<:Mag64} = a ⊘ F(b)

⊗(a::Float64, b::Mag64) = b ⊗ a


(*)(a::Mag64, b::Real) = Float64(a) * b
(*)(a::Real, b::Mag64) = b * a

(+)(a::Mag64, b::Real) = Float64(a) + b
(+)(a::Real, b::Mag64) = b + a

(-)(a::Mag64, b::Real) = Float64(a) - b
(-)(a::Real, b::Mag64) = -(b - a)
(-)(a::F) where {F<:Mag64} = f2m(F, -m2f(a))

(-)(a::Mag64, b::Mag64) = Float64(a) - Float64(b)

sign(a::Mag64) = sign(m2f(a))
signbit(a::Mag64) = signbit(m2f(a))
sign0(a::Union{Mag64,Real}) = (1 - 2signbit(a))

show(io::IO, a::Mag64) = show(io, Float64(a))
showcompact(io::IO, a::Mag64) = showcompact(io, Float64(a))
showinner(io::IO, a::Mag64) = show(io, m2f(a))

mfill(::Type{F}, x::Float64, n::Int) where {F<:Mag64} = F[F(x) for i = 1:n]
mflatp(::Type{F}, n::Int) where {F<:Mag64} = mfill(F, 0.0, n)

mrand(::Type{F}, x::Float64, n::Int) where {F<:Mag64} = F[F(x * (2*rand()-1)) for i = 1:n]

reinforce(m::Mag64, m0::Mag64, γ::Float64) = m ⊗ reinforce(m0, γ)

damp(newx::Float64, oldx::Float64, λ::Float64) = newx * (1 - λ) + oldx * λ

Base.:(==)(a::F, b::F) where {F<:Mag64} = (m2f(a) == m2f(b))
Base.:(==)(a::Mag64, b::Float64) = (Float64(a) == b)
Base.:(==)(a::Float64, b::Mag64) = (b == a)

isless(m::Mag64, x::Real) = isless(Float64(m), x)
isless(x::Real, m::Mag64) = isless(x, Float64(m))

function erfmix(H::Float64, m₊::Float64, m₋::Float64)
    erf₊ = erf(m₊)
    erf₋ = erf(m₋)
    return H * (erf₊ - erf₋) / (2 + H * (erf₊ + erf₋))
end

logZ(u::Vector{F}) where {F<:Mag64} = logZ(zero(F), u)

↑(m::F, x::Real) where {F<:Mag64} = mtanh(F, x * atanh(m))

conv_diff(x::F, y::F) where {F<:Mag64} = abs(m2f(x) - m2f(y))

include("MagnetizationsP.jl")
include("MagnetizationsT.jl")

end # module
