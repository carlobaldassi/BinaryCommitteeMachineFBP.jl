# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

module BinaryCommitteeMachineFBP

export focusingBP, MagT64, MagP64,
       read_messages, write_messages,
       FocusingProtocol, StandardReinforcement, Scoping, PseudoReinforcement, FreeScoping

using StatsFuns
using GZip
using ExtractMacro
using Iterators
using Compat

include("Magnetizations.jl")
using .Magnetizations

include("Util.jl")
using .Util

immutable Messages{F<:Mag64}
    M::Int
    N::Int
    K::Int

    # 2 layers:
    #   0  --> external fields
    #   1  --> theta (perceptron) nodes
    #   2  --> theta (perceptron) nodes
    #
    # notation:
    #   m* --> total fields (includes external fields)
    #   u* --> messages directed down (node->variable)
    #   U* --> messages directed up (node->variable)
    #
    # variable names (quantity):
    #   w  --> weights (NxK)
    #   τ1 --> outputs from the first layer of perceptrons (KxM)
    #          also inputs to the second layer
    #   τ2 --> outputs from the second layer of perceptrons (M)
    #
    #                 DEPTH
    ux::MagVec2{F}    # 0
    mw::MagVec2{F}    # 0+1
    mτ1::MagVec2{F}   # 1+2
    uw::MagVec3{F}    # 1
    Uτ1::MagVec2{F}   # 1
    mτ2::MagVec{F}    # 2+
    uτ1::MagVec2{F}   # 2

    function Messages(M::Int, N::Int, K::Int, ux::MagVec2{F}, mw::MagVec2{F}, mτ1::MagVec2{F}, uw::MagVec3{F},
                      Uτ1::MagVec2{F}, mτ2::MagVec{F}, uτ1::MagVec2{F}; check::Bool = true)
        if check
            checkdims(ux, K, N)
            checkdims(mw, K, N)
            checkdims(mτ1, M, K)
            checkdims(uw, M, K, N)
            checkdims(Uτ1, M, K)
            checkdims(mτ2, M)
            checkdims(uτ1, M, K)
        end
        new(M, N, K, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1)
    end

end

function Messages{F<:Mag64}(::Type{F}, M::Integer, N::Integer, K::Integer, x::Real)
    ux = [mflatp(F, N) for k = 1:K]
    mw = [mflatp(F, N) for k = 1:K]
    mτ1 = [mflatp(F, K) for a = 1:M]
    uw = [MagVec{F}[map(F, x*(2*rand(N)-1)) for k = 1:K] for a = 1:M]
    Uτ1 = [mflatp(F, K) for a = 1:M]
    mτ2 = mflatp(F, M)
    uτ1 = [mflatp(F, K) for a = 1:M]

    for k = 1:K, i = 1:N, a = 1:M
        mw[k][i] = mw[k][i] ⊗ uw[a][k][i]
    end
    for a = 1:M, k = 1:K
        mτ1[a][k] = mτ1[a][k] ⊗ Uτ1[a][k] ⊗ uτ1[a][k]
    end

    return Messages{F}(M, N, K, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1, check=false)
end

Messages{F<:Mag64}(::Type{F}, messages::Messages{F}) = messages
function Messages{F<:Mag64}(::Type{F}, messages::Messages)
    @extract messages : M N K ux mw mτ1 uw Uτ1 mτ2 uτ1
    return Messages{F}(M, N, K,
                       chgeltype(ux, F), chgeltype(mw, F), chgeltype(mτ1, F), chgeltype(uw, F),
                       chgeltype(Uτ1, F), chgeltype(mτ2, F), chgeltype(uτ1, F),
                       check=false)
end

function read_messages{F<:Mag64}(io::IO, ::Type{F})
    l = split(readline(io))
    length(l) == 2 && l[1] == "fmt:" || error("invalid messages file")
    fmt = Val{Symbol(l[2])}
    l = split(readline(io))
    length(l) == 4 && l[1] == "N,K,M:" || error("invalid messgaes file")
    N, K, M = parse(Int, l[2]), parse(Int, l[3]), parse(Int, l[4])

    ux = [mflatp(F, N) for k = 1:K]
    mw = [mflatp(F, N) for k = 1:K]
    mτ1 = [mflatp(F, K) for a = 1:M]
    uw = [[mflatp(F, N) for k = 1:K] for a = 1:M]
    Uτ1 = [mflatp(F, K) for a = 1:M]
    mτ2 = mflatp(F, M)
    uτ1 = [mflatp(F, K) for a = 1:M]

    expected_lines = K + M + M*K + M + 1 + M + K
    for (i,l) in enumerate(eachline(io))
        i > expected_lines && (strip(l) == "END" || error("invalid messages file"); break)
        @readmagvec(l, fmt, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1)
    end
    eof(io) || error("invalid messages file")
    return Messages{F}(M, N, K, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1, check=false)
end

"""
    read_messages(filename, mag_type)

Reads messages from a file. `mag_type` is the internal storage format used in the resulting `Messages` object,
it can be either `MagT64` (uses tanhs, accurate but slower) or `MagP64` (plain format, faster but inaccurate).

The file format is the one produced by [`write_messages`](@ref).
"""
read_messages{F<:Mag64}(filename::AbstractString, ::Type{F}) = gzopen(io->read_messages(io, F), filename, "r")


"""
    write_messages(filename, messages)

Writes messages to a file. The messages can be read back with [`read_messages`](@ref). Note that the output
is a plain text file compressed with gzip.
"""
function write_messages(filename::AbstractString, messages::Messages)
    gzopen(filename, "w") do f
        write_messages(f, messages)
    end
end

function write_messages{F<:Mag64}(io::IO, messages::Messages{F})
    @extract messages : N K M ux mw mτ1 uw Uτ1 mτ2 uτ1

    println(io, "fmt: ", magformat(F))
    println(io, "N,K,M: $N $K $M")
    @dumpmagvecs(io, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1)
    println(io, "END")
end

Base.eltype{F<:Mag64}(messages::Messages{F}) = F

function Base.copy!{F<:Mag64}(dest::Messages{F}, src::Messages{F})
    dest.N == src.N || throw(ArgumentError("incompatible arguments: dest.N=$(dest.N) src.N=$(src.N)"))
    dest.K == src.K || throw(ArgumentError("incompatible arguments: dest.K=$(dest.K) src.K=$(src.K)"))
    dest.M == src.M || throw(ArgumentError("incompatible arguments: dest.M=$(dest.M) src.M=$(src.M)"))
    for k = 1:dest.K
        copy!(dest.ux[k], src.ux[k])
        copy!(dest.mw[k], src.mw[k])
    end
    for a = 1:dest.M, k = 1:dest.K
        copy!(dest.uw[a][k], src.uw[a][k])
    end
    for a = 1:dest.M
        copy!(dest.mτ1[a], src.mτ1[a])
        copy!(dest.Uτ1[a], src.Uτ1[a])
        copy!(dest.uτ1[a], src.uτ1[a])
    end
    copy!(dest.mτ2, src.mτ2)
    return dest
end

function set_outfields!{F<:Mag64}(messages::Messages{F}, output::Vector, β::Float64)
    @extract messages : N K M mτ2
    @assert length(output) == M
    t = tanh(β / 2)
    for a = 1:M
        mτ2[a] = forcedmag(F, output[a] * t) # forced avoids clamping
    end
end

print_mags(messages::Messages) = print_mags(STDOUT, messages)
function print_mags(io::IO, messages::Messages)
    @extract messages : N K mw
    for k = 1:K, i = 1:N
        @printf(io, "%i %i %.15f\n", k, i, Float64(mw[k][i]))
    end
end

type Params{F<:Mag64}
    damping::Float64
    ϵ::Float64
    β::Float64
    max_iters::Int
    accuracy1::Symbol
    accuracy2::Symbol
    r::Float64
    pol::F
    dγ::Float64
    quiet::Bool
end

immutable Patterns
    M::Int
    X::Vec2
    output::IVec
    function Patterns(X::AbstractVector, output::AbstractVector)
        M = length(X)
        length(output) == M || throw(ArgumentError("incompatible lengths of inputs and outputs: $M vs $(length(output))"))
        all(ξ->all(ξi->abs(ξi) == 1, ξ), X) || throw(ArgumentError("inputs must be ∈ {-1,1}"))
        all(o->abs(o) == 1, output) || throw(ArgumentError("outputs must be ∈ {-1,1}"))
        new(M, X, output)
    end
end

Patterns(Xo::Tuple{Vec2,Vec}) = Patterns(Xo...)

Patterns(NM::Tuple{Integer,Integer}) = ((N,M) = NM; Patterns([rand(-1.0:2.0:1.0, N) for a = 1:M], ones(M)))
function Patterns(NM::Tuple{Integer,Integer}; teacher::Union{Bool,Int,Vec2} = false)
    N, M = NM
    X = [rand(-1.0:2.0:1.0, N) for a = 1:M]
    if teacher == false
        output = ones(M)
    else
        if !isa(teacher, Vec2)
            K::Int = isa(teacher, Bool) ? 1 : teacher
            tw = [rand(-1.0:2.0:1.0, N) for k = 1:K]
        else
            K = length(teacher)
            K > 0 || throw(ArgumentError("empty teacher"))
            tw = teacher
            all(w -> length(w) == N, tw) || throw(ArgumentError("invalid teacher length, expected $N, given: $(sort!(unique(collect(w->length(w) for w in tw))))"))
            all(w -> all(x -> abs(x) == 1, w), tw) || throw(ArgumentError("invalid teacher, entries must be all ±1"))
        end
        r0 = Array(Float64, K)
        output = Float64[transf1!(r0, tw, ξ) for ξ in X]
        @assert all(o->abs(o) == 1, output)
    end
    Patterns(X, output)
end
Patterns(patterns::Patterns) = deepcopy(patterns)

function Patterns(patternsfile::AbstractString)
    X = Vec[]
    N = 0
    M = 0
    gzopen(patternsfile) do f
        M = 0
        for l in eachline(f)
            push!(X, map(float, split(l)))
            M += 1
        end
    end
    o = Int[1.0 for a = 1:M]

    return Patterns(X,o)
end

function computeσ²(w::Vec)
    σ² = 0.0
    @inbounds for wi in w
        σ² += (1 - wi^2)
    end
    return σ²
end

function computeσ²(w::Vec, ξ::Vec)
    σ² = 0.0
    @inbounds @itr for (wi,ξi) in zip(w,ξ)
        σ² += (1 - wi^2) * ξi^2
    end
    return σ²
end

computeσ(σ²::Float64) = √(2σ²)
computeσ(w::Vec) = √(2computeσ²(w))
computeσ(w::Vec, ξ::Vec) = √(2computeσ²(w, ξ))

transf0(w::Vec, ξ::Vec) = transf0(w, computeσ(w, ξ), ξ)
transf0(w::Vec, σ::Float64, ξ::Vec) = erf(dot(ξ, w) / σ)

function transf0!(r::Vec, w1::Vec2, ξ::Vec)
    @itr for (i,w) in enumerate(w1)
        r[i] = transf0(w, ξ)
    end
    return r
end

gauss(x, dσ²) = e^(-x^2 / dσ²) / √(π * dσ²)
gauss(x) = e^(-x^2 / 2) / √(2π)

function subfield!(h::MagVec, m::MagVec, u::MagVec)
    @inbounds for i = 1:length(m)
        h[i] = m[i] ⊘ u[i]
    end
end

let hsT = Dict{Int,MagVec{MagT64}}(), hsP = Dict{Int,MagVec{MagP64}}(), vhs = Dict{Int,Vec}(),
    vHs = Dict{Int,Vec}(), leftCs = Dict{Int,Vec2}(), rightCs = Dict{Int,Vec2}()

    geth(::Type{MagT64}, N::Int) = Base.@get!(hsT, N, Array(MagT64, N))
    geth(::Type{MagP64}, N::Int) = Base.@get!(hsP, N, Array(MagP64, N))

    global theta_node_update_approx!
    function theta_node_update_approx!{F<:Mag64}(m::MagVec{F}, M::F, ξ::Vec, u::MagVec{F}, U::F, params::Params)
        @extract params : λ=damping

        N = length(m)
        h::MagVec{F} = geth(F, N)
        vh = Base.@get!(vhs, N, Array(Float64, N))

        subfield!(h, m, u)
        H = M ⊘ U

        @inbounds for i = 1:N
            vh[i] = h[i]
        end


        vH = Float64(H)
        σ² = computeσ²(vh, ξ)

        μ = dot(vh, ξ)

        dσ² = 2σ²
        newU = merf(F, μ / √dσ²)

        maxdiff = abs(U - newU)
        U = damp(newU, U, λ)
        newM = H ⊗ U
        M = newM

        g = gauss(μ, dσ²)

        p0 = 2vH * g / (1 + vH * U)

        pμ = p0 * (p0 + μ / σ²)

        pσ = p0 * (1 - μ / σ² - μ * p0) / dσ²

        @inbounds for i = 1:N
            ξi = ξ[i]
            hi = vh[i]
            newu = convert(F, clamp(ξi * (p0 + ξi * (hi * pμ + ξi * (1-hi^2) * pσ)), -1+eps(-1.0), 1-eps(1.0))) # use mag-functions?
            maxdiff = max(maxdiff, abs(newu - u[i]))
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]
        end

        return maxdiff, U, M
    end

    global theta_node_update_accurate!
    function theta_node_update_accurate!{F<:Mag64}(m::MagVec{F}, M::F, ξ::Vec, u::MagVec{F}, U::F, params::Params)
        @extract params : λ=damping

        N = length(m)
        h::MagVec{F} = geth(F, N)
        vh = Base.@get!(vhs, N, Array(Float64, N))

        subfield!(h, m, u)
        H = M ⊘ U

        @inbounds for i = 1:N
            vh[i] = h[i]
        end

        σ² = computeσ²(vh, ξ)
        μ = dot(vh, ξ)

        dσ² = 2σ²
        newU = merf(F, μ / √dσ²)

        maxdiff = 0.0
        U = damp(newU, U, λ)
        M = H ⊗ U

        @inbounds for i = 1:N
            ξi = ξ[i]
            hi = vh[i]
            μ̄ = μ - ξi * hi
            σ̄² = σ² - (1-hi^2) * ξi^2
            sdσ̄² = √(2σ̄²)
            m₊ = (μ̄ + ξi) / sdσ̄²
            m₋ = (μ̄ - ξi) / sdσ̄²
            newu = erfmix(H, m₊, m₋)
            maxdiff = max(maxdiff, abs(newu - u[i]))
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]
        end
        return maxdiff, U, M
    end

    global theta_node_update_exact!
    function theta_node_update_exact!{F<:Mag64}(m::MagVec{F}, M::F, ξ::Vec, u::MagVec{F}, U::F, params::Params)
        @extract params : λ=damping

        N = length(m)
        h::MagVec{F} = geth(F, N)
        vh = Base.@get!(vhs, N, Array(Float64, N))
        leftC = Base.@get!(leftCs, N, [zeros(i+1) for i = 1:N])
        rightC = Base.@get!(rightCs, N, [zeros((N-i+1)+1) for i = 1:N])

        subfield!(h, m, u)
        H = M ⊘ U

        @inbounds for i = 1:N
            vh[i] = h[i]
        end

        leftC[1][1] = (1-ξ[1]*vh[1])/2
        leftC[1][2] = (1+ξ[1]*vh[1])/2
        for i = 2:N
            lC0 = leftC[i-1]
            lC = leftC[i]
            hi = ξ[i] * vh[i]
            lC[1] = lC0[1] * (1-hi)/2
            for j = 2:i
                lC[j] = lC0[j-1] * (1+hi)/2 + lC0[j] * (1-hi)/2
            end
            lC[end] = lC0[end] * (1+hi)/2
        end

        rightC[end][1] = (1-ξ[end]*vh[end])/2
        rightC[end][2] = (1+ξ[end]*vh[end])/2
        for i = (N-1):-1:1
            rC0 = rightC[i+1]
            rC = rightC[i]
            hi = ξ[i] * vh[i]
            rC[1] = rC0[1] * (1-hi)/2
            for j = 2:(N-i+1)
                rC[j] = rC0[j-1] * (1+hi)/2 + rC0[j] * (1-hi)/2
            end
            rC[end] = rC0[end] * (1+hi)/2
        end

        @compat @assert maximum(abs.(leftC[end] .- rightC[1])) ≤ 1e-10 (leftC[end], rightC[1])

        @assert isodd(N)
        z = (N+1) ÷ 2
        pm = sum(rightC[1][1:z])
        pp = sum(rightC[1][(z+1):end])

        newU = Mag64(F, pp, pm)

        @assert isfinite(newU)

        maxdiff = 0.0
        U = damp(newU, U, λ)
        newM = H ⊗ U
        M = newM

        @assert isfinite(newM) (H, U)

        u1 = ones(1)

        @inbounds for i = 1:N
            ξi = ξ[i]
            @assert ξi^2 == 1

            lC = i > 1 ? leftC[i-1] : u1
            rC = i < N ? rightC[i+1] : u1

            pm = 0.0
            pz = 0.0
            pp = 0.0
            for j = 1:N
                p = 0.0
                for k = max(1,j+i-N):min(j,i)
                    p += lC[k] * rC[j-k+1]
                end
                if j < z
                    pm += p
                elseif j == z
                    pz = p
                else
                    pp += p
                end
            end

            mp = convert(F, clamp(pp + ξi * pz - pm, -1.0, 1.0))
            mm = convert(F, clamp(pp - ξi * pz - pm, -1.0, 1.0))
            newu = exactmix(H, mp, mm)

            maxdiff = max(maxdiff, abs(newu - u[i]))
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]

            @assert isfinite(u[i]) (u[i],)
        end
        return maxdiff, U, M
    end

    global free_energy_theta
    function free_energy_theta{F<:Mag64}(m::MagVec{F}, M::F, ξ::Vec, u::MagVec{F}, U::F)
        N = length(m)
        h::MagVec{F} = geth(F, N)
        vh = Base.@get!(vhs, N, Array(Float64, N))

        f = 0.0

        subfield!(h, m, u)
        H = M ⊘ U

        @inbounds for i = 1:N
            vh[i] = h[i]
        end

        σ = computeσ(vh, ξ)
        μ = dot(vh, ξ)

        b = merf(F, μ / σ)

        f -= log1pxy(H, b)
        @assert isfinite(f)

        for i = 1:N
            f += log1pxy(h[i], u[i])
        end
        return f
    end

    global free_energy_theta_exact
    function free_energy_theta_exact{F<:Mag64}(m::MagVec{F}, M::F, ξ::Vec, u::MagVec{F}, U::F)

        N = length(m)
        h::MagVec{F} = geth(F, N)
        vh = Base.@get!(vhs, N, Array(Float64, N))
        leftC = Base.@get!(leftCs, N, [zeros(i+1) for i = 1:N])
        rightC = Base.@get!(rightCs, N, [zeros((N-i+1)+1) for i = 1:N])

        f = 0.0

        subfield!(h, m, u)
        H = M ⊘ U

        @inbounds for i = 1:N
            vh[i] = h[i]
        end
        vH = Float64(H)

        leftC[1][1] = (1-ξ[1]*vh[1])/2
        leftC[1][2] = (1+ξ[1]*vh[1])/2
        for i = 2:N
            lC0 = leftC[i-1]
            lC = leftC[i]
            hi = ξ[i] * vh[i]
            lC[1] = lC0[1] * (1-hi)/2
            for j = 2:i
                lC[j] = lC0[j-1] * (1+hi)/2 + lC0[j] * (1-hi)/2
            end
            lC[end] = lC0[end] * (1+hi)/2
        end

        rightC[end][1] = (1-ξ[end]*vh[end])/2
        rightC[end][2] = (1+ξ[end]*vh[end])/2
        for i = (N-1):-1:1
            rC0 = rightC[i+1]
            rC = rightC[i]
            hi = ξ[i] * vh[i]
            rC[1] = rC0[1] * (1-hi)/2
            for j = 2:(N-i+1)
                rC[j] = rC0[j-1] * (1+hi)/2 + rC0[j] * (1-hi)/2
            end
            rC[end] = rC0[end] * (1+hi)/2
        end

        @compat @assert maximum(abs.(leftC[end] .- rightC[1])) ≤ 1e-10 (leftC[end], rightC[1])

        @assert isodd(N)
        z = (N+1) ÷ 2
        pm = sum(rightC[1][1:z])
        pp = sum(rightC[1][(z+1):end])

        b = Mag64(F, pp, pm)
        f -= log1pxy(H, b)
        @assert isfinite(f)

        for i = 1:N
            f += log1pxy(h[i], u[i])
        end
        return f
    end

end

function entro_node_update{F<:Mag64}(m::F, u::F, params::Params{F})
    @extract params : λ=damping r pol

    h = m ⊘ u
    if r == 0 || pol == 0
        newu = zero(F)
    elseif r == Inf
        newu = ifelse(h == 0.0, zero(F), copysign(pol, h))
    else
        newu::F = ((h * pol) ↑ r) * pol
    end

    diff = abs(newu - u)
    newu = damp(newu, u, λ)
    newm = h ⊗ newu

    return diff, newu, newm
end

function iterate!{F<:Mag64}(messages::Messages{F}, patterns::Patterns, params::Params)
    @extract messages : N M K ux mw mτ1 uw Uτ1 mτ2 uτ1
    @extract patterns : X output
    @extract params   : accuracy1 accuracy2
    maxdiff = 0.0
    tnu1! = accuracy1 == :exact ? theta_node_update_exact! :
            accuracy1 == :accurate ? theta_node_update_accurate! :
            accuracy1 == :none ? theta_node_update_approx! :
            error("accuracy must be one of :exact, :accurate, :none (was given $accuracy)")
    tnu2! = accuracy2 == :exact ? theta_node_update_exact! :
            accuracy2 == :accurate ? theta_node_update_accurate! :
            accuracy2 == :none ? theta_node_update_approx! :
            error("accuracy must be one of :exact, :accurate, :none (was given $accuracy)")
    for a = randperm(M + N*K)
        if a ≤ M
            ξ = X[a]
            for k = 1:K
                diff, Uτ1[a][k], mτ1[a][k] = tnu1!(mw[k], mτ1[a][k], ξ, uw[a][k], Uτ1[a][k], params)
                maxdiff = max(maxdiff, diff)
            end
            diff, _, mτ2[a] = tnu2!(mτ1[a], mτ2[a], ones(K), uτ1[a], zero(F), params)
            maxdiff = max(maxdiff, diff)
        else
            (params.r == 0 || params.pol == 0.0) && continue
            j = a - M
            k = (j-1) ÷ N + 1
            i = (j-1) % N + 1

            diff, ux[k][i], mw[k][i] = entro_node_update(mw[k][i], ux[k][i], params)
            maxdiff = max(diff, maxdiff)
        end
    end
    return maxdiff
end

function converge!(messages::Messages, patterns::Patterns, params::Params)
    @extract params : ϵ max_iters λ₀=damping quiet

    λ = λ₀
    ok = false
    strl = 0
    t = @elapsed for it = 1:max_iters
        diff = iterate!(messages, patterns, params)

        if !quiet
            str = "[it=$it Δ=$diff λ=$λ]"
            print("\r", " "^strl, "\r", str)
            strl = length(str)
            #println(str)
            flush(STDOUT)
            strl = length(str)
        end
        if diff < ϵ
            ok = true
            quiet || println("\nok")
            break
        end
    end
    if !quiet
        ok || println("\nfailed")
        println("elapsed time = $t seconds")
    end
    return ok
end

transf1(w::Vec) = sign(sum(w))

transf1!(r0::Vec, ws::Vec2, ξ::Vec) = transf1(transf0!(r0, ws, ξ))

function test!(r0::Vec, ws::Vec2, ξ::Vec, out::Int)
    o = transf1!(r0, ws, ξ)
    return o != out
end

function test(ws::Vec2, ξs::Vec2, output::IVec)
    r0 = Array(Float64, length(ws))
    sum([test!(r0, ws, ξ, out) for (ξ,out) in zip(ξs, output)])
end

function test(messages::Messages, patterns::Patterns)
    @extract messages : N K mw
    @extract patterns : X output
    ws = [Float64[mw[k][i] for i = 1:N] for k = 1:K]
    return test(ws, X, output)
end

function nonbayes_test(messages::Messages, patterns::Patterns)
    @extract messages : N K mw
    @extract patterns : X output
    ws = [Float64[sign0(mw[k][i]) for i = 1:N] for k = 1:K]
    return test(ws, X, output)
end

function free_energy{F<:Mag64}(messages::Messages{F}, patterns::Patterns, params::Params{F})
    @extract messages : M N K ux mw mτ1 uw Uτ1 mτ2 uτ1
    @extract patterns : X output
    @extract params   : r pol

    f = 0.0

    for a = 1:M
        ξ = X[a]
        for k = 1:K
            f += free_energy_theta(mw[k], mτ1[a][k], ξ, uw[a][k], Uτ1[a][k])
        end
        f += free_energy_theta_exact(mτ1[a], mτ2[a], ones(K), uτ1[a], zero(F))
    end

    for k = 1:K, i = 1:N
        f -= logZ(ux[k][i], F[uw[a][k][i] for a = 1:M])

        f -= logtwo / 2
        f += log1pxy(pol, -pol) / 2
        hkix = mw[k][i] ⊘ ux[k][i]
        f += log1pxy(hkix, ux[k][i])
        hpol = hkix * pol
        mx = hpol ↑ (r + 1)
        f += mcrossentropy(mx, hpol)
    end

    return f / (N * K)
end

function compute_S{F<:Mag64}(messages::Messages{F}, params::Params{F})
    @extract messages : N K ux mw
    @extract params   : r pol
    S = 0.0
    for k = 1:K, i = 1:N
        hkix = mw[k][i] ⊘ ux[k][i]
        hxki = (hkix * pol) ↑ r
        hh = hkix * hxki
        S += Float64(hh ⊗ pol)
    end
    return S / (N * K)
end

function compute_q̃{F<:Mag64}(messages::Messages{F}, params::Params{F})
    @extract messages : N K ux mw
    @extract params   : r pol
    q̃ = 0.0
    for k = 1:K, i = 1:N
        hkix = mw[k][i] ⊘ ux[k][i]
        mx = (hkix * pol) ↑ (r + 1)
        q̃ += mx^2
    end
    return q̃ / (N * K)
end

function compute_q(messages::Messages)
    @extract messages : N K mw
    q = 0.0
    for k = 1:K, i = 1:N
        q += Float64(mw[k][i])^2
    end
    return q / (N * K)
end

function mags_symmetry(messages::Messages)
    @extract messages : N K mw
    overlaps = eye(K)
    qs = zeros(K)
    for k1 = 1:K
        z = 0.0
        for i = 1:N
            z += Float64(mw[k1][i])^2
        end
        qs[k1] = √z
    end
    for k1 = 1:K, k2 = k1+1:K
        s = 0.0
        for i = 1:N
            s += Float64(mw[k1][i]) * Float64(mw[k2][i])
        end
        s /= qs[k1] * qs[k2]
        overlaps[k1,k2] = s
        overlaps[k2,k1] = s
    end
    return overlaps, qs / N
end

"""
    FocusingProtocol

Abstract type representing a protocol for the focusing procedure, i.e. a way to produce
successive values for the quantities `γ`, `y` and `β`. Currently, however, only `β=Inf`
is supported. To be provided as an argument to [`focusingBP`](@ref).

Available protocols are: [`StandardReinforcement`](@ref), [`Scoping`](@ref), [`PseudoReinforcement`](@ref) and
[`FreeScoping`](@ref).
"""
abstract FocusingProtocol

immutable StandardReinforcement <: FocusingProtocol
    r::FloatRange{Float64}
    StandardReinforcement{T<:Real}(r::Range{T}) = new(r)
end

@doc """
    StandardReinforcement(r::Range) <: FocusingProtocol

Standard reinforcement protocol, returns `γ=Inf` and `y=1/(1-x)`, where `x` is taken from the given range `r`.
""" -> StandardReinforcement(r::Range)

"""
    StandardReinforcement(dr::Float64) <: FocusingProtocol

Shorthand for [`StandardReinforcement`](@ref)`(0:dr:(1-dr))`.
"""
StandardReinforcement(dr::Float64) = StandardReinforcement(0.0:dr:(1-dr))

Base.start(s::StandardReinforcement) = start(s.r)
function Base.next(s::StandardReinforcement, i)
    n = next(s.r, i)
    return (Inf, 1/(1-n[1]), Inf), n[2]
end
Base.done(s::StandardReinforcement, i) = done(s.r, i)

"""
    Scoping(γr::Range, y) <: FocusingProtocol

Focusing protocol with fixed `y` and a varying `γ` taken from the given `γr` range.
"""
immutable Scoping <: FocusingProtocol
    γr::FloatRange{Float64}
    y::Float64
    Scoping(γr::Range, y) = new(γr, y)
end

Base.start(s::Scoping) = start(s.γr)
function Base.next(s::Scoping, i)
    n = next(s.γr, i)
    return (n[1], s.y, Inf), n[2]
end
Base.done(s::Scoping, i) = done(s.γr, i)


immutable PseudoReinforcement <: FocusingProtocol
    r::Vector{Float64}
    x::Float64
    PseudoReinforcement{T<:Real}(r::Range{T}...; x::Real=0.5) = new(vcat(map(collect, r)...), x)
end
@doc """
    PseudoReinforcement(r::Range...; x=0.5) <: FocusingProtocol

A focusing protocol in which both `γ` and `y` are progressively increased, according to
the formulas

```julia
γ = atanh(ρ^x)
y = 1+ρ^(1-2x)/(1-ρ)
```

where `ρ` is taken from the given range(s) `r`. With `x=0`, this is basically the same as
[`StandardReinforcement`](@ref).
""" -> PseudoReinforcement(r::Range...)

"""
    PseudoReinforcement(dr::Float64; x=0.5) <: FocusingProtocol

Shorthand for [`PseudoReinforcement`](@ref)`(0:dr:(1-dr); x=x)`.
"""
PseudoReinforcement(dr::Float64; x::Real=0.5) = PseudoReinforcement(0.0:dr:(1-dr), x=x)

Base.start(s::PseudoReinforcement) = start(s.r)
function Base.next(s::PseudoReinforcement, i)
    if done(s.r, i)
        n = (Inf, Inf, Inf), i
    else
        n = next(s.r, i)
    end
    x = s.x
    ρ = n[1]
    # some special cases just to avoid possible 0^0
    if x == 0.5
        return (atanh(√ρ), (2-ρ)/(1-ρ), Inf), n[2]
    elseif x == 0
        return (Inf, 1/(1-ρ), Inf), n[2]
    else
        return (atanh(ρ^x), 1+ρ^(1-2x)/(1-ρ), Inf), n[2]
    end
end
Base.done(s::PseudoReinforcement, i) = done(s.r, i)

"""
    FreeScoping(list::Vector{NTuple{2,Float64}}) <: FocusingProtocol

A focusing protocol which just returns the values of `(γ,y)` from the given `list`.

Example:

```julia
FreeScoping([(1/(1-x), (2-x)/(1-x)) for x = 0:0.01:0.99])
```
"""
immutable FreeScoping <: FocusingProtocol
    list::Vector{NTuple{3,Float64}}
    FreeScoping(list::Vector{NTuple{3,Float64}}) = new(list)
end
FreeScoping(list::Vector{NTuple{2,Float64}}) = FreeScoping(NTuple{3,Float64}[(γ,y,Inf) for (γ,y) in list])

Base.start(s::FreeScoping) = start(s.list)
Base.next(s::FreeScoping, i) = next(s.list, i)
Base.done(s::FreeScoping, i) = done(s.list, i)

"""
    focusingBP(N, K, patternspec; keywords...)

Run the Focusing Belief Propagation algorithm on a fully-connected committee machine with binary weights.
`N` is the input (first layer) size, `K` the number of the hidden units (second layer size), and
`patternspec` specifies how to build the patterns for the training set. Note that with the defult
settings `K` must be odd (see notes for the `accuracy1` and `accuracy2` arguments below).

Possible values of `patternspec` are:

* a `Float64` number: this is interpreted as the `α` parameter, and `M = α*N*K` random ±1 patterns are generated.
* a `Tuple` with `Vector{Vector{Float64}}` and a `Vector{Float64}`: these are the inputs and associated desired outputs.
* a string: the patterns are read from a file (one input pattern per line, entries separated by whitespace, outputs are
            assumed to be all 1); the file can be gzipped.
* a `Patterns` object (which could be the output of a previous run of the function).

*Note*: all inputs and outputs must be ∈ {-1,1}.

The keyword arguments are:

* `max_iters` (default = `1000`): maximum number of BP iterations per step. If convergence is not achieved in this many iterations,
                                  the algorithm proceeds to the next step.
* `max_steps` (default = `typemax(Int)`): maximum number of focusing steps.
* `seed` (default = `1`): random seed.
* `damping` (default = `0`): BP damping parameter (between `0` and `1`; `0` = no damping).
* `quiet` (default = `false`): whether to print on screen
* `accuracy1` (default = `:accurate`): accuracy of the messages computation at the hidden units level. Possible values are:
                                       `:accurate` (Gaussian approximation, good for large `N`, works in `O(N)` time),
                                       `:exact` (no approximation, uses convolutions, good for small `N`, works in `O(N³)` time, requires
                                       `N` to be odd),
                                       `:none` (a TAP-like approximation, fast but never very good, should probably be removed or done
                                       properly...).
* `accuracy2` (default = `:exact`): accuracy of the messages computation at the output node level.
                                    See `accuracy1` (and think of `K` instead of `N`).
* `randfact` (default = `0.01`): random factor used in the initialization of the messages. Must be between `0` and `1`. Large values are
                                 not a good idea.
* `fprotocol` (default = `StandardReinforcement(1e-2)`): focusing protocol specification. See [`FocusingProtocol`](@ref).
* `ϵ` (default = `1e-3`): convergence criterion: BP is assumed to have converged when the difference between messages in two successive
                          iterations is smaller than this value. Reduce it (e.g. to 1e-6) for more precise results, while increasing
                          `max_iters`.
* `messfmt` (default = `:tanh`): internal storage format for messages: it can be either `:tanh` or `:plain`; `:tanh` is much more
                                 precise but slower.
* `initmess` (default = `nothing`): how to initialize the messages. If `nothing`, they are initialized randomly; If a string is given,
                                    they are read from a file (see also [`read_messages`](@ref) and [`write_messages`](@ref)). If a
                                    `Messages` object is given (e.g. one returned from an earlier run of the function, or from
                                    [`read_messages`](@ref)) it is used (and overwritten).
* `outatzero` (default = `true`): if `true`, the algorithm exits as soon as a solution to the learning problem is found, without waiting
                                  for the focusing protocol to terminate.
* `writeoutfile` (default = `:auto`): whether to write results on an output file. Can be `:never`, `:always` or `:auto`. The latter means
                                      that a file is written when `outatzero` is set to `false` and only when BP converges. It can make sense
                                      setting this to `:always` even when `outfile` is `nothing` to force the computation of the local
                                      entropy and other thermodynamic quantities.
* `outfile` (default = `nothing`): the output file name. `nothing` means no output file is written. An empty string means using a default
                                   file name of the form: `"results_BPCR_N\$(N)_K\$(K)_M\$(M)_s\$(seed).txt"`.
* `outmessfiletmpl` (default = `nothing`): template file name for writing the messages at each focusing step. The file name should include a
                                           substring `"%gamma%"` which will be substituted with the value of the `γ` parameter at each step.
                                           If `nothing`, it is not used. If empty, the default will be used:
                                           `"messages_BPCR_N\$(N)_K\$(K)_M\$(M)_g%gamma%_s\$(seed).txt.gz"`.
                                           *Note*: this can produce a lot of fairly large files!.

The function returns three objects: the number of training errors, the messages and the patterns. The last two can be used as inputs to
successive runs of the algorithms, as the `initmessages` keyword argument and the `patternspec` argument, respectively.

Example of a run which solves a problem with `N * K = 1605` synapses with `K = 5` at `α = 0.3`:
```
julia> errs, messages, patterns = B.focusingBP(321, 5, 0.3, randfact=0.1, seed=135, max_iters=1, damping=0.5);
```
"""
function focusingBP(N::Integer, K::Integer,
                    initpatt::Union{AbstractString, Tuple{Vec2,Vec}, Real, Patterns};

                    max_iters::Integer = 1000,
                    max_steps::Integer = typemax(Int),
                    seed::Integer = 1,
                    damping::Real = 0.0,
                    quiet::Bool = false,
                    accuracy1::Symbol = :accurate,
                    accuracy2::Symbol = :exact,
                    randfact::Real = 0.01,
                    fprotocol::FocusingProtocol = StandardReinforcement(1e-2),
                    ϵ::Real = 1e-3,
                    messfmt::Symbol = :tanh,
                    initmess::Union{Messages,Void,AbstractString} = nothing,
                    outatzero::Bool = true,
                    writeoutfile::Symbol = :auto, # note: ∈ [:auto, :always, :never]; auto => !outatzero && converged
                    outfile::Union{AbstractString,Void} = nothing, # note: "" => default, nothing => no output
                    outmessfiletmpl::Union{AbstractString,Void} = nothing) # note: same as outfile

    srand(seed)

    N > 0 || throw(ArgumentError("N must be positive; given: $N"))
    K > 0 || throw(ArgumentError("K must be positive; given: $K"))

    writeoutfile ∈ [:auto, :always, :never] || error("invalide writeoutfile, expected one of :auto, :always, :never; given: $writeoutfile")
    max_iters ≥ 0 || throw(ArgumentError("max_iters must be non-negative; given: $max_iters"))
    max_steps ≥ 0 || throw(ArgumentError("max_steps must be non-negative; given: $max_steps"))
    0 ≤ damping < 1 || throw(ArgumentError("damping must be ∈ [0,1); given: $damping"))
    0 ≤ randfact ≤ 1 || throw(ArgumentError("randfact must be ∈ [0,1]; given: $randfact"))
    accuracy1 ∈ [:exact, :accurate, :none] || error("accuracy1 must be one of :exact, :accurate, :none; given: $accuracy1")
    accuracy2 ∈ [:exact, :accurate, :none] || error("accuracy2 must be one of :exact, :accurate, :none; given: $accuracy2")

    accuracy1 == :exact && iseven(N) && throw(ArgumentError("when accuracy1==:exact N must be odd, given: $N"))
    accuracy2 == :exact && iseven(K) && throw(ArgumentError("when accuracy2==:exact K must be odd, given: $K"))

    messfmt ∈ [:tanh, :plain] || throw(ArgumentError("invalid messfmt, should be :tanh or :plain; given: $messfmt"))
    F = messfmt == :tanh ? MagT64 : MagP64

    if isa(initpatt, Real)
        initpatt ≥ 0 || throw(ArgumentError("invalide negative initpatt; given: $initpatt"))
        initpatt = (N, round(Int, K * N * initpatt))
    end

    patterns = Patterns(initpatt)

    M = patterns.M

    messages::Messages = initmess ≡ nothing ? Messages(F, M, N, K, randfact) :
                         isa(initmess, AbstractString) ? read_messages(initmess, F) :
                         initmess

    messages.N == N || throw(ArgumentError("wrong messages size, expected N=$N; given: $(messages.N)"))
    messages.K == K || throw(ArgumentError("wrong messages size, expected K=$K; given: $(messages.K)"))
    messages.M == M || throw(ArgumentError("wrong messages size, expected M=$M; given: $(messages.M)"))
    F = eltype(messages)

    params = Params{F}(damping, ϵ, NaN, max_iters, accuracy1, accuracy2, 0.0, 0.0, 0.0, quiet)

    outfile == "" && (outfile = "results_BPCR_N$(N)_K$(K)_M$(M)_s$(seed).txt")
    outmessfiletmpl == "" && (outmessfiletmpl = "messages_BPCR_N$(N)_K$(K)_M$(M)_g%gamma%_s$(seed).txt.gz")
    lockfile = "bpcomm.lock"
    if outfile ≢ nothing && writeoutfile ∈ [:always, :auto]
        !quiet && println("writing outfile $outfile")
        exclusive(lockfile) do
            !isfile(outfile) && open(outfile, "w") do f
                println(f, "#1=pol 2=y 3=β 4=S 5=q 6=q̃ 7=βF 8=𝓢ᵢₙₜ 9=Ẽ")
            end
        end
    end

    ok = true
    if initmess ≢ nothing
        errs = nonbayes_test(messages, patterns)
        !quiet && println("initial errors = $errs")

        outatzero && errs == 0 && return errs, messages, patterns
    end
    !quiet && K > 1 && (println("mags overlaps="); display(mags_symmetry(messages)[1]); println())

    it = 1
    for (γ,y,β) in fprotocol
        isfinite(β) && error("finite β not yet supported; given: $β")
        pol = mtanh(F, γ)
        params.pol = pol
        params.r = y - 1
        params.β = β
        set_outfields!(messages, patterns.output, params.β)
        ok = converge!(messages, patterns, params)
        !quiet && K > 1 && (println("mags overlaps="); display(mags_symmetry(messages)[1]); println())
        errs = nonbayes_test(messages, patterns)

        if writeoutfile == :always || (writeoutfile == :auto && !outatzero)
            S = compute_S(messages, params)
            q = compute_q(messages)
            q̃ = compute_q̃(messages, params)
            βF = free_energy(messages, patterns, params)
            Σint = -βF - γ * S

            !quiet && println("it=$it pol=$pol y=$y β=$β (ok=$ok) S=$S βF=$βF Σᵢ=$Σint q=$q q̃=$q̃ Ẽ=$errs")
            (ok || writeoutfile == :always) && outfile ≢ nothing && exclusive(lockfile) do
                open(outfile, "a") do f
                    println(f, "$pol $y $β $S $q $q̃ $βF $Σint $errs")
                end
            end
            if outmessfiletmpl ≢ nothing
                outmessfile = replace(outmessfiletmpl, "%gamma%", γ)
                write_messages(outmessfile, messages)
            end
        else
            !quiet && println("it=$it pol=$pol y=$y β=$β (ok=$ok) Ẽ=$errs")
            errs == 0 && return 0, messages, patterns
        end
        it += 1
        it ≥ max_steps && break
    end
    return errs, messages, patterns
end

end # module
