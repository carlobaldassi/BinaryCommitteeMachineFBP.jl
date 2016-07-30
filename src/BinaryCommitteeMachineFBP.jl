# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

module BinaryCommitteeMachineFBP

using StatsFuns
using GZip
using ExtractMacro
using Iterators

const MAGFORMAT = isdefined(Main, :MAGFORMAT) ? Main.MAGFORMAT : :tanh
MAGFORMAT ∈ [:plain, :tanh] || error("MAGFORMAT must be either :plain of :tanh, found: $(Main.MAGFORMAT)")

if MAGFORMAT == :plain
    info("BinaryCommitteeMachineFBP: using plain magnetizations")
    include("Magnetizations.jl")
    using .Magnetizations
else
    info("BinaryCommitteeMachineFBP: using tanh magnetizations")
    include("MagnetizationsT.jl")
    using .MagnetizationsT
end

include("Util.jl")
using .Util

immutable Messages
    N::Int
    K::Int
    M::Int

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
    #              DEPTH
    ux::MagVec2    # 0
    mw::MagVec2    # 0+1
    mτ1::MagVec2   # 1+2
    uw::MagVec3    # 1
    Uτ1::MagVec2   # 1
    mτ2::MagVec    # 2+
    uτ1::MagVec2   # 2

    function Messages(M::Int, N::Int, K::Int, x::Float64)
        ux = [mflatp(N) for k = 1:K]
        mw = [mflatp(N) for k = 1:K]
        mτ1 = [mflatp(K) for a = 1:M]
        uw = [MagVec[map(Mag64, x*(2*rand(N)-1)) for k = 1:K] for a = 1:M]
        Uτ1 = [mflatp(K) for a = 1:M]
        mτ2 = mflatp(M)
        uτ1 = [mflatp(K) for a = 1:M]

        for k = 1:K, i = 1:N, a = 1:M
            mw[k][i] = mw[k][i] ⊗ uw[a][k][i]
        end
        for a = 1:M, k = 1:K
            mτ1[a][k] = mτ1[a][k] ⊗ Uτ1[a][k] ⊗ uτ1[a][k]
        end

        new(N, K, M, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1)
    end

    global read_messages
    function read_messages(io::IO)
        l = split(readline(io))
        @assert length(l) == 2 && l[1] == "fmt:"
        fmt = Val{symbol(l[2])}
        l = split(readline(io))
        @assert length(l) == 4 && l[1] == "N,K,M:"
        N, K, M = parse(Int, l[2]), parse(Int, l[3]), parse(Int, l[4])

        ux = [mflatp(N) for k = 1:K]
        mw = [mflatp(N) for k = 1:K]
        mτ1 = [mflatp(K) for a = 1:M]
        uw = [[mflatp(N) for k = 1:K] for a = 1:M]
        Uτ1 = [mflatp(K) for a = 1:M]
        mτ2 = mflatp(M)
        uτ1 = [mflatp(K) for a = 1:M]

        expected_lines = K + K + M + M*K + M + 1 + M + K
        for (i,l) in enumerate(eachline(io))
            i > expected_lines && (@assert strip(l) == "END"; break)
            @readmagvec(l, fmt, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1)
        end
        @assert eof(io)
        return new(N, K, M, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1)
    end
end

read_messages(filename::AbstractString) = gzopen(read_messages, filename, "r")

function write_messages(filename::AbstractString, messages::Messages)
    gzopen(filename, "w") do f
        write_messages(f, messages)
    end
end

function write_messages(io::IO, messages::Messages)
    @extract messages : N K M ux mw mτ1 uw Uτ1 mτ2 uτ1

    println(io, "fmt: ", magformat())
    println(io, "N,K,M: $N $K $M")
    @dumpmagvecs(io, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1)
    println(io, "END")
end

function Base.copy!(dest::Messages, src::Messages)
    @assert dest.N == src.N
    @assert dest.K == src.K
    @assert dest.M == src.M
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

function set_outfields!(messages::Messages, output::Vector, β::Float64)
    @extract messages : N K M mτ2
    @assert length(output) == M
    t = tanh(β / 2)
    for a = 1:M
        mτ2[a] = forcedmag(output[a] * t) # forced avoids clamping
    end
end

print_mags(messages::Messages) = print_mags(STDOUT, messages)
function print_mags(io::IO, messages::Messages)
    @extract messages : N K mw
    for k = 1:K, i = 1:N
        @printf(io, "%i %i %.15f\n", k, i, Float64(mw[k][i]))
    end
end

type Params
    damping::Float64
    ϵ::Float64
    β::Float64
    max_iters::Int
    accuracy1::Symbol
    accuracy2::Symbol
    r::Float64
    pol::Mag64
    dγ::Float64
    quiet::Bool
end

immutable Patterns
    M::Int
    X::Vec2
    output::IVec
    Patterns(X, o) = new(length(X), X, o)
end

Patterns(Xo::Tuple{Vec2,Vec}) = Patterns(Xo...)

Patterns(NM::Tuple{Int,Int}) = ((N,M) = NM; Patterns([rand(-1.0:2.0:1.0, N) for a = 1:M], ones(M)))
Patterns(patterns::Patterns) = deepcopy(patterns)

function Patterns(patternsfile::AbstractString)
    X = Vec[]
    N = 0
    M = 0
    open(patternsfile) do f
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

let hs = Dict{Int,MagVec}(), vhs = Dict{Int,Vec}(), newUs = Dict{Int,MagVec}(), vHs = Dict{Int,Vec}(), leftCs = Dict{Int,Vec2}(), rightCs = Dict{Int,Vec2}()

    global theta_node_update_approx!
    function theta_node_update_approx!(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64, params::Params)
        @extract params : λ=damping

        N = length(m)
        h = Base.@get!(hs, N, Array(Mag64, N))
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
        newU = merf(μ / √dσ²)

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
            newu = Mag64(clamp(ξi * (p0 + ξi * (hi * pμ + ξi * (1-hi^2) * pσ)), -1+eps(-1.0), 1-eps(1.0))) # use mag-functions?
            maxdiff = max(maxdiff, abs(newu - u[i]))
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]
        end

        return maxdiff, U, M
    end

    global theta_node_update_accurate!
    function theta_node_update_accurate!(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64, params::Params)
        @extract params : λ=damping

        N = length(m)
        h = Base.@get!(hs, N, Array(Mag64, N))
        vh = Base.@get!(vhs, N, Array(Float64, N))

        subfield!(h, m, u)
        H = M ⊘ U

        @inbounds for i = 1:N
            vh[i] = h[i]
        end

        σ² = computeσ²(vh, ξ)
        μ = dot(vh, ξ)

        dσ² = 2σ²
        newU = merf(μ / √dσ²)

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
    function theta_node_update_exact!(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64, params::Params)
        @extract params : λ=damping

        N = length(m)
        h = Base.@get!(hs, N, Array(Mag64, N))
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

        @assert maximum(abs(leftC[end] .- rightC[1])) ≤ 1e-10 (leftC[end], rightC[1])

        @assert isodd(N)
        z = (N+1) ÷ 2
        pm = sum(rightC[1][1:z])
        pp = sum(rightC[1][(z+1):end])

        newU = Mag64(pp, pm)

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

            mp = Mag64(clamp(pp + ξi * pz - pm, -1.0, 1.0))
            mm = Mag64(clamp(pp - ξi * pz - pm, -1.0, 1.0))
            newu = exactmix(H, mp, mm)

            maxdiff = max(maxdiff, abs(newu - u[i]))
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]

            @assert isfinite(u[i]) (u[i],)
        end
        return maxdiff, U, M
    end
end

function entro_node_update(m::Mag64, u::Mag64, params::Params)
    @extract params : λ=damping r pol

    h = m ⊘ u
    if r == 0 || pol == 0
        newu = zero(Mag64)
    elseif r == Inf
        newu = ifelse(h == 0.0, zero(Mag64), copysign(pol, h))
    else
        newu::Mag64 = ((h * pol) ↑ r) * pol
    end

    diff = abs(newu - u)
    newu = damp(newu, u, λ)
    newm = h ⊗ newu

    return diff, newu, newm
end

function iterate!(messages::Messages, patterns::Patterns, params::Params)
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
            diff, _, mτ2[a] = tnu2!(mτ1[a], mτ2[a], ones(K), uτ1[a], zero(Mag64) #=Uτ2[a]=#, params)
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

let hs = Dict{Int,MagVec}(), vhs = Dict{Int,Vec}(), leftCs = Dict{Int,Vec2}(), rightCs = Dict{Int,Vec2}()
    global free_energy_theta
    function free_energy_theta(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64)
        N = length(m)
        h = Base.@get!(hs, N, Array(Mag64, N))
        vh = Base.@get!(vhs, N, Array(Float64, N))

        f = 0.0

        subfield!(h, m, u)
        H = M ⊘ U

        @inbounds for i = 1:N
            vh[i] = h[i]
        end

        σ = computeσ(vh, ξ)
        μ = dot(vh, ξ)

        b = merf(μ / σ)

        f -= log1pxy(H, b)
        @assert isfinite(f)

        for i = 1:N
            f += log1pxy(h[i], u[i])
        end
        return f
    end

    global free_energy_theta_exact
    function free_energy_theta_exact(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64)

        N = length(m)
        h = Base.@get!(hs, N, Array(Mag64, N))
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

        @assert maximum(abs(leftC[end] .- rightC[1])) ≤ 1e-10 (leftC[end], rightC[1])

        @assert isodd(N)
        z = (N+1) ÷ 2
        pm = sum(rightC[1][1:z])
        pp = sum(rightC[1][(z+1):end])

        b = Mag64(pp, pm)
        f -= log1pxy(H, b)
        @assert isfinite(f)

        for i = 1:N
            f += log1pxy(h[i], u[i])
        end
        return f
    end
end

# used with pseudo-reinforcement.
function free_energy2(messages::Messages, patterns::Patterns, params::Params)
    @extract messages : M N K ux mw mτ1 uw Uτ1 mτ2 uτ1
    @extract patterns : X output
    @extract params   : r pol

    f = 0.0

    for a = 1:M
        ξ = X[a]
        for k = 1:K
            f += free_energy_theta(mw[k], mτ1[a][k], ξ, uw[a][k], Uτ1[a][k])
        end
        f += free_energy_theta_exact(mτ1[a], mτ2[a], ones(K), uτ1[a], zero(Mag64))
    end

    for k = 1:K, i = 1:N
        f -= logZ(ux[k][i], Mag64[uw[a][k][i] for a = 1:M])

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

function compute_S(messages::Messages, params::Params)
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

function compute_q̃(messages::Messages, params::Params)
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

abstract IterationProtocol

immutable StandardReinforcement <: IterationProtocol
    r::FloatRange{Float64}
    StandardReinforcement{T<:Real}(r::Range{T}) = new(r)
end
StandardReinforcement(dr::Float64) = StandardReinforcement(0.0:dr:(1-dr))

Base.start(s::StandardReinforcement) = start(s.r)
function Base.next(s::StandardReinforcement, i)
    n = next(s.r, i)
    return (Inf, 1/(1-n[1]), Inf), n[2]
end
Base.done(s::StandardReinforcement, i) = done(s.r, i)

immutable Scoping <: IterationProtocol
    γr::FloatRange{Float64}
    y::Float64
    β::Float64
    Scoping(γr::Range, y, β=Inf) = new(γr, y, β)
end

Base.start(s::Scoping) = start(s.γr)
function Base.next(s::Scoping, i)
    n = next(s.γr, i)
    return (n[1], s.y, s.β), n[2]
end
Base.done(s::Scoping, i) = done(s.γr, i)


immutable PseudoReinforcement <: IterationProtocol
    r::Vector{Float64}
    x::Float64
    PseudoReinforcement{T<:Real}(r::Range{T}...; x::Real=0.5) = new(vcat(map(collect, r)...), x)
end
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
Base.done(s::PseudoReinforcement, i) = false #done(s.r, i)

immutable FreeScoping <: IterationProtocol
    list::Vector{NTuple{3,Float64}}
    FreeScoping(list::Vector{NTuple{3,Float64}}) = new(list)
end
FreeScoping(list::Vector{NTuple{2,Float64}}) = FreeScoping(NTuple{3,Float64}[(γ,y,Inf) for (γ,y) in list])

Base.start(s::FreeScoping) = start(s.list)
Base.next(s::FreeScoping, i) = next(s.list, i)
Base.done(s::FreeScoping, i) = done(s.list, i)

function rsolve(N::Int, K::Int,
                initpatt::Union{AbstractString, Tuple{Vec2,Vec}, Float64, Patterns},
                ginitpatt::Union{AbstractString, Tuple{Vec2,Vec}, Float64, Patterns};
                max_iters::Int = 1000,
                max_epochs::Int = typemax(Int),
                seed::Int = 1,
                damping::Real = 0.0,
                quiet::Bool = false,
                accuracy1::Symbol = :accurate,
                accuracy2::Symbol = :exact,
                randfact::Float64 = 0.01,
                iteration::IterationProtocol = StandardReinforcement(1e-2),
                ϵ::Float64 = 1e-3,
                initmessages::Union{Messages,Void,AbstractString} = nothing,
                outatzero::Bool = true,
                writeoutfile::Symbol = :auto, # note: ∈ [:auto, :always, :never]; auto => !outatzero && converged
                outfile::Union{AbstractString,Void} = nothing, # note: "" => default, nothing => no output
                outmessfiletmpl::Union{AbstractString,Void} = nothing) # note: same as outfile

    srand(seed)

    writeoutfile ∈ [:auto, :always, :never] || error("invalide writeoutfile, expected one of :auto, :always, :never, given: $writeoutfile")

    isa(initpatt, Float64) && (initpatt = (N, round(Int, K * N * initpatt)))
    isa(ginitpatt, Float64) && (ginitpatt = (N, round(Int, K * N * ginitpatt)))

    print("generating patterns... ")
    print("T")
    patterns = Patterns(initpatt)
    print("G")
    gpatterns = Patterns(ginitpatt)
    println(" done")

    M = patterns.M
    gM = gpatterns.M

    messages::Messages = initmessages ≡ nothing ? Messages(M, N, K, randfact) :
                         isa(initmessages, AbstractString) ? read_messages(initmessages) :
                         initmessages
    @assert messages.N == N
    @assert messages.K == K
    @assert messages.M == M

    params = Params(damping, ϵ, NaN, max_iters, accuracy1, accuracy2, 0.0, 0.0, 0.0, quiet)

    outfile == "" && (outfile = "results_BPCR_N$(N)_K$(K)_M$(M)_s$(seed).txt")
    outmessfiletmpl == "" && (outmessfiletmpl = "messages_BPCR_N$(N)_K$(K)_M$(M)_g%gamma%_s$(seed).txt.gz")
    lockfile = "bpcomm.lock"
    if outfile ≢ nothing && writeoutfile ∈ [:always, :auto]
        println("writing outfile $outfile")
        exclusive(lockfile) do
            !isfile(outfile) && open(outfile, "w") do f
                println(f, "#1=pol 2=y 3=β 4=S 5=q 6=q̃ 7=βF 8=𝓢ᵢₙₜ 9=Ẽ")
            end
        end
    end

    ok = true
    if initmessages ≢ nothing
        errs = nonbayes_test(messages, patterns)
        println("initial errors = $errs")

        outatzero && err == 0 && return 0
    end
    println("mags overlaps=\n", mags_symmetry(messages))

    it = 1
    for (γ,y,β) in iteration
        isfinite(β) && error("finite β not supported (needs energy computation in freeenergy2, see BPPerc.jl); given: $β")
        pol = mtanh(γ)
        params.pol = pol
        params.r = y - 1
        params.β = β
        set_outfields!(messages, patterns.output, params.β)
        ok = converge!(messages, patterns, params)
        println("mags overlaps=\n", mags_symmetry(messages))
        errs = nonbayes_test(messages, patterns)

        if writeoutfile == :always || (writeoutfile == :auto && !outatzero)
            S = compute_S(messages, params)
            q = compute_q(messages)
            q̃ = compute_q̃(messages, params)
            βF = free_energy2(messages, patterns, params)
            Σint = -βF - γ * S

            println("it=$it pol=$pol y=$y β=$β (ok=$ok) S=$S βF=$βF Σᵢ=$Σint q=$q q̃=$q̃ Ẽ=$errs")
            (ok || writeoutfile == :always) && outfile ≢ nothing && open(outfile, "a") do f
                println(f, "$pol $y $β $S $q $q̃ $βF $Σint $errs")
            end
            if outmessfiletmpl ≢ nothing
                outmessfile = replace(outmessfiletmpl, "%gamma%", γ)
                write_messages(outmessfile, messages)
            end
        else
            println("it=$it pol=$pol y=$y β=$β (ok=$ok) Ẽ=$errs")
            errs == 0 && return 0, messages, patterns
        end
        it += 1
        it ≥ max_epochs && break
    end
    return ok, messages, patterns
end

end # module
