module BinaryCommitteeMachineFBP

using DataFrames
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

    mw0::MagVec2

    function Messages(M::Int, N::Int, K::Int, x::Float64)
        ux = [mflatp(N) for k = 1:K]
        mw = [mflatp(N) for k = 1:K]
        mτ1 = [mflatp(K) for a = 1:M]
        uw = [MagVec[map(Mag64, x*(2*rand(N)-1)) for k = 1:K] for a = 1:M]
        Uτ1 = [mflatp(K) for a = 1:M]
	#mτ2 = mrand(x, M)
	mτ2 = mflatp(M)
        #uτ1 = [x*(2*rand(K)-1) for a = 1:M]
        uτ1 = [mflatp(K) for a = 1:M]

        for k = 1:K, i = 1:N, a = 1:M
            mw[k][i] = mw[k][i] ⊗ uw[a][k][i]
        end
        for a = 1:M, k = 1:K
            mτ1[a][k] = mτ1[a][k] ⊗ Uτ1[a][k] ⊗ uτ1[a][k]
        end

        mw0 = [mflatp(N) for k = 1:K]
        new(N, K, M, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1, mw0)
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
        mw0 = [mflatp(N) for k = 1:K]

        expected_lines = K + K + M + M*K + M + 1 + M + K
        for (i,l) in enumerate(eachline(io))
            i > expected_lines && (@assert strip(l) == "END"; break)
            #@show i
            @readmagvec(l, fmt, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1, mw0)
        end
        @assert eof(io)
        return new(N, K, M, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1, mw0)
    end
end

read_messages(filename::AbstractString) = gzopen(read_messages, filename, "r")

function write_messages(filename::AbstractString, messages::Messages)
    gzopen(filename, "w") do f
        write_messages(f, messages)
    end
end

function write_messages(io::IO, messages::Messages)
    @extract messages : N K M ux mw mτ1 uw Uτ1 mτ2 uτ1 mw0

    println(io, "fmt: ", magformat())
    println(io, "N,K,M: $N $K $M")
    @dumpmagvecs(io, ux, mw, mτ1, uw, Uτ1, mτ2, uτ1, mw0)
    println(io, "END")
end

function Base.copy!(dest::Messages, src::Messages)
    @assert dest.N == src.N
    @assert dest.K == src.K
    @assert dest.M == src.M
    for k = 1:dest.K
        copy!(dest.ux[k], src.ux[k])
        copy!(dest.mw[k], src.mw[k])
        copy!(dest.mw0[k], src.mw0[k])
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
    @extract messages N K M mτ2
    @assert length(output) == M
    t = tanh(β / 2)
    for a = 1:M
        mτ2[a] = forcedmag(output[a] * t) # forced avoids clamping
    end
end

function save_mags!(messages::Messages)
    @extract messages N K mw mw0
    for k = 1:K, i = 1:N
        mw0[k][i] = mw[k][i]
    end
end

function reinforce!(messages::Messages, γ::Float64)
    @extract messages M N K ux mw mw0 uw
    for k = 1:K, i = 1:N
        #rand() < 0.1 && (mw[k][i] = reinforce(mw[k][i], mw0[k][i], γ))
        #mw[k][i] = reinforce(mw[k][i], mw0[k][i], γ)
        m = mw[k][i]
        m0 = mw0[k][i]
        u = ux[k][i]
        h = m ⊘ u
        newu = reinforce(m0, γ)
        #newu = rand() ≤ γ ? m0 : zero(Mag64)
        mw[k][i] = h ⊗ newu
        #println("i=$i : oldm=$m newm=$(mw[k][i]) h=$(h) u=$u newu=$newu")
        ux[k][i] = newu
    end
end

function update_pol!(messages::Messages, ws::Vec2, pol::Mag64)
    @extract messages N K ux mw
    @assert length(ws) == K
    @assert all(w->(length(w) == N), ws)
    for k = 1:K, i = 1:N
        x0 = ux[k][i]
        @assert ws[k][i] == 1 || ws[k][i] == -1
        x1 = copysign(pol, ws[k][i])
        ux[k][i] = x1
        mw[k][i] = mw[k][i] ⊘ x0 ⊗ x1
    end
end

function flip_field!(messages::Messages, ws::Vec2, j::Int)
    @extract messages N ux mw
    k = (j-1) ÷ N + 1
    i = (j-1) % N + 1

    x = ux[k][i]
    ux[k][i] = -x
    mw[k][i] = mw[k][i] ⊘ x ⊗ -x
    ws[k][i] = -ws[k][i]
end

print_mags(messages::Messages) = print_mags(STDOUT, messages)
function print_mags(io::IO, messages::Messages)
    @extract messages N K mw
    for k = 1:K, i = 1:N
        @printf(io, "%i %i %.15f\n", k, i, Float64(mw[k][i]))
    end
end

function compare_signs(messages::Messages, ws::Vec2)
    @extract messages N K mw
    @assert length(ws) == K
    @assert all(w->(length(w) == N), ws)
    diff = 0
    for k = 1:K, i = 1:N
        diff += (sign(mw[k][i]) ≠ sign(ws[k][i]))
    end
    return diff
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

function set_τ1_fields!(messages::Messages, ws::Vec2, pol::Mag64, patterns::Patterns)
    @extract messages N M K mτ1
    @extract patterns X
    @assert length(ws) == K
    @assert all(w->(length(w) == N), ws)
    for a = 1:M, k = 1:K
        v = sign(ws[k] ⋅ X[a])
        @assert v ≠ 0
        mτ1[a][k] = copysign(pol, v)
    end
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

    global theta_node_update!
    function theta_node_update!(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64, params::Params)
        @extract params λ=damping

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
            newu = Mag64(clamp(ξi * (p0 + ξi * (hi * pμ + ξi * (1-hi^2) * pσ)), -1+eps(-1.0), 1-eps(1.0))) # XXX mag
            maxdiff = max(maxdiff, abs(newu - u[i]))
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]
        end

        return maxdiff, U, M
    end

    global theta_node_update_accurate!
    function theta_node_update_accurate!(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64, params::Params)
        @extract params λ=damping

        N = length(m)
        h = Base.@get!(hs, N, Array(Mag64, N))
        vh = Base.@get!(vhs, N, Array(Float64, N))

        subfield!(h, m, u)
        H = M ⊘ U
        #ξ == ones(N) && abs(M) == 1.0 && @assert abs(H) == 1.0 (M,H,U) #DBG

        #all(mi->abs(mi)==1, m) && @assert all(hi->abs(hi)==1.0, h) (m,h,u) #DBG

        @inbounds for i = 1:N
            vh[i] = h[i]
        end
        #@assert m == vh

        #vH = Float64(H)

        σ² = computeσ²(vh, ξ)
        μ = dot(vh, ξ)

        #@assert σ² == 0

        dσ² = 2σ²
        newU = merf(μ / √dσ²)

        maxdiff = 0.0
        U = damp(newU, U, λ)
        M = H ⊗ U
        #@assert isfinite(M) (M,H,U)

        @inbounds for i = 1:N
            ξi = ξ[i]
            hi = vh[i]
            μ̄ = μ - ξi * hi
            σ̄² = σ² - (1-hi^2) * ξi^2
            sdσ̄² = √(2σ̄²)
            #erf₊ = erf((μ̄ + ξi) / sdσ̄²)
            #erf₋ = erf((μ̄ - ξi) / sdσ̄²)
            #newu = vH * (erf₊ - erf₋) / (2 + vH * (erf₊ + erf₋))
            #newu = Mag64(tanh((log1p(vH * erf₊) - log1p(vH * erf₋)) / 2))

            m₊ = (μ̄ + ξi) / sdσ̄²
            m₋ = (μ̄ - ξi) / sdσ̄²
            newu = erfmix(H, m₊, m₋)
            maxdiff = max(maxdiff, abs(newu - u[i]))
            #u[i] = clamp(newu * (1-λ) + u[i] * λ, -1+1e-15, 1-1e-15)
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]
        end
        return maxdiff, U, M
    end

    global theta_node_update_exact!
    function theta_node_update_exact!(m::MagVec, M::Mag64, ξ::Vec, u::MagVec, U::Mag64, params::Params)
        @extract params λ=damping

        N = length(m)
        h = Base.@get!(hs, N, Array(Mag64, N))
        vh = Base.@get!(vhs, N, Array(Float64, N))
        leftC = Base.@get!(leftCs, N, [zeros(i+1) for i = 1:N])
        rightC = Base.@get!(rightCs, N, [zeros((N-i+1)+1) for i = 1:N])

        subfield!(h, m, u)
        H = M ⊘ U
        #abs(M) == 1.0 && @assert (H == M || U == M) (H,M,U)

        @inbounds for i = 1:N
            vh[i] = h[i]
        end

        #vH = Float64(H)

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
        #maxdiff = abs(U - newU)
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

            #newu = Mag64(ξi * pz * vH / (1 + (pp - pm) * vH))
            mp = Mag64(clamp(pp + ξi * pz - pm, -1.0, 1.0))
            mm = Mag64(clamp(pp - ξi * pz - pm, -1.0, 1.0))
            newu = exactmix(H, mp, mm)

            maxdiff = max(maxdiff, abs(newu - u[i]))
            #u[i] = clamp(newu * (1-λ) + u[i] * λ, -1+1e-5, 1-1e-15)
            u[i] = damp(newu, u[i], λ)
            m[i] = h[i] ⊗ u[i]

            @assert isfinite(u[i]) (u[i],)
        end
        return maxdiff, U, M
    end
end

function entro_node_update(m::Mag64, u::Mag64, params::Params)
    @extract params λ=damping r pol

    h = m ⊘ u
    if r == 0 || pol == 0
        newu = zero(Mag64)
    elseif r == Inf
        newu = ifelse(h == 0.0, zero(Mag64), copysign(pol, h))
    else
        newu::Mag64 = ((h * pol) ↑ r) * pol

        # alternative version:
        #
        # hp = h * pol
        # pp = (1 + hp)^r
        # mm = (1 - hp)^r
        # newu = pol * (pp - mm) / (pp + mm)
    end

    diff = abs(newu - u)
    newu = damp(newu, u, λ)
    newm = h ⊗ newu

    return diff, newu, newm
end

function iterate!(messages::Messages, patterns::Patterns, params::Params)
    @extract messages N M K ux mw mτ1 uw Uτ1 mτ2 uτ1
    @extract patterns X output
    @extract params accuracy1 accuracy2
    maxdiff = 0.0
    tnu1! = accuracy1 == :exact ? theta_node_update_exact! :
            accuracy1 == :accurate ? theta_node_update_accurate! :
            accuracy1 == :none ? theta_node_update! :
            error("accuracy must be one of :exact, :accurate, :none (was given $accuracy)")
    tnu2! = accuracy2 == :exact ? theta_node_update_exact! :
            accuracy2 == :accurate ? theta_node_update_accurate! :
            accuracy2 == :none ? theta_node_update! :
            error("accuracy must be one of :exact, :accurate, :none (was given $accuracy)")
    for a = randperm(M + N*K)
        if a ≤ M
            ξ = X[a]
            #out = output[a]
            #println("a = $a")
            #println("out = $out")
            #println("Uτ1 pre  = ", Uτ1[a])
            for k = 1:K
                diff, Uτ1[a][k], mτ1[a][k] = tnu1!(mw[k], mτ1[a][k], ξ, uw[a][k], Uτ1[a][k], params)
                maxdiff = max(maxdiff, diff)
            end
            #println("Uτ1 post = ", Uτ1[a])
            #println()
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
    @extract params ϵ max_iters λ₀=damping quiet
    @extract patterns M

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
    #params.damping = λ₀
    return ok
end

function rein_solve!(messages::Messages, patterns::Patterns, params::Params)
    @extract params ϵ max_iters λ₀=damping dγ
    @extract patterns M

    λ = λ₀
    ok = false
    strl = 0
    γ = 0.0
    t = @elapsed for it = 1:max_iters
        #γ += dγ
        γ = 1 - (1 - γ) * (1 - dγ)
        #γ = dγ
        save_mags!(messages)
        diff = iterate!(messages, patterns, params)
        reinforce!(messages, γ)

        errs = nonbayes_test(messages, patterns)

        println("it=$it γ=$γ errs=$errs")
        if errs == 0
            ok = true
            println("\nok")
            break
        end
    end
    ok || println("\nfailed")
    println("elapsed time = $t seconds")
    #=open("mags.txt", "w") do f
        print_mags(f, messages)
    end=#
    #params.damping = λ₀
    return ok
end

transf1(w::Vec) = sign(sum(w))

transf1!(r0::Vec, ws::Vec2, ξ::Vec) = transf1(transf0!(r0, ws, ξ))

function test!(r0::Vec, ws::Vec2, ξ::Vec, out::Int)
    o = transf1!(r0, ws, ξ)
    #println("out=$out o=$o")
    #println("r0=$r0")
    #println()
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
    @extract messages N K mw
    @extract patterns X output
    ws = [Float64[sign0(mw[k][i]) for i = 1:N] for k = 1:K]
    return test(ws, X, output)
end

function parse_ws(filename::AbstractString)
    error("needs fixing after changes (?)")
    K = length(lines(filename))
    ws = Array(Vec, K)
    k = 1
    open(filename) do f
        while k ≤ K
            ws[k] = map(x->pol*parse(Float64, x), split(readline(f)))
            k += 1
        end
    end
    return ws
end

init_ws(filename::AbstractString) = parse_ws(filename)
init_ws(ws::Vec2) = ws
init_ws(NK::Tuple{Int,Int}) = ((N,K) = NK; Vec[rand(-1.0:2.0:1.0, N) for k = 1:K])

function generate_magsfile_name(magsfile_template::AbstractString, pol, seed, K, M, max_iters, damping, ϵ)
    return replace(magsfile_template,
        r"%(pol|seed|K|M|max_iters|damping|ϵ)%", r->begin
            r == "%pol%" && return pol
            r == "%seed%" && return seed
            r == "%K%" && return K
            r == "%M%" && return M
            r == "%max_iters%" && return max_iters
            r == "%ϵ%" && return ϵ
            error("wat")
        end)
end

function main(init::Union{AbstractString, Vec2, Tuple{Int,Int}},
              initpatt::Union{AbstractString, Tuple{Vec2,Vec}, Float64},
              ginitpatt::Union{AbstractString, Tuple{Vec2,Vec}, Float64};
              pol::Union{Vec,Float64} = 0.99,
              polτ::Float64 = 0.0,
              max_iters::Int = 1000,
              seed::Int = 1,
              ϵ::Float64 = 1e-5,
              damping::Real = 0.0,
              β::Float64 = Inf,
              accuracy1::Symbol = :exact,
              accuracy2::Symbol = :exact,
              magsfile_template = nothing,
              randfact::Float64 = 0.01)

    @assert all(p->(0 ≤ p ≤ 1), pol)
    pol::MagVec = Mag64[pol...]

    srand(seed)

    ws = init_ws(init)
    K = length(ws)
    N = length(ws[1])

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

    messages = Messages(M, N, K, randfact)

    params = Params(damping, ϵ, β, max_iters, accuracy1, accuracy2, 0.0, 0.0, 0.0, false)

    set_outfields!(messages, patterns.output, params.β)

    tr_errs = Dict{Mag64,Int}()
    tst_errs = Dict{Mag64,Int}()

    nb_tr_errs = Dict{Mag64,Int}()
    nb_tst_errs = Dict{Mag64,Int}()

    magsfile_template == nothing && (magsfile_template = "mags_%pol%_s%seed%.tst")
    @assert isa(magsfile_template, AbstractString)

    ok = false
    for i = 1:length(pol)
        println("pol=$(pol[i])")
        update_pol!(messages, ws, pol[i])
        if i == 1
            set_τ1_fields!(messages, ws, polτ[1], patterns)
            tr_errs0 = test(messages, patterns)
            tst_errs0 = test(messages, gpatterns)
            println("initial training errors = ", tr_errs0)
            println("initial general. errors = ", tst_errs0)
        end

        ok = converge!(messages, patterns, params)

        mags_outfile = generate_magsfile_name(magsfile_template, pol[i], seed, K, M, max_iters, damping, ϵ)
        println("mags_outfile=$mags_outfile")
        #=ok && =#open(mags_outfile, "w") do f
            print_mags(f, messages)
        end

        println("flipped=", compare_signs(messages, ws))
        tr_errs[pol[i]] = test(messages, patterns)
        tst_errs[pol[i]] = test(messages, gpatterns)
        println("training errors = ", tr_errs[pol[i]], " / $M [ ", 100 * tr_errs[pol[i]] / M, " % ]")
        println("general. errors = ", tst_errs[pol[i]], " / $gM [ ", gM > 0 ? 100 * tst_errs[pol[i]] / gM : 0.0, " % ]")
        println("  ---")
        nb_tr_errs[pol[i]] = nonbayes_test(messages, patterns)
        nb_tst_errs[pol[i]] = nonbayes_test(messages, gpatterns)
        println("nonb. training errors = ", nb_tr_errs[pol[i]], " / $M [ ", 100 * nb_tr_errs[pol[i]] / M, " % ]")
        println("nonb. general. errors = ", nb_tst_errs[pol[i]], " / $gM [ ", gM > 0 ? 100 * nb_tst_errs[pol[i]] / gM : 0.0, " % ]")
        println("-------------")
        γ = atanh(pol[i])
        F = free_energy(messages, patterns, ws, γ)
        S = overlap(messages, ws)
        Σ = entropy(F, S, γ)
        println("free energy = ", F)
        println("overlap     = ", S)
        println("entropy     = ", Σ)
        #ok || (pol = pol[1:i]; break)
    end

    results = DataFrame(pol=pol,
                        terr=[tr_errs[p] for p in pol],
                        terrp=[100*tr_errs[p]/M for p in pol],
                        gerr=[tst_errs[p] for p in pol],
                        gerrp=[100*tst_errs[p]/gM for p in pol])

    println("summary:")
    println("--------")
    println(results)
    println()

    return ok, patterns, messages #, BitVector[ms.>0.0 for ms in messages.mw]
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
        #vH = Float64(H)

        σ = computeσ(vh, ξ)
        μ = dot(vh, ξ)

        b = merf(μ / σ)

        #f -= log((1 + vH) / 2 * (1 + b) / 2 + (1 - vH) / 2 * (1 - b) / 2)
        #f -= log((1 + vH * b) / 2)
        f -= log1pxy(H, b)
        @assert isfinite(f)

        for i = 1:N
            #f += log((1+vh[i])/2 * (1+u[i])/2 + (1-vh[i])/2 * (1-u[i])/2)
            #f += log((1 + vh[i] * u[i]) / 2)
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
        #@show pp + pm

        #f -= log((1 + vH) / 2 * pp + (1 - vH) / 2 * pm)
        b = Mag64(pp, pm)
        f -= log1pxy(H, b)
        @assert isfinite(f)

        for i = 1:N
            #f += log((1+vh[i])/2 * (1+u[i])/2 + (1-vh[i])/2 * (1-u[i])/2)
            #f += log((1 + vh[i] * u[i]) / 2)
            f += log1pxy(h[i], u[i])
        end
        return f
    end
end

function free_energy(messages::Messages, patterns::Patterns, ws::Vec2, γ::Float64)
    @extract messages M N K mw mτ1 uw Uτ1 mτ2 uτ1
    @extract patterns X output

    f = 0.0

    for a = 1:M
        ξ = X[a]
        #out = output[a]
        for k = 1:K
            f += free_energy_theta(mw[k], mτ1[a][k], ξ, uw[a][k], Uτ1[a][k])
        end
        #f += free_energy_theta(mτ1[a], mτ2[a], ones(K), uτ1[a], 0.0 #=Uτ2[a]=#)
        f += free_energy_theta_exact(mτ1[a], mτ2[a], ones(K), uτ1[a], zero(Mag64) #=Uτ2[a]=#)
    end

    zkip = [zeros(N) for k = 1:K]
    zkim = [zeros(N) for k = 1:K]
    for a = 1:M, k = 1:K, i = 1:N
        zkip[k][i] += log((1 + uw[a][k][i]) / 2) # XXX mag?
        zkim[k][i] += log((1 - uw[a][k][i]) / 2)
    end

    for k = 1:K, i = 1:N
        ap = γ * ws[k][i] + zkip[k][i]
        am = -γ * ws[k][i] + zkim[k][i]

        zki = exp(ap) + exp(am)
        #@show  γ,ws[k][i],zkip[k][i],zkim[k][i]
        f -= log(zki)
        @assert isfinite(f)
        #println("k=$k i=$i zki=$zki f=$f")
    end

    return -f / (N * K)
end

# used with pseudo-reinforcement.
# Would be nice to merge with the other (note that the other has the wrong sign)
function free_energy2(messages::Messages, patterns::Patterns, params::Params)
    @extract messages : M N K ux mw mτ1 uw Uτ1 mτ2 uτ1
    @extract patterns : X output
    @extract params   : r pol

    f = 0.0

    for a = 1:M
        ξ = X[a]
        #out = output[a]
        for k = 1:K
            f += free_energy_theta(mw[k], mτ1[a][k], ξ, uw[a][k], Uτ1[a][k])
        end
        #f += free_energy_theta(mτ1[a], mτ2[a], ones(K), uτ1[a], 0.0 #=Uτ2[a]=#)
        f += free_energy_theta_exact(mτ1[a], mτ2[a], ones(K), uτ1[a], zero(Mag64) #=Uτ2[a]=#)
    end

    #zkip = [zeros(N) for k = 1:K]
    #zkim = [zeros(N) for k = 1:K]
    #for a = 1:M, k = 1:K, i = 1:N
    #    zkip[k][i] += log((1 + uw[a][k][i]) / 2)
    #    zkim[k][i] += log((1 - uw[a][k][i]) / 2)
    #end

    for k = 1:K, i = 1:N
        ## This is a simplified version, see BPerc.jl for "derivation"

        ##zki = ((1 + ux[k][i]) * exp(zkip[k][i]) + (1 - ux[k][i]) * exp(zkim[k][i])) / 2
        #zkip = log((1 + ux[k][i]) / 2)
        #zkim = log((1 - ux[k][i]) / 2)
        #for a = 1:M
        #    zkip += log((1 + uw[a][k][i]) / 2)
        #    zkim += log((1 - uw[a][k][i]) / 2)
        #end
        #zki = exp(zkip) + exp(zkim)
        #f -= log(zki)

        f -= logZ(ux[k][i], Mag64[uw[a][k][i] for a = 1:M])

        f -= logtwo / 2
        #f += log((1 - pol^2) / 2) / 2
        f += log1pxy(pol, -pol) / 2
        hkix = mw[k][i] ⊘ ux[k][i]
        #f += log((1 + hkix * ux[k][i]) / 2)
        f += log1pxy(hkix, ux[k][i])
        hpol = hkix * pol
        mx = hpol ↑ (r + 1)
        #f -= xlogy((1 + mx) / 2, (1 + hpol) / 2) + xlogy((1 - mx) / 2, (1 - hpol) / 2)
        f += mcrossentropy(mx, hpol)
    end

    return f / (N * K)
end

# This returns the free etropy of the replicated model, including the w̃ nodes.
# NOTE: not density; not divided by y
function free_entropy(messages::Messages, patterns::Patterns, params::Params)
    @extract messages : M N K ux mw mτ1 uw Uτ1 mτ2 uτ1
    @extract patterns : X output
    @extract params   : r pol

    f = 0.0

    for a = 1:M
        ξ = X[a]
        #out = output[a]
        for k = 1:K
            f += free_energy_theta(mw[k], mτ1[a][k], ξ, uw[a][k], Uτ1[a][k])
        end
        #f += free_energy_theta(mτ1[a], mτ2[a], ones(K), uτ1[a], 0.0 #=Uτ2[a]=#)
        f += free_energy_theta_exact(mτ1[a], mτ2[a], ones(K), uτ1[a], zero(Mag64) #=Uτ2[a]=#)
    end

    zkip = [zeros(N) for k = 1:K]
    zkim = [zeros(N) for k = 1:K]
    for a = 1:M, k = 1:K, i = 1:N
        zkip[k][i] += log((1 + uw[a][k][i]) / 2) # XXX mag?
        zkim[k][i] += log((1 - uw[a][k][i]) / 2)
    end

    for k = 1:K, i = 1:N
        ## Slight simplification below
        # zki = ((1 + ux[k][i]) * zkip[k][i] + (1 - ux[k][i]) * zkim[k][i]) / 2
        # f -= log(zki)                                                                    # ki varnode

        # hkix = Float64(mw[k][i] ⊘ ux[k][i])
        # hxki = tanh(r * atanh(hkix * pol))
        # f -= log((1 + hkix * hxki * pol) / √(1 - pol^2))                                 # γ node
        # f += log((1 + hkix * ux[k][i]) / 2)                                              # ki ↔ γ edge
        # f += log((1 + hxki * hkix * pol) / 2)                                            # γ ↔ x node
        # f -= log(((1 + hkix * pol) / 2)^(r+1) + ((1 - hkix * pol) / 2)^(r+1)) / (r + 1)  # x varnode

        zki = (1 + ux[k][i]) * exp(zkip[k][i]) + (1 - ux[k][i]) * exp(zkim[k][i])
        f -= log(zki)
        #f += log(1 - pol^2) / 2
        f += log1pxy(pol, -pol) / 2 + log(2)/2
        # XXX mag? ↓
        hkix = Float64(mw[k][i] ⊘ ux[k][i])
        f += log((1 + hkix * ux[k][i]) / 2)
        f -= log(((1 + hkix * pol) / 2)^(r+1) + ((1 - hkix * pol) / 2)^(r+1)) / (r + 1)
    end

    return f * (r + 1)
end

function overlap(messages::Messages, ws::Vec2)
    @extract messages K N mw
    S = 0.0
    for k = 1:K, i = 1:N
        S += mw[k][i] * ws[k][i]
    end
    return S / (N * K)
end

entropy(F::Float64, S::Float64, γ::Float64) = F - γ * S
entropy(messages::Messages, patterns::Patterns, ws::Vec2, γ::Float64) = free_energy(messages, patterns, ws, γ) - γ * overlap(messages, ws)

function compute_S(messages::Messages, params::Params)
    @extract messages : N K ux mw
    @extract params   : r pol
    S = 0.0
    for k = 1:K, i = 1:N
        hkix = mw[k][i] ⊘ ux[k][i]
        hxki = (hkix * pol) ↑ r
        hh = hkix * hxki
        #S += (hkix * hxki + pol) / (1 + hkix * hxki * pol)
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

function solve(N::Int, K::Int,
               initpatt::Union{AbstractString, Tuple{Vec2,Vec}, Integer},
               ginitpatt::Union{AbstractString, Tuple{Vec2,Vec}, Integer};
               max_iters::Int = 1000,
               seed::Int = 1,
               damping::Real = 0.0,
               β::Float64 = Inf,
               accuracy1::Symbol = :exact,
               accuracy2::Symbol = :exact,
               randfact::Float64 = 0.01,
               dγ::Float64 = 0.01)

    srand(seed)

    isa(initpatt, Integer) && (initpatt = (N, initpatt))
    isa(ginitpatt, Integer) && (ginitpatt = (N, ginitpatt))

    print("generating patterns... ")
    print("T")
    patterns = Patterns(initpatt)
    print("G")
    gpatterns = Patterns(ginitpatt)
    println(" done")

    M = patterns.M
    gM = gpatterns.M

    messages = Messages(M, N, K, randfact)

    params = Params(damping, 1e-3, β, max_iters, accuracy1, accuracy2, 0.0, 0.0, dγ, false)

    set_outfields!(messages, patterns.output, params.β)

    #ok = converge!(messages, patterns, params)
    params.damping = 0.0
    ok = rein_solve!(messages, patterns, params)

    b_tr_errs = test(messages, patterns)
    b_tst_errs = test(messages, gpatterns)
    println("training errors = ", b_tr_errs, " / $M [ ", 100 * b_tr_errs / M, " % ]")
    println("general. errors = ", b_tst_errs, " / $gM [ ", gM > 0 ? 100 * b_tst_errs / gM : 0.0, " % ]")
    println("  ---")
    nb_tr_errs = nonbayes_test(messages, patterns)
    nb_tst_errs = nonbayes_test(messages, gpatterns)
    println("nonb. training errors = ", nb_tr_errs, " / $M [ ", 100 * nb_tr_errs / M, " % ]")
    println("nonb. general. errors = ", nb_tst_errs, " / $gM [ ", gM > 0 ? 100 * nb_tst_errs / gM : 0.0, " % ]")
    println("-------------")

    return ok
end

function mags_symmetry(messages::Messages)
    @extract messages N K mw
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
    #set_outfields!(messages, patterns.output, params.β)

    outfile == "" && (outfile = "results_BPCR_N$(N)_K$(K)_M$(M)_s$(seed).txt")
    outmessfiletmpl == "" && (outmessfiletmpl = "messages_BPCR_N$(N)_K$(K)_M$(M)_g%gamma%_s$(seed).txt.gz")
    #outfile ≠ nothing && !force_overwrite && isfile(outfile) && error("file exists: $outfile")
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
            #βE = E == 0 ? 0.0 : β1 * E
            Σint = -βF - γ * S #+ βE
            #Σext = ext_entropy(messages, params)
            #Ẽ = error_prob(messages, patterns, params)

            #r = params.r
            #δr = 1e-3
            #Φ0 = free_entropy(messages, patterns, params)
            #params.r += δr
            #ok1 = converge!(messages, patterns, params)
            #Φ1 = free_entropy(messages, patterns, params)
            #params.r -= δr

            #βF = Φ0 / (N * K * (r+2))
            #Φ′ = (Φ1 - Φ0) / δr
            #Σint = -Φ′ / (N * K) - γ * S

            println("it=$it pol=$pol y=$y β=$β (ok=$ok) S=$S βF=$βF Σᵢ=$Σint q=$q q̃=$q̃ Ẽ=$errs")
            #println("  Σ2 = ", free_energy(messages, patterns, params, ws, γ) - γ * S)
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

function compute_errors(patterns::Patterns, ws::Vec2)
    @extract patterns X output
    sum([o .≠ fsign(sum([fsign(ξ ⋅ wsk) for wsk in ws])) for (ξ,o) in zip(X,output)])
end

let hs = Dict{Int,MagVec}(), vhs = Dict{Int,Vec}()
    global sorted_perm
    function sorted_perm(messages::Messages, ws::Vec2, pol::Mag64)
        @extract messages N K mw

        NK = N * K

        h = Base.@get!(hs, N, Array(Mag64, N))
        vh = Base.@get!(vhs, NK, Array(Float64, NK))

        for k = 1:K
            @assert ws[k] == 1 || ws[k] == -1
            subfield!(h, mw[k], copysign(pol, ws[k]))
            @inbounds for i = 1:N
                vh[N * (k-1) + i] = ws[k][i] * h[i]
                # = ws[i] * mw[i]
            end
        end
        sp = sortperm(vh)
        nn = findfirst(x->x≥0, vh[sp]) - 1
        nn == -1 && (nn = length(sp))
        return sortperm(vh), nn
    end
end

function randomwalk(init::Union{AbstractString, Vec2, Tuple{Int,Int}},
                    initpatt::Union{AbstractString, Tuple{Vec2,Vec}, Float64};
                    pol::Union{Vec,Float64} = 0.2,
                    max_iters::Int = 1000,
                    rw_max_iters::Int = 100,
                    seed::Int = 1,
                    ϵ::Float64 = 1e-5,
                    damping::Real = 0.0,
                    β::Float64 = Inf,
                    y = Inf,
                    flip_frac::Float64 = 0.1,
                    early_giveup::Bool = false,
                    accuracy1::Symbol = :accurate,
                    accuracy2::Symbol = :exact,
                    randfact::Float64 = 0.01)

    @assert all(p->(0 ≤ p ≤ 1), pol)
    pol::MagVec = Mag64[pol...]

    srand(seed)

    ws = init_ws(init)
    K = length(ws)
    N = length(ws[1])

    #println("ws=$ws")

    isa(initpatt, Float64) && (initpatt = (N, round(Int, K * N * initpatt)))

    print("generating patterns... ")
    print("T")
    patterns = Patterns(initpatt)
    println(" done")

    #println("patts=", patterns)

    M = patterns.M
    @show K,N,M

    messages = Messages(M, N, K, randfact)

    params0 = Params(0.9, ϵ, β, max_iters, :accurate, accuracy2, 0.0, 0.0, 0.0, false)
    params = Params(damping, ϵ, β, max_iters, accuracy1, accuracy2, 0.0, 0.0, 0.0, true)

    set_outfields!(messages, patterns.output, params.β)

    tr_errs = Dict{Float64,Int}()
    tst_errs = Dict{Float64,Int}()

    ok = converge!(messages, patterns, params0)
    isa(init, Tuple{Int,Int}) && (ws = [Float64[sign0(m) for m in mk] for mk in messages.mw])
    wbest = copy(ws)
    messages_best = deepcopy(messages)

    flip_num0 = max(1, round(Int, flip_frac * N * K))

    init_accuracy = accuracy1

    ok = false
    for ip = 1:length(pol)
        params.accuracy1 = init_accuracy
        copy!(messages, messages_best)
        cpol = pol[ip]
        γ = atanh(cpol)
        println("===============================")
        println("pol=$cpol")
        update_pol!(messages, ws, cpol)

        ok = converge!(messages, patterns, params)
        #=for a = 1:min(2,M)
            writedlm(STDOUT, map(atanh,messages.uw[a])')
        end=#
        F0 = free_energy(messages, patterns, ws, γ)
        S0 = overlap(messages, ws)
        Σ0 = entropy(F0, S0, γ)
        ΣM,SM,FM = Σ0,S0,F0
        copy!(wbest, ws)
        println("initial F  = $F0")
        println("        Σ  = $Σ0")
        println("        S  = $S0")
        errs = compute_errors(patterns, ws)
        println("        err = $errs ( $(100 * errs / M)% )")
        perm = collect(1:(N*K))
        copy!(messages_best, messages)
        bk_messages = deepcopy(messages)
        fast_track = true
        #fast_track = false
        for y1 in y, it = 1:rw_max_iters
            println("IT = $it (y=$y1)")
            #shuffle!(perm)
            if fast_track
                perm, nn = sorted_perm(messages, ws, cpol)
            else
                #shuffle!(perm)
                perm = shuffle!(repeat(collect(1:(N*K)), outer=[10]))
                nn = 0
                #nn = 10
                #nn = flip_num0
            end
            copy!(bk_messages, messages)

            flip_num = max(1, min(flip_num0, nn))
            println("nn = $nn flip_num = $flip_num")
            accepted = false
            j = 0
            while !accepted
            #for i in perm
                print(".")
                for k = 1:flip_num
                    i = perm[j + k]
                    flip_field!(messages, ws, i)
                end
                ok = converge!(messages, patterns, params)
                F1 = free_energy(messages, patterns, ws, γ)
                S1 = overlap(messages, ws)
                Σ1 = entropy(F1, S1, γ)

                if ok && rand() < exp(y1 * (F1 - F0))
                    newmax = false
                    if F1 > FM
                        ΣM,SM,FM = Σ1,S1,F1
                        copy!(wbest, ws)
                        copy!(messages_best, messages)
                        newmax = true
                    end
                    println()
                    println("Accepted F = $F1 (diff=$(F1 - F0)) [max=$FM]", newmax ? " (*)" : "")
                    println("         Σ = $Σ1 (diff=$(Σ1 - Σ0)) [max=$ΣM]")
                    println("         S = $S1 [max=$SM]")
                    errs = compute_errors(patterns, ws)
                    println("         err = $errs ( $(100 * errs / M)% )")
                    Σ0,S0,F0 = Σ1,S1,F1
                    accepted = true
                    break
                end
                for k = 1:flip_num
                    i = perm[j + k]
                    flip_field!(messages, ws, i)
                end
                if flip_num == 1
                    j += 1
                else
                    flip_num = max(1, round(Int, flip_num / 2))
                end
                copy!(messages, bk_messages)
                j ≥ length(perm) && break
                if fast_track && j == 1
                    early_giveup && break
                    print("[!]")
                    j = 0
                    shuffle!(perm)
                    fast_track = false
                end
            end
            if !accepted
                if params.accuracy1 == :none
                    params.accuracy1 = :accurate
                    #=println("[A!]")
                    ok = converge!(messages, patterns, params)
                    @assert ok
                    Σ1 = entropy(messages, patterns)
                    S1 = overlap(messages, ws)
                    F1 = Σ1 + S1 * γ
                    Σ0,S0,F0 = Σ1,S1,F1=#
                    ok = converge!(messages_best, patterns, params)
                    @assert ok
                    FM = free_energy(messages, patterns, ws, γ)
                    SM = overlap(messages, ws)
                    ΣM = entropy(FM, SM, γ)
                    println()
                    println("Accurate F = $FM")
                    println("         Σ = $ΣM")
                    println("         S = $SM")
                end
                #else
                    println("[Giving up]")
                    break
                #end
            end
        end
    end

    return ok, wbest
end

end # module
