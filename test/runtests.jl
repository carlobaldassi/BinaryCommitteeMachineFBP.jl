module BCMFBPTests

using BinaryCommitteeMachineFBP
using Base.Test

using GZip

const N = 21
const K = 5
const α = 0.1
const kw = Dict([:randfact=>0.1, :seed=>135, :max_iters=>1, :damping=>0.5, :quiet=>false]);

patternsfile = "patterns.txt.gz"

fps = [Scoping(0:0.1:10, 21), PseudoReinforcement(0:0.01:1), FreeScoping([(atanh(√ρ), (2-ρ)/(1-ρ), Inf) for ρ in 0:0.01:1])]

# write your own tests here
function tst()
    errs, messages, patterns = focusingBP(N, K, α; kw...)
    @test errs == 0

    kw[:quiet] = true
    errs, messages, patterns = focusingBP(N, K, α; kw...)
    @test errs == 0
    errs, messages, patterns = focusingBP(N, K, patterns; kw...)
    @test errs == 0
    errs, messages, patterns = focusingBP(N, K, (patterns.X, float(patterns.output)); kw...)
    @test errs == 0
    errs, messages, patterns = focusingBP(N, K, patternsfile; kw...)
    @test errs == 0
    errs, messages, patterns = focusingBP(N, K, α; kw..., accuracy2=:accurate)
    @test errs == 0
    errs, messages, patterns = focusingBP(N, K, α; kw..., accuracy1=:none)
    @test errs == 0
    for fp in fps
        errs, messages, patterns = focusingBP(N, K, α; kw..., fprotocol=fp)
        @test errs == 0
    end
    errs, messages, patterns = focusingBP(N, K, α; kw..., messfmt=:plain)
    @test errs == 0
    errs, messages, patterns = focusingBP(N, K, α; kw..., initmess=messages)
    @test errs == 0

    f = tempname()
    isfile(f) && rm(f)
    try
        write_messages(f, messages)
        errs, messages, patterns = focusingBP(N, K, α; kw..., initmess=f)
        @test errs == 0
        m2 = read_messages(f, MagT64)
        errs, messages, patterns = focusingBP(N, K, α; kw..., initmess=m2)
        @test errs == 0
        m2 = read_messages(f, MagP64)
        errs, messages, patterns = focusingBP(N, K, α; kw..., initmess=m2)
        @test errs == 0
    finally
        isfile(f) && rm(f)
    end

    kw[:max_iters] = 100

    f = tempname()
    isfile(f) && rm(f)
    try
        errs, messages, patterns = focusingBP(N, K, α; kw..., fprotocol=PseudoReinforcement(0:0.01:0.99, 0.992:0.002:0.998),
                                              outatzero=false, outfile=f)
        @test errs == 0
        errs, messages, patterns = focusingBP(N, K, α; kw..., fprotocol=PseudoReinforcement(0:0.01:0.99, 0.992:0.002:0.998),
                                              outatzero=false, outfile=f, messfmt=:plain)
        @test errs == 0
    finally
        isfile(f) && rm(f)
    end

    f = tempname()
    isfile(f) && rm(f)
    ft = f * ".%gamma%"
    try
        errs, messages, patterns = focusingBP(N, K, α; kw..., fprotocol=PseudoReinforcement(0:0.01:0.99, 0.992:0.002:0.998),
                                              outatzero=false, outmessfiletmpl=ft)
        @test errs == 0
        errs, messages, patterns = focusingBP(N, K, α; kw..., fprotocol=PseudoReinforcement(0:0.01:0.99, 0.992:0.002:0.998),
                                              outatzero=false, outmessfiletmpl=ft, messfmt=:plain)
        @test errs == 0
    finally
        for fg in readdir(dirname(ft))
            startswith(fg, basename(f)) || continue
            dfg = joinpath(dirname(ft), fg)
            isfile(dfg) && rm(dfg)
        end
    end
end

tst()

end # module
