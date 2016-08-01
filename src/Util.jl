# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

module Util

export exclusive,
       @readmagvec, @dumpmagvecs,
       IVec, Vec, Vec2, MagVec, MagVec2, MagVec3

using ..Mag64, ..showinner, ..parseinner

if success(`which lockfile`)
    function exclusive(f::Function, fn::AbstractString = "lock.tmp")
        run(`lockfile -1 $fn`)
        try
            f()
        finally
            run(`rm -f $fn`)
        end
    end
else
    warn("could not find the `lockfile` program. Try installing it (it's included with `procmail` – www.procmail.org).")
    exclusive(f::Function, fn::AbstractString) = nothing
end

typealias IVec Vector{Int}
typealias Vec Vector{Float64}
typealias Vec2 Vector{Vec}
typealias MagVec Vector{Mag64}
typealias MagVec2 Vector{MagVec}
typealias MagVec3 Vector{MagVec2}

macro readmagvec(l, fmt, vs...)
    ex = :()
    for v in vs
        vn = Regex(string(v, "(\\[|\\s)"))
        ex = quote
            $ex
            if ismatch($vn, $(esc(l)))
                _readmagvec($(esc(l)), $(esc(fmt)), $(esc(v)))
                @goto found
            end
        end
    end
    ex = quote
        $ex
        error("unrecognized line $l")
        @label found
    end
    ex
end

function _readmagvec{F}(l::AbstractString, fmt::Type{Val{F}}, v::Array)
    sl = split(l)
    pre = sl[1]
    ismatch(r"^[^[]+(?:\[\d+\])*$", pre) || error("invalid messages file")
    inds = map(x->parse(Int,x), split(replace(pre, r"[][]", " "))[2:end])
    for i in inds
        v = v[i]
    end
    length(v) == length(sl) - 1 || error("invalid messages file")
    for i = 1:length(v)
        v[i] = parseinner(fmt, sl[i+1])
    end
end

macro dumpmagvecs(io, vs...)
    ex = :()
    for v in vs
        vn = string(v)
        ex = quote
            $ex
            _dumpmagvec($(esc(io)), $(esc(v)), $vn)
        end
    end
    ex
end

function _dumpmagvec{T<:Array}(io::IO, a::Array{T}, s::AbstractString)
    for (i,v) in enumerate(a)
        _dumpmagvec(io, v, "$s[$i]")
    end
end
function _dumpmagvec(io::IO, a::Array, s::AbstractString)
    print(io, s)
    for x in a
        print(io, ' ')
        showinner(io, x)
    end
    println(io)
end


end
