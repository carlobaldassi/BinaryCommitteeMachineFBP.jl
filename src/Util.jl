# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

module Util

export exclusive, checkdims, chgeltype,
       @readmagvec, @dumpmagvecs,
       IVec, Vec, Vec2, MagVec, MagVec2, MagVec3

using ..Magnetizations: Mag64, showinner, parseinner

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

annotate_new!(T, ex) = ex
function annotate_new!(T, ex::Expr)
    if ex.head == :call && ex.args[1] == :new
        ex.args[1] = Expr(:curly, :new, T)
    end
    map!(x->annotate_new!(T, x), ex.args, ex.args)
    return ex
end

const IVec = Vector{Int}
const Vec = Vector{Float64}
const Vec2 = Vector{Vec}
const MagVec{M<:Mag64} = Vector{M}
const MagVec2{M<:Mag64} = Vector{MagVec{M}}
const MagVec3{M<:Mag64} = Vector{MagVec2{M}}

function checkdims(x::AbstractArray{<:AbstractArray}, N::Integer, r::Integer...)
    @assert length(x) == N
    @assert all(v->checkdims(v, r...), x)
    return true
end
function checkdims(x::AbstractArray, N::Integer)
    @assert length(x) == N
    return true
end
checkdims(x::AbstractArray{<:AbstractArray}, N::Integer) = error("too few dimensions")
checkdims(x::AbstractArray, N::Integer, r::Integer...) = error("not enough dimensions")

changedeltype(::Type{Vector{V}}, ::Type{T}) where {V<:Vector,T} = Vector{changedeltype(V, T)}
changedeltype(::Type{Vector{X}}, ::Type{T}) where {X,T} = Vector{T}

function chgeltype(x::Vector{<:Vector}, ::Type{T}) where {T}
    eltype(changedeltype(typeof(x), T))[chgeltype(y, T) for y in x]
end
chgeltype(x::Vector, ::Type{T}) where {T} = convert(Vector{T}, x)


macro readmagvec(l, fmt, vs...)
    ex = :()
    for v in vs
        vn = Regex(string(v, "(\\[|\\s)"))
        ex = quote
            $ex
            if occursin($vn, $(esc(l)))
                _readmagvec($(esc(l)), $(esc(fmt)), $(esc(v)))
                @goto found
            end
        end
    end
    ex = quote
        $ex
        error("unrecognized line: ", $(esc(l)))
        @label found
    end
    ex
end

function _readmagvec(l::AbstractString, fmt::Type{Val{F}}, v::Array) where {F}
    sl = split(l)
    pre = sl[1]
    occursin(r"^[^[]+(?:\[\d+\])*$", pre) || error("invalid messages file")
    inds = map(x->parse(Int,x), split(replace(pre, r"[][]" => " "))[2:end])
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

function _dumpmagvec(io::IO, a::Array{<:Array}, s::AbstractString)
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
