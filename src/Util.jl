# This file is a part of BinaryCommitteeMachineFBP.jl. License is MIT: http://github.com/carlobaldassi/BinaryCommitteeMachineFBP.jl/LICENCE.md

module Util

export exclusive, checkdims, chgeltype,
       @readmagvec, @dumpmagvecs,
       IVec, Vec, Vec2, MagVec, MagVec2, MagVec3,
       @inner

using Compat
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

annotate_new!(T, ex) = ex
function annotate_new!(T, ex::Expr)
    if ex.head == :call && ex.args[1] == :new
        ex.args[1] = Expr(:curly, :new, T)
    end
    map!(x->annotate_new!(T, x), ex.args, ex.args)
    return ex
end

# horrible macro to keep compatibility with both julia 0.5 and 0.6,
# while avoiding some even more horrible syntax
macro inner(T, ex)
    VERSION < v"0.6-" && return esc(ex)
    @assert Base.Meta.isexpr(ex, [:(=), :function])
    @assert length(ex.args) == 2
    @assert isa(ex.args[1], Expr) && ex.args[1].head == :call
    @assert isa(ex.args[1].args[1], Symbol)
    fn = ex.args[1].args[1]
    fargs = ex.args[1].args[2:end]
    body = ex.args[2]
    annotate_new!(T, body)

    return esc(Expr(ex.head, Expr(:where, Expr(:call, Expr(:curly, fn, T), fargs...), T), body))
end

const IVec = Vector{Int}
const Vec = Vector{Float64}
const Vec2 = Vector{Vec}
@compat const MagVec{M<:Mag64} = Vector{M}
@compat const MagVec2{M<:Mag64} = Vector{MagVec{M}}
@compat const MagVec3{M<:Mag64} = Vector{MagVec2{M}}

function checkdims{V<:AbstractArray}(x::AbstractArray{V}, N::Integer, r::Integer...)
    @assert length(x) == N
    @assert all(v->checkdims(v, r...), x)
    return true
end
function checkdims(x::AbstractArray, N::Integer)
    @assert length(x) == N
    return true
end
checkdims{V<:AbstractArray}(x::AbstractArray{V}, N::Integer) = error("too few dimensions")
checkdims(x::AbstractArray, N::Integer, r::Integer...) = error("not enough dimensions")

changedeltype{V<:Vector,T}(::Type{Vector{V}}, ::Type{T}) = Vector{changedeltype(V, T)}
changedeltype{X,T}(::Type{Vector{X}}, ::Type{T}) = Vector{T}

function chgeltype{V<:Vector,T}(x::Vector{V}, ::Type{T})
    eltype(changedeltype(typeof(x), T))[chgeltype(y, T) for y in x]
end
chgeltype{X,T}(x::Vector{X}, ::Type{T}) = convert(Vector{T}, x)


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
        error("unrecognized line: ", $(esc(l)))
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
