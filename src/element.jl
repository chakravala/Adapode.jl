
#   This file is part of Adapode.jl. It is licensed under the AGPL license
#   Adapode Copyright (C) 2019 Michael Reed

export assembleglobal, assembleconvection, hatgradient, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assemblestiffness
export assembleload, assemblemassload, assemblerobin, tri2edge, tri2tri
import Grassmann: points
export assemble, submesh, detsimplex, iterable, callable, value, solvepoisson

@inline iterpts(t,f) = iterable(points(t),f)
@inline iterable(p,f) = range(f,f,length=length(p))
@inline iterable(p,f::F) where F<:Function = f.(value(p))
@inline iterable(p,f::ChainBundle) = value(f)
@inline iterable(p,f::F) where F<:AbstractVector = f

@inline callable(c::F) where F<:Function = c
@inline callable(c) = x->c

function assemblelocal!(M,mat::SMatrix{N,N},m,tk::SVector{N}) where N
    for i ∈ 1:N, j∈ 1:N
        M[tk[i],tk[j]] += mat[i,j]*m[1]
    end
end

assembleglobal(M,t,m=detsimplex(t),c=1,v=0) = assembleglobal(M,t,iterable(t,m),iterable(t,c),iterable(t,v))
function assembleglobal(M,t,m::T,c::C,v::F) where {T<:AbstractVector,C<:AbstractVector,F<:AbstractVector}
    p = points(t); np = length(p)
    A = spzeros(np,np) # allocate global matrix
    for k ∈ 1:length(t)
        assemblelocal!(A,M(c[k],v[k],Val(ndims(p))),m[k],value(t[k]))
    end
    return A
end

assemblefunction(t,f,v=detsimplex(t)) = assemblefunction(t,iterpts(t,f),iterable(t,v))
function assemblefunction(t,f::F,v::V) where {F<:AbstractVector,V<:AbstractVector}
    b,l = zeros(length(points(t))),v/ndims(t)
    for k ∈ 1:length(t)
        tk = value(t[k])
        b[tk] .+= f[tk]*l[k][1]
    end
    return b
end

assemblemassfunction(t,f,m=detsimplex(t),L=m) = assemblemassfunction(t,iterpts(t,f),iterable(t,m),iterable(t,L))
function assemblemassfunction(t,f::F,m::V,L::T) where {F<:AbstractVector,V<:AbstractVector,T<:AbstractVector}
    np,N = length(points(t)),ndims(t)
    M,b,n,l = spzeros(np,np), zeros(np), Val(N), L/N
    for k ∈ 1:length(t)
        tk = value(t[k])
        assemblelocal!(M,mass(nothing,nothing,n),m[k],tk)
        b[tk] .+= f[tk]*l[k][1]
    end
    return M,b
end

assembleload(t,l=detsimplex(t)) = assemblefunction(t,1,l)
assemblemassload(t,m=detsimplex(t),l=m) = assemblemassfunction(t,1,m,l)

mass(a,b,::Val{N}) where N = (ones(SMatrix{N,N,Int})+I)/Int(factorial(N+1)/factorial(N-1))
assemblemass(t,m=detsimplex(t)) = assembleglobal(mass,t,iterpts(t,m))

function hatgradient(t,m=detsimplex(t))
    if ndims(t) == 2
        inv.(getindex.(value(m),1))
    else
        curls(t,value(points(t))/(ndims(t)-1))./getindex.(value(m),1)
    end
end

stiffness(c,g,::Val{2}) = SMatrix{2,2,Int}([1 -1; -1 1])*(c*g^2)
function stiffness(c,g,::Val{3})
    A = zeros(MMatrix{3,3,Float64})
    for i ∈ 1:3, j ∈ 1:3
        A[i,j] += c*(g[i]⋅g[j])[1]
    end
    return SMatrix{3,3,Float64}(A)
end
assemblestiffness(t,c=1,m=detsimplex(t),v=hatgradient(t,m)) = assembleglobal(stiffness,t,m,iterable(c≠1 ? means(t) : t,c),v)
# iterable(means(t),c) # mapping of c.(means(t))

convection(b,g,::Val{3}) = ones(SVector{3,Int})*getindex.((b/3).⋅value(g),1)'
assembleconvection(t,b,m=detsimplex(t),v=hatgradient(t,m)) = assembleglobal(convection,t,m,b,v)

SD(b,g,::Val{3}) = (x=getindex.((b/3).⋅value(g),1);x*x')
assembleSD(t,b,m=detsimplex(t),v=hatgradient(t,m)) = assembleglobal(SD,t,m,b,v)

function robinmass(c::Vector{Chain{V,G,T,X}} where {G,T,X},κ::F,a,::Val{1}) where {V,F<:Function}
    p = Grassmann.isbundle(V) ? V : DirectSum.supermanifold(V)
    [Chain{Manifold(p),0,Float64}((κ(a[k]),)) for k ∈ 1:length(c)]
end
function robinmass(c::Vector{Chain{V,G,T,X}} where {G,T,X},κ::F,a,::Val{2}) where {V,F<:Function}
    [Chain(κ(a[k])*abs(2a[k])) for k ∈ 1:length(c)] # abs(pk[1]-pk[2])
end
robinmass(c::ChainBundle,κ::F,a) where F<:Function = robinmass(value(c),κ,a,Val{ndims(c)}())
robinmass(c,κ,a=means(m)) = robinmass(c,callable(κ),a)
robinload(c,m,gD::D,gN::N,a) where {D<:Function,N<:Function} = [m[k]*gD(a[k])+gN(a[k]) for k ∈ 1:length(c)]
robinload(c,m,gD,gN,a=means(m)) = robinload(c,m,callable(gD),callable(gN),a)

function assemble(t,c=1,f=0,m=detsimplex(t),v=hatgradient(t,m))
    M,b = assemblemassfunction(t,f,m,m)
    return assemblestiffness(t,c,m,v),M,b
end

function assemblerobin(e,κ=1e6,gD=0,gN=0)
    a = means(e)
    m = robinmass(e,κ,a)
    l = robinload(e,m,gD,gN,a)
    return assemblemassload(e,m,l)
end

function solvepoisson(t,e,c,f,κ,gD=0,gN=0)
    m = detsimplex(t)
    b = assemblefunction(t,f,m)
    A = assemblestiffness(t,c,m)
    R,r = assemblerobin(e,κ,gD,gN)
    return (A+R)\(b+r)
end

