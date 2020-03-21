
#   This file is part of Adapode.jl. It is licensed under the AGPL license
#   Adapode Copyright (C) 2019 Michael Reed

export assemble, assembleglobal, assemblestiffness, assembleconvection, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assembledivergence
export assembleload, assemblemassload, assemblerobin, edges, edgesindices
export gradienthat, gradientCR, submesh, detsimplex, iterable, callable, value
export solvepoisson, solveSD, solvehomogeneous, boundary, interior
import Grassmann: points

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

assembleglobal(M,t,m=detsimplex(t),c=1,g=0) = assembleglobal(M,t,iterable(t,m),iterable(t,c),iterable(t,g))
function assembleglobal(M,t,m::T,c::C,g::F) where {T<:AbstractVector,C<:AbstractVector,F<:AbstractVector}
    p = points(t); np = length(p); A = spzeros(np,np)
    for k ∈ 1:length(t)
        assemblelocal!(A,M(c[k],g[k],Val(ndims(p))),m[k],value(t[k]))
    end
    return A
end

assemblefunction(t,f,m=detsimplex(t)) = assemblefunction(t,iterpts(t,f),iterable(t,m))
function assemblefunction(t,f::F,m::V) where {F<:AbstractVector,V<:AbstractVector}
    b,l = zeros(length(points(t))),m/ndims(t)
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

hgshift(hk::Chain{V}) where V = Chain{V(2,3),1}(SVector(-hk[3],hk[2]))
function gradienthat(t,m=detsimplex(t))
    if ndims(t) == 2
        inv.(getindex.(value(m),1))
    elseif ndims(t) == 3
        h = curls(t,value(points(t))/(ndims(t)-1))./getindex.(value(m),1)
        [Chain{Manifold(h),1}(SVector(hgshift.(value(h[k])))) for k ∈ 1:length(h)]
    else
        throw(error("hat gradient on Manifold{$(ndims(t))} not defined"))
    end
end

gradientCR(t,m) = gradientCR(gradienthat(t,m))
gradientCR(g) = gradientCR.(g)
function gradientCR(g::Chain{V}) where V
    Chain{V,1}(g.⋅SVector(
        Chain{V,1}(SVector{3,Int}(-1,1,1)),
        Chain{V,1}(SVector{3,Int}(1,-1,1)),
        Chain{V,1}(SVector{3,Int}(1,1,-1))))
end

function assembledivergence(t,m,g)
    p = points(t); np = length(p);
    D1,D2 = spzeros(np,np), spzeros(np,np)
    for k ∈ 1:length(t)
        tk,gm = value(t[k]),g[k]*m[k]
        for i ∈ tk
            D1[k,tk[i]] = gm[i][1]
            D2[k,tk[i]] = gm[i][2]
        end
    end
    return D1,D2
end

stiffness(c,g,::Val{2}) = SMatrix{2,2,Int}([1 -1; -1 1])*(c*g^2)
function stiffness(c,g,::Val{3})
    A = zeros(MMatrix{3,3,Float64})
    for i ∈ 1:3, j ∈ 1:3
        A[i,j] += c*(g[i]⋅g[j])[1]
    end
    return SMatrix{3,3,Float64}(A)
end
assemblestiffness(t,c=1,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(c≠1 ? means(t) : t,c),g)
# iterable(means(t),c) # mapping of c.(means(t))

convection(b,g,::Val{3}) = ones(SVector{3,Int})*getindex.((b/3).⋅value(g),1)'
assembleconvection(t,b,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(convection,t,m,b,g)

SD(b,g,::Val{3}) = (x=getindex.(b.⋅value(g),1);x*x')
assembleSD(t,b,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(SD,t,m,b,g)

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

function assemble(t,c=1,f=0,m=detsimplex(t),g=gradienthat(t,m))
    M,b = assemblemassfunction(t,f,m,m)
    return assemblestiffness(t,c,m,g),M,b
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

function solveSD(t,e,c,f,δ,κ,gD=0,gN=0)
    m = detsimplex(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,c,m,g)
    b = means(t,f(p))
    C = assembleconvection(t,b,m,g)
    Sd = assembleSD(t,sqrt(δ)*b,m,g)
    R,r = assemblerobin(e,κ,gD,gN)
    return (A-C'+R+Sd)\r
end

function solvehomogeneous(e,M,b)
    free = interior(e)
    ξ = zeros(length(points(e)))
    ξ[free] = M[free,free]\b[free]
    return ξ
end

boundary(e) = sort!(unique(vcat(value.(value(e))...)))
interior(e) = sort!(setdiff(1:length(points(e)),boundary(e)))

edges(t::ChainBundle) = edges(value(t))
function edges(t,i=getindex.(t,1),j=getindex.(t,2),k=getindex.(t,3))
    np = length(points(t))
    A = sparse(j,k,1,np,np)+sparse(i,k,1,np,np)+sparse(i,j,1,np,np)
    f = findall(x->x>0,LinearAlgebra.triu(A+transpose(A)))
    p = points(t); V = p(2:ndims(p)...); N = ndims(V)
    e = [SVector{N,Int}(f[n].I) for n ∈ 1:length(f)]
    M = ChainBundle(means(e,p))(2:(N+1)...)
    return Chain{M,1}.(e)
end

edgesindices(t::ChainBundle) = edgesindices(value(t))
function edgesindices(t,i=getindex.(t,1),j=getindex.(t,2),k=getindex.(t,3))
    np,nt = length(points(t)),length(t)
    e = edges(t,i,j,k)
    A = sparse(getindex.(e,1),getindex.(e,2),1:length(e),np,np)
    V = parent(e); nV = ndims(V); A = A+A'
    e,[Chain{V,1}(SVector{nV,Int}(A[j[n],k[n]],A[i[n],k[n]],A[i[n],j[n]])) for n ∈ 1:nt]
end
