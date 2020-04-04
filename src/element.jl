
#   This file is part of Adapode.jl. It is licensed under the AGPL license
#   Adapode Copyright (C) 2019 Michael Reed

export assemble, assembleglobal, assemblestiffness, assembleconvection, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assembledivergence
export assembleload, assemblemassload, assemblerobin, edges, edgesindices, neighbors
export solvepoisson, solveSD, solvehomogeneous, solveboundary, boundary, interior
export submesh, detsimplex, iterable, callable, value, edgelengths
export gradienthat, gradientCR, gradient, interp, nedelecmean
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
    np = length(points(t)); A = spzeros(np,np)
    for k ∈ 1:length(t)
        assemblelocal!(A,M(c[k],g[k],Val(ndims(t))),m[k],value(t[k]))
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

revrot(hk::Chain{V,1},f=identity) where V = Chain{V,1}(SVector(-f(hk[2]),f(hk[1])))
function gradienthat(t,m=detsimplex(t))
    if ndims(t) == 2
        inv.(getindex.(value(m),1))
    elseif ndims(t) == 3
        h = curls(t,value(points(t))/(ndims(t)-1))./getindex.(value(m),1)
        V = Manifold(h); V2 = V(2,3)
        [Chain{V,1}(SVector(revrot.(V2.(value(h[k]))))) for k ∈ 1:length(h)]
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

gradient(t,u,m=detsimplex(t),g=gradienthat(t,m)) = [u[value(t[k])]⋅value(g[k]) for k ∈ 1:length(t)]

function interp(t,ut)
    np,nt = length(points(t)),length(t)
    A = spzeros(np,nt)
    for i ∈ 1:ndims(t)
        A += sparse(getindex.(value(t),i),1:nt,1,np,nt)
    end
    sparse(1:np,1:np,inv.(sum(A,dims=2))[:],np,np)*A*ut
end

function assembledivergence(t,m,g)
    p = points(t); np,nt = length(p),length(t)
    D1,D2 = spzeros(nt,np), spzeros(nt,np)
    for k ∈ 1:length(t)
        tk,gm = value(t[k]),g[k]*m[k][1]
        for i ∈ 1:ndims(t)
            D1[k,tk[i]] = gm[i][1]
            D2[k,tk[i]] = gm[i][2]
        end
    end
    return D1,D2
end

stiffness(c,g,::Val{2}) = SMatrix{2,2,Int}([1 -1; -1 1])*(c*g^2)
function stiffness(c,g,::Val{3})
    A = zeros(MMatrix{3,3,typeof(c)})
    for i ∈ 1:3, j ∈ 1:3
        A[i,j] += c*(g[i]⋅g[j])[1]
    end
    return SMatrix{3,3,typeof(c)}(A)
end
assemblestiffness(t,c=1,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(isreal(c) ? t : means(t),c),g)
# iterable(means(t),c) # mapping of c.(means(t))

convection(b,g,::Val{3}) = ones(SVector{3,Int})*getindex.((b/3).⋅value(g),1)'
assembleconvection(t,b,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(convection,t,m,b,g)

SD(b,g,::Val{3}) = (x=getindex.(b.⋅value(g),1);x*x')
assembleSD(t,b,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(SD,t,m,b,g)

function nedelec(λ,g,v::Val{3})
    f = stiffness(λ,g,v)
    m11 = (f[3,3]-f[2,3]+f[2,2])/6
    m22 = (f[1,1]-f[1,3]+f[3,3])/6
    m33 = (f[2,2]-f[1,2]+f[1,1])/6
    m12 = (f[3,1]-f[3,3]-2f[2,1]+f[2,3])/12
    m13 = (f[3,2]-2f[3,1]-f[2,2]+f[2,1])/12
    m23 = (f[1,2]-f[1,1]-2f[3,2]+f[3,1])/12
    @SMatrix [m11 m12 m13; m12 m22 m23; m13 m23 m33]
end

function rotrot(λ,g,μ,len,fhat)
    AK = (inv(μ).+nedelec(λ,g,Val(3))).*(len*len')
    bK = getindex.(fhat.⋅value(curl(g)),1).*len
    return AK,bK
end

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
    b = means(t,f)
    C = assembleconvection(t,b,m,g)
    Sd = assembleSD(t,sqrt(δ)*b,m,g)
    R,r = assemblerobin(e,κ,gD,gN)
    return (A+R-C'+Sd)\r
end

function solveboundary(A,b,fixed,boundary=zeros(length(fixed)))
    neq = length(b)
    free = sort!(setdiff(1:neq,fixed))
    ξ = zeros(eltype(b),neq)
    ξ[fixed] = boundary
    ξ[free] = A[free,free]\(b[free]-A[free,fixed]*boundary)
    return ξ
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
    np,N = length(points(t)),ndims(t); M = points(t)(2:N...)
    A = sparse(j,k,1,np,np)+sparse(i,k,1,np,np)+sparse(i,j,1,np,np)
    f = findall(x->x>0,LinearAlgebra.triu(A+transpose(A)))
    [Chain{M,1}(SVector{N-1,Int}(f[n].I)) for n ∈ 1:length(f)]
end

edgesindices(t::ChainBundle) = edgesindices(value(t))
function edgesindices(t,i=getindex.(t,1),j=getindex.(t,2),k=getindex.(t,3))
    np,nt = length(points(t)),length(t)
    e = edges(t,i,j,k)
    A = sparse(getindex.(e,1),getindex.(e,2),1:length(e),np,np)
    V = ChainBundle(means(e,points(t))); nV = ndims(V); A += A'
    e,[Chain{V,1}(SVector{nV,Int}(A[j[n],k[n]],A[i[n],k[n]],A[i[n],j[n]])) for n ∈ 1:nt]
end

edgelengths(e) = value.(abs.(getindex.(diff.(getindex.(Ref(DirectSum.supermanifold(Manifold(e))),value.(e))),1)))

function neighbor!(out,m,t,i,::Val{a},::Val{b},::Val{c}) where {a,b,c}
    n = setdiff(intersect(findall(x->x>0,m[t[i][a],:]),findall(x->x>0,m[t[i][b],:])),i)
    !isempty(n) && (out[c] = n[1])
end

function neighbors(T)
    np,nt,t = length(points(T)),length(T),value(T)
    n2e = spzeros(np,nt) # node-to-element adjacency matrix, n2e[i,j]==1 -> i ∈ j
    for i ∈ 1:nt
        n2e[value(t[i]),i] .= (1,1,1)
    end
    V = SubManifold(ℝ^3)
    hood = Chain{V,1,Int,3}[]
    resize!(hood,nt)
    out = zeros(MVector{3,Int})
    for i ∈ 1:nt
        out .= 0
        neighbor!(out,n2e,t,i,Val(2),Val(3),Val(1))
        neighbor!(out,n2e,t,i,Val(3),Val(1),Val(2))
        neighbor!(out,n2e,t,i,Val(1),Val(2),Val(3))
        hood[i] = Chain{V,1}(out)
    end
    return hood
end

basetransform(t::ChainBundle) = basetransform(value(t))
function basetransform(t,i=getindex.(t,1),j=getindex.(t,2),k=getindex.(t,3))
    nt,p = length(t),points(t)
    M,A = Manifold(p)(2,3),p[i]
    Chain{M,1}.(SVector.(M.(p[j]-A),M.(p[k]-A)))
end

function basisnedelec(p)
    M = ℝ^3; V = M(2,3)
    Chain{M,1}(SVector(
        Chain{V,1}(SVector(-p[2],p[1])),
        Chain{V,1}(SVector(-p[2],p[1]-1)),
        Chain{V,1}(SVector(1-p[2],p[1]))))
end

function nedelecmean(t,t2e,signs,u)
    base = basetransform(t)
    B = revrot.(base,revrot)./getindex.(.∧(value.(base)),1)
    N = basisnedelec(ones(2)/3)
    x,y,z = getindex.(t2e,1),getindex.(t2e,2),getindex.(t2e,3)
    X,Y,Z = getindex.(signs,1),getindex.(signs,2),getindex.(signs,3)
    (u[x].*X).*(B.⋅N[1]) + (u[y].*Y).*(B.⋅N[2]) + (u[z].*Z).*(B.⋅N[3])
end
