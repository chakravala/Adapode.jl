
#   This file is part of Adapode.jl. It is licensed under the AGPL license
#   Adapode Copyright (C) 2019 Michael Reed

export assemble, assembleglobal, assemblestiffness, assembleconvection, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assembledivergence
export assembleload, assemblemassload, assemblerobin, edges, edgesindices, neighbors
export solvepoisson, solveSD, solvedirichlet, boundary, interior, trilength, trinormals
export gradienthat, gradientCR, gradient, interp, nedelec, nedelecmean, jumps
export submesh, detsimplex, iterable, callable, value, edgelengths, adaptpoisson
import Grassmann: points, norm
using Base.Threads

@inline iterpts(t,f) = iterable(points(t),f)
@inline iterable(p,f) = range(f,f,length=length(p))
@inline iterable(p,f::F) where F<:Function = f.(value(p))
@inline iterable(p,f::ChainBundle) = value(f)
@inline iterable(p,f::F) where F<:AbstractVector = f
@inline callable(c::F) where F<:Function = c
@inline callable(c) = x->c

for T ∈ (:SVector,:MVector)
    @eval function assemblelocal!(M,mat::SMatrix{N,N},m,tk::$T{N}) where N
        for i ∈ 1:N, j∈ 1:N
            M[tk[i],tk[j]] += mat[i,j]*m[1]
        end
    end
end

assembleglobal(M,t,m=detsimplex(t),c=1,g=0) = assembleglobal(M,t,iterable(t,m),iterable(t,c),iterable(t,g))
function assembleglobal(M,t,m::T,c::C,g::F) where {T<:AbstractVector,C<:AbstractVector,F<:AbstractVector}
    np = length(points(t)); A = spzeros(np,np)
    for k ∈ 1:length(t)
        assemblelocal!(A,M(c[k],g[k],Val(ndims(Manifold(t)))),m[k],value(t[k]))
    end
    return A
end

assemblefunction(t,f,m=detsimplex(t)) = assemblefunction(t,iterpts(t,f),iterable(t,m))
function assemblefunction(t,f::F,m::V) where {F<:AbstractVector,V<:AbstractVector}
    b,l = zeros(length(points(t))),m/ndims(Manifold(t))
    for k ∈ 1:length(t)
        tk = value(t[k])
        b[tk] .+= f[tk]*l[k][1]
    end
    return b
end

assemblemassfunction(t,f,m=detsimplex(t),L=m) = assemblemassfunction(t,iterpts(t,f),iterable(t,m),iterable(t,L))
function assemblemassfunction(t,f::F,m::V,L::T) where {F<:AbstractVector,V<:AbstractVector,T<:AbstractVector}
    np,N = length(points(t)),ndims(Manifold(t))
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

revrot(hk::Chain{V,1},f=identity) where V = Chain{V,1}(-f(hk[2]),f(hk[1]))

function gradienthat(t,m=detsimplex(t))
    N = ndims(Manifold(t))
    if N == 2
        inv.(getindex.(value(m),1))
    elseif N == 3
        h = curls(t)./2getindex.(value(m),1)
        V = Manifold(h); V2 = V(2,3)
        [Chain{V,1}(revrot.(V2.(value(h[k])))) for k ∈ 1:length(h)]
    else
        throw(error("hat gradient on Manifold{$N} not defined"))
    end
end

trilength(rc) = value.(abs.(value(rc)))
function trinormals(t)
    c = curls(t)
    ds = trilength.(c)
    V = Manifold(c); V2 = V(2,3)
    dn = [Chain{V,1}(revrot.(V2.(value(c[k]))./-ds[k])) for k ∈ 1:length(c)]
    return ds,dn
end

gradientCR(t,m) = gradientCR(gradienthat(t,m))
gradientCR(g) = gradientCR.(g)
function gradientCR(g::Chain{V}) where V
    Chain{V,1}(g.⋅SVector(
        Chain{V,1}(-1,1,1),
        Chain{V,1}(1,-1,1),
        Chain{V,1}(1,1,-1)))
end

gradient(t,u,m=detsimplex(t),g=gradienthat(t,m)) = [u[value(t[k])]⋅value(g[k]) for k ∈ 1:length(t)]

function interp(t,ut)
    np,nt = length(points(t)),length(t)
    A = spzeros(np,nt)
    for i ∈ 1:ndims(Manifold(t))
        A += sparse(getindex.(value(t),i),1:nt,1,np,nt)
    end
    sparse(1:np,1:np,inv.(sum(A,dims=2))[:],np,np)*A*ut
end

function assembledivergence(t,m,g)
    p = points(t); np,nt = length(p),length(t)
    D1,D2 = spzeros(nt,np), spzeros(nt,np)
    for k ∈ 1:length(t)
        tk,gm = value(t[k]),g[k]*m[k][1]
        for i ∈ 1:ndims(Manifold(t))
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
assemblestiffness(t,c=1,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
# iterable(means(t),c) # mapping of c.(means(t))

function sonicstiffness(c,g,::Val{3})
    A = zeros(MMatrix{3,3,typeof(c)})
    for i ∈ 1:3, j ∈ 1:3
        A[i,j] += c*g[i][1]^2+g[j][2]^2
    end
    return SMatrix{3,3,typeof(c)}(A)
end
assemblesonic(t,c=1,m=detsimplex(t),g=gradienthat(t,m)) = assembleglobal(sonicstiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
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

function assemble(t,c=1,a=1,f=0,m=detsimplex(t),g=gradienthat(t,m))
    M,b = assemblemassfunction(t,f,isone(a) ? m : a.*m,m)
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

function adaptpoisson(g,p,e,t,c=1,a=0,f=1,κ=1e6,gD=0,gN=0)
    ϵ = 1.0
    while ϵ > 5e-5 && length(t) < 10000
        m = detsimplex(t)
        h = gradienthat(t,m)
        A,M,b = assemble(t,c,a,f,m,h)
        ξ = solvedirichlet(A+M,b,e)
        η = jumps(t,c,a,f,ξ,m,h)
        ϵ = sqrt(norm(η)^2/length(η))
        println(t,", ϵ=$ϵ, α=$(ϵ/maximum(η))")
        refinemesh!(g,p,e,t,select(η,ϵ),"regular")
    end
    return g,p,e,t
end

solvedirichlet(M,b,e::ChainBundle) = solvedirichlet(M,b,boundary(e))
solvedirichlet(M,b,e::ChainBundle,u) = solvedirichlet(M,b,boundary(e),u)
function solvedirichlet(A,b,fixed,boundary)
    neq = length(b)
    free,ξ = interior(fixed,neq),zeros(eltype(b),neq)
    ξ[fixed] = boundary
    ξ[free] = A[free,free]\(b[free]-A[free,fixed]*boundary)
    return ξ
end
function solvedirichlet(M,b,fixed)
    neq = length(b)
    free,ξ = interior(fixed,neq),zeros(eltype(b),neq)
    ξ[free] = M[free,free]\b[free]
    return ξ
end

boundary(e) = sort!(unique(vcat(value.(value(e))...)))
interior(e) = interior(length(points(e)),boundary(e))
interior(fixed,neq) = sort!(setdiff(1:neq,fixed))
solvehomogenous(e,M,b) = solvedirichlet(M,b,e)
const solveboundary = solvedirichlet
export solvehomogenous, solveboundary # deprecate

edges(t::ChainBundle) = edges(value(t))
function edges(t,i=getindex.(t,1),j=getindex.(t,2),k=getindex.(t,3))
    np,N = length(points(t)),ndims(Manifold(t)); M = points(t)(2:N...)
    A = sparse(j,k,1,np,np)+sparse(i,k,1,np,np)+sparse(i,j,1,np,np)
    f = findall(x->x>0,LinearAlgebra.triu(A+transpose(A)))
    [Chain{M,1}(SVector{N-1,Int}(f[n].I)) for n ∈ 1:length(f)]
end

edgesindices(t::ChainBundle) = edgesindices(value(t))
function edgesindices(t,i=getindex.(t,1),j=getindex.(t,2),k=getindex.(t,3))
    np,nt = length(points(t)),length(t)
    e = edges(t,i,j,k)
    A = sparse(getindex.(e,1),getindex.(e,2),1:length(e),np,np)
    V = ChainBundle(means(e,points(t))); A += A'
    e,[Chain{V,1}(A[j[n],k[n]],A[i[n],k[n]],A[i[n],j[n]]) for n ∈ 1:nt]
end

edgelengths(e) = value.(abs.(getindex.(diff.(getindex.(Ref(DirectSum.supermanifold(Manifold(e))),value.(e))),1)))

function neighbor(k,a,b)::Int
    n = setdiff(intersect(a,b),k)
    isempty(n) ? 0 : n[1]
end

function neighbors(T)
    np,nt,t = length(points(T)),length(T),value(T)
    n2e = spzeros(np,nt) # node-to-element adjacency matrix, n2e[i,j]==1 -> i ∈ j
    for i ∈ 1:nt
        n2e[value(t[i]),i] .= (1,1,1)
    end
    V,f = SubManifold(ℝ^3),(x->x>0)
    n = Chain{V,1,Int,3}[]; resize!(n,nt)
    @threads for k ∈ 1:nt
        tk = t[k]
        a,b,c = findall(f,n2e[tk[1],:]),findall(f,n2e[tk[2],:]),findall(f,n2e[tk[3],:])
        n[k] = Chain{V,1}(neighbor(k,b,c),neighbor(k,c,a),neighbor(k,a,b))
    end
    return n
end

basetransform(t::ChainBundle) = basetransform(value(t))
function basetransform(t,i=getindex.(t,1),j=getindex.(t,2),k=getindex.(t,3))
    nt,p = length(t),points(t)
    M,A = Manifold(p)(2,3),p[i]
    Chain{M,1}.(SVector.(M.(p[j]-A),M.(p[k]-A)))
end

function basisnedelec(p)
    M = ℝ^3; V = M(2,3)
    Chain{M,1}(
        Chain{V,1}(-p[2],p[1]),
        Chain{V,1}(-p[2],p[1]-1),
        Chain{V,1}(1-p[2],p[1]))
end

function nedelecmean(t,t2e,signs,u)
    base = basetransform(t)
    B = revrot.(base,revrot)./getindex.(.∧(value.(base)),1)
    N = basisnedelec(ones(2)/3)
    x,y,z = getindex.(t2e,1),getindex.(t2e,2),getindex.(t2e,3)
    X,Y,Z = getindex.(signs,1),getindex.(signs,2),getindex.(signs,3)
    (u[x].*X).*(B.⋅N[1]) + (u[y].*Y).*(B.⋅N[2]) + (u[z].*Z).*(B.⋅N[3])
end

function jumps(t,c,a,f,u,m=detsimplex(t),g=gradienthat(t,m))
    N,np,nt = ndims(Manifold(t)),length(points(t)),length(t)
    η = zeros(nt)
    if N == 2
        fau = iterable(points(t),f).-a*u
        @threads for i ∈ 1:nt
            η[i] = m[i][1]*sqrt((fau[i]^2+fau[i+1]^2)*m[i][1]/2)
        end
    elseif N == 3
        ds,dn = trinormals(t) # ds.^1
        du,F = gradient(t,u,m,g),iterable(t,f)
        fl = [-c*getindex.(value(dn[k]).⋅du[k],1) for k ∈ 1:length(du)]
        i,j,k = getindex.(value(t),1),getindex.(value(t),2),getindex.(value(t),3)
        intj = sparse(j,k,1,np,np)+sparse(k,i,1,np,np)+sparse(i,j,1,np,np)
        intj = round.((intj+transpose(intj))/3)
        jmps = sparse(j,k,getindex.(fl,1),np,np)+sparse(k,i,getindex.(fl,2),np,np)+sparse(i,j,getindex.(fl,3),np,np)
        jmps = abs.(intj.*abs.(jmps+jmps'))
        @threads for k = 1:nt
            tk,dsk = t[k],ds[k]
            η[k] = sqrt(((dsk[3]*jmps[tk[1],tk[2]])^2+(dsk[1]*jmps[tk[2],tk[3]])^2+(dsk[2]*jmps[tk[3],tk[1]])^2)/2)
        end
        η += [sqrt(norm(F[k].-a*u[value(t[k])])/3m[k][1]) for k ∈ 1:nt].*maximum.(ds)
    else
        throw(error("jumps on Manifold{$N} not defined"))
    end
    return η
end
