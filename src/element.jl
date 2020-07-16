
#   This file is part of Adapode.jl. It is licensed under the AGPL license
#   Adapode Copyright (C) 2019 Michael Reed

export assemble, assembleglobal, assemblestiffness, assembleconvection, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assembledivergence
export assembleload, assemblemassload, assemblerobin, edges, edgesindices, neighbors
export solvepoisson, solveSD, solvetransport, solvedirichlet, adaptpoisson
export gradienthat, gradientCR, gradient, interp, nedelec, nedelecmean, jumps
export submesh, detsimplex, iterable, callable, value, edgelengths
export boundary, interior, trilength, trinormals
import Grassmann: norm, column, columns, points, pointset, edges
using Base.Threads

@inline iterpts(t,f) = iterable(points(t),f)
@inline iterable(p,f) = range(f,f,length=length(p))
@inline iterable(p,f::F) where F<:Function = f.(value(p))
@inline iterable(p,f::ChainBundle) = value(f)
@inline iterable(p,f::F) where F<:AbstractVector = f
@inline callable(c::F) where F<:Function = c
@inline callable(c) = x->c

revrot(hk::Chain{V,1},f=identity) where V = Chain{V,1}(-f(hk[2]),f(hk[1]))

function gradienthat(t,m=volumes(t))
    N = ndims(Manifold(t))
    if N == 2
        inv.(m)
    elseif N == 3
        h = curls(t)./2m
        V = Manifold(h); V2 = ↓(V)
        [Chain{V,1}(revrot.(V2.(value(h[k])))) for k ∈ 1:length(h)]
    else
        Grassmann.grad.(points(t)[value(t)])
    end
end

trilength(rc) = value.(abs.(value(rc)))
function trinormals(t)
    c = curls(t)
    ds = trilength.(c)
    V = Manifold(c); V2 = ↓(V)
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

laplacian(t,u,m=volumes(t),g=gradienthat(t,m)) = value.(abs.(gradient(t,u,m,g)))
gradient(t,u,m=volumes(t),g=gradienthat(t,m)) = [u[value(t[k])]⋅value(g[k]) for k ∈ 1:length(t)]

interp(t,ut,A::SparseMatrixCSC) = A*ut
interp(t,A::SparseMatrixCSC=incidence(t)) = Diagonal(inv.(degrees(t,A)))*A
pretni(t,A::SparseMatrixCSC=incidence(t)) = interp(t,sparse(A'))
pretni(t,ut,A=pretni(t)) = A*ut

degrees(t) = assembleload(t,ndims(t))
interp(t,f,m=degrees(t)) = interp(t,iterpts(t,f),iterable(t,m))
function interp(t,f::F,m::V) where {F<:AbstractVector,V<:AbstractVector}
    b = zeros(length(points(t)))
    for k ∈ value(t)
        tk = value(k)
        b[tk] .+= f[tk]./m[tk]
    end
    return b
end

for T ∈ (:SVector,:MVector)
    @eval function assemblelocal!(M,mat::SMatrix{N,N},m,tk::$T{N}) where N
        for i ∈ 1:N, j∈ 1:N
            M[tk[i],tk[j]] += mat[i,j]*m
        end
    end
end

assembleglobal(M,t,m=volumes(t),c=1,g=0) = assembleglobal(M,t,iterable(t,m),iterable(t,c),iterable(t,g))
function assembleglobal(M,t,m::T,c::C,g::F) where {T<:AbstractVector,C<:AbstractVector,F<:AbstractVector}
    np = length(points(t)); A = spzeros(np,np)
    for k ∈ 1:length(t)
        assemblelocal!(A,M(c[k],g[k],Val(ndims(Manifold(t)))),m[k],value(t[k]))
    end
    return A
end

assemblefunction(t,f,m=volumes(t)) = assemblefunction(t,iterpts(t,f),iterable(t,m))
function assemblefunction(t,f::F,m::V) where {F<:AbstractVector,V<:AbstractVector}
    b,l = zeros(length(points(t))),m/ndims(Manifold(t))
    for k ∈ 1:length(t)
        tk = value(t[k])
        b[tk] .+= f[tk]*l[k]
    end
    return b
end

assemblemassfunction(t,f,m=volumes(t),L=m) = assemblemassfunction(t,iterpts(t,f),iterable(t,m),iterable(t,L))
function assemblemassfunction(t,f::F,m::V,L::T) where {F<:AbstractVector,V<:AbstractVector,T<:AbstractVector}
    np,N = length(points(t)),ndims(Manifold(t))
    M,b,n,l = spzeros(np,np), zeros(np), Val(N), L/N
    for k ∈ 1:length(t)
        tk = value(t[k])
        assemblelocal!(M,mass(nothing,nothing,n),m[k],tk)
        b[tk] .+= f[tk]*l[k]
    end
    return M,b
end

assembleload(t,l=volumes(t)) = assemblefunction(t,1,l)
assemblemassload(t,m=volumes(t),l=m) = assemblemassfunction(t,1,m,l)

mass(a,b,::Val{N}) where N = (ones(SMatrix{N,N,Int})+I)/Int(factorial(N+1)/factorial(N-1))
assemblemass(t,m=volumes(t)) = assembleglobal(mass,t,iterpts(t,m))

stiffness(c,g,::Val{2}) = (cg = c*g^2; SMatrix{2,2,typeof(c)}(cg,-cg,-cg,cg))
function stiffness(c,g,::Val{N}) where N
    A = zeros(MMatrix{N,N,typeof(c)})
    for i ∈ 1:N, j ∈ 1:N
        A[i,j] = c*(g[i]⋅g[j])[1]
    end
    return SMatrix{N,N,typeof(c)}(A)
end
assemblestiffness(t,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
# iterable(means(t),c) # mapping of c.(means(t))

function sonicstiffness(c,g,::Val{N}) where N
    A = zeros(MMatrix{N,N,typeof(c)})
    for i ∈ 1:N, j ∈ 1:N
        A[i,j] = c*g[i][1]^2+g[j][2]^2
    end
    return SMatrix{N,N,typeof(c)}(A)
end
assemblesonic(t,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(sonicstiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
# iterable(means(t),c) # mapping of c.(means(t))

convection(b,g,::Val{3}) = ones(SVector{3,Int})*column((b/3).⋅value(g))'
convection(b,g,::Val{N}) where N = ones(SVector{N,Int})*column((b/N).⋅value(g))'
assembleconvection(t,b,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(convection,t,m,b,g)

SD(b,g,::Val{3}) = (x=column(b.⋅value(g));x*x')
SD(b,g,::Val) = (x=column(b.⋅value(g));x*x')
assembleSD(t,b,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(SD,t,m,b,g)

function assembledivergence(t,m,g)
    p = points(t); np,nt = length(p),length(t)
    D1,D2 = spzeros(nt,np), spzeros(nt,np)
    for k ∈ 1:length(t)
        tk,gm = value(t[k]),g[k]*m[k]
        for i ∈ 1:ndims(Manifold(t))
            D1[k,tk[i]] = gm[i][1]
            D2[k,tk[i]] = gm[i][2]
        end
    end
    return D1,D2
end

function assemble(t,c=1,a=1,f=0,m=volumes(t),g=gradienthat(t,m))
    M,b = assemblemassfunction(t,f,isone(a) ? m : a.*m,m)
    return assemblestiffness(t,c,m,g),M,b
end

function assemblerobin(e,κ=1e6,gD=0,gN=0)
    a = means(e)
    v = volumes(e)
    m = iterable(a,κ)
    l = m.*iterable(a,gD).+iterable(a,gN)
    return assemblemassload(e,m.*v,l.*v)
end

function solvepoisson(t,e,c,f,κ,gD=0,gN=0)
    m = volumes(t)
    b = assemblefunction(t,f,m)
    A = assemblestiffness(t,c,m)
    R,r = assemblerobin(e,κ,gD,gN)
    return (A+R)\(b+r)
end

function solveSD(t,e,c,f,δ,κ,gD=0,gN=0)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,c,m,g)
    b = means(t,f)
    C = assembleconvection(t,b,m,g)
    Sd = assembleSD(t,sqrt(δ)*b,m,g)
    R,r = assemblerobin(e,κ,gD,gN)
    return (A+R-C'+Sd)\r
end

function solvetransport(t,e,c,ϵ=0.1)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,ϵ,m,g)
    b = assembleload(t,m)
    C = assembleconvection(t,c,m,g)
    return solvedirichlet(A+C,b,e)
end

function adaptpoisson(g,p,e,t,c=1,a=0,f=1,κ=1e6,gD=0,gN=0)
    ϵ = 1.0
    while ϵ > 5e-5 && length(t) < 10000
        m = volumes(t)
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

solvedirichlet(M,b,e::ChainBundle) = solvedirichlet(M,b,pointset(e))
solvedirichlet(M,b,e::ChainBundle,u) = solvedirichlet(M,b,pointset(e),u)
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

interior(e) = interior(length(points(e)),pointset(e))
interior(fixed,neq) = sort!(setdiff(1:neq,fixed))
solvehomogenous(e,M,b) = solvedirichlet(M,b,e)
export solvehomogenous, solveboundary
const solveboundary = solvedirichlet # deprecate
const edgelengths = volumes # deprecate
const boundary = pointset # deprecate

facesindices(t,cols=columns(t)) = ndims(t) == 3 ? edgesindices(t,cols) : throw(error())

function edgesindices(t,cols=columns(t))
    np,nt = length(points(t)),length(t)
    e = edges(t,cols); i,j,k = cols
    A = sparse(getindex.(e,1),getindex.(e,2),1:length(e),np,np)
    V = ChainBundle(means(e,points(t))); A += A'
    e,[Chain{V,2}(A[j[n],k[n]],A[i[n],k[n]],A[i[n],j[n]]) for n ∈ 1:nt]
end

function neighbor(k::Int,ab...)::Int
    n = setdiff(intersect(ab...),k)
    isempty(n) ? 0 : n[1]
end

@generated function neighbors(A::SparseMatrixCSC,V,tk,k)
    N,F = ndims(Manifold(V)),(x->x>0)
    N1 = Grassmann.list(1,N)
    x = SVector{N}([Symbol(:x,i) for i ∈ N1])
    f = SVector{N}([:(findall($F,A[:,tk[$i]])) for i ∈ N1])
    b = SVector{N}([Expr(:call,:neighbor,:k,x[setdiff(N1,i)]...) for i ∈ N1])
    Expr(:block,Expr(:(=),Expr(:tuple,x...),Expr(:tuple,f...)),
        Expr(:call,:(Chain{V,1}),b...))
end

function neighbors(t,n2e=incidence(t))
    V,A = Manifold(Manifold(t)),sparse(n2e')
    nt = length(t)
    n = Chain{V,1,Int,ndims(V)}[]; resize!(n,nt)
    @threads for k ∈ 1:nt
        n[k] = neighbors(A,V,t[k],k)
    end
    return n
end

function centroidvectors(t,m=means(t))
    p,nt = points(t),length(t)
    V = Manifold(p)(2,3)
    c = Vector{SizedVector{3,Chain{V,1,Float64,2}}}(undef,nt)
    δ = Vector{SizedVector{3,Float64}}(undef,nt)
    for k ∈ 1:nt
        c[k] = V.(m[k].-p[value(t[k])])
        δ[k] = value.(abs.(c[k]))
    end
    return c,δ
end

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

function basisnedelec(p)
    M = SubManifold(ℝ^3); V = ↓(M)
    Chain{M,1}(
        Chain{V,1}(-p[2],p[1]),
        Chain{V,1}(-p[2],p[1]-1),
        Chain{V,1}(1-p[2],p[1]))
end

function nedelecmean(t,t2e,signs,u)
    base = Grassmann.vectors(t)
    B = revrot.(base,revrot)./column(.∧(value.(base)))
    N = basisnedelec(SVector(1,1)/3)
    x,y,z = columns(t2e); X,Y,Z = columns(signs,1,3)
    (u[x].*X).*(B.⋅N[1]) + (u[y].*Y).*(B.⋅N[2]) + (u[z].*Z).*(B.⋅N[3])
end

function jumps(t,c,a,f,u,m=volumes(t),g=gradienthat(t,m))
    N,np,nt = ndims(Manifold(t)),length(points(t)),length(t)
    η = zeros(nt)
    if N == 2
        fau = iterable(points(t),f).-a*u
        @threads for i ∈ 1:nt
            η[i] = m[i]*sqrt((fau[i]^2+fau[i+1]^2)*m[i]/2)
        end
    elseif N == 3
        ds,dn = trinormals(t) # ds.^1
        du,F,cols = gradient(t,u,m,g),iterable(t,f),columns(t)
        fl = [-c*column(value(dn[k]).⋅du[k]) for k ∈ 1:length(du)]
        intj = round.(adjacency(t,cols)/3)
        i,j,k = cols; x,y,z = getindex.(fl,1),getindex.(fl,2),getindex.(fl,3)
        jmps = sparse(j,k,x,np,np)+sparse(k,i,y,np,np)+sparse(i,j,z,np,np)
        jmps = abs.(intj.*abs.(jmps+jmps'))
        @threads for k = 1:nt
            tk,dsk = t[k],ds[k]
            η[k] = sqrt(((dsk[3]*jmps[tk[1],tk[2]])^2+(dsk[1]*jmps[tk[2],tk[3]])^2+(dsk[2]*jmps[tk[3],tk[1]])^2)/2)
        end
        η += [sqrt(norm(F[k].-a*u[value(t[k])])/3m[k]) for k ∈ 1:nt].*maximum.(ds)
    else
        throw(error("jumps on Manifold{$N} not defined"))
    end
    return η
end
