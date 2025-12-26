
#   This file is part of Adapode.jl
#   It is licensed under the AGPL license
#   Adapode Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

export assemble, assembleglobal, assemblestiffness, assembleconvection, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assembledivergence
export assemblemassincidence, asssemblemasstotalnodes, assembletotalnodes
export assembleelastic, assemblestokes, assemblemaxwell, assembleDIPG
export assembleload, assemblemassload, assemblerobin, edges, edgesindices, neighbors
export solvepoisson, solvetransport, solvetransportdiffusion, solvemaxwell, adaptpoisson
export solveelastic, solvenonlinearpoisson, solvebistable, solvebistable_newton
export solvedirichlet, solveheat, solvewave, solvestokes, solvenavierstokes
export gradienthat, gradientCR, gradient, interp, nedelec, nedelecmean, jumps
export submesh, detsimplex, iterable, callable, value, laplacian, biharmonic
export interior, trilength, trinormals, incidence, degrees, solvepoissonDIPG
export helmholtz, solvehelmholtz, solvenonlinearhelmholtz, polarlaplacian
export reshapedirichlet, reshapepolardirichlet, solvepolardirichlet, orrsommerfeld
export solveiteration, dirichlet!

using Base.Threads
import Grassmann: norm, column, columns
import Cartan: points, pointset, edges, iterpts, iterable, callable, revrot
import Cartan: gradienthat, laplacian, gradient, assemblelocal!
import Cartan: weights, degrees, assembleincidence, incidence
import Cartan: assembleload, interp, pretni, interior
import Cartan: facesindices, edgesindices, neighbor, neighbors

trilength(t::ElementBundle) = trilength.(fiber(curls(t)))
trilength(rc) = value.(abs.(value(rc)))
function trinormals(t)
    c = fiber(curls(t))
    ds = trilength.(c)
    V = Manifold(c); V2 = ↓(V)
    dn = [Chain{V,1}(revrot.(V2.(value(c[k]))./-ds[k])) for k ∈ 1:length(c)]
    return ds,dn
end

gradientCR(t,m) = gradientCR(gradienthat(t,m))
gradientCR(g) = TensorField(base(g),gradientCR.(fiber(g)))
gradientCR(g::TensorOperator) = gradientCR(value(g))
function gradientCR(g::Chain{V}) where V
    Chain{V,1}(g.⋅Values(
        Chain{V,1}(-1,1,1),
        Chain{V,1}(1,-1,1),
        Chain{V,1}(1,1,-1)))
end

assembleglobal(M,t::SimplexBundle,m=volumes(t),c=1,g=0) = assembleglobal(M,immersion(t),iterable(t,m),iterable(t,c),iterable(t,g))
function assembleglobal(M,t::ImmersedTopology{N},m::AbstractVector,c::AbstractVector,g::AbstractVector) where N
    np,v = totalnodes(t),Val(N)
    A = spzeros(np,np)
    for k ∈ 1:length(t)
        assemblelocal!(A,M(fiber(c)[k],fiber(g)[k],v),fiber(m)[k],t[k])
    end
    return A
end

assemblemassincidence(t::SimplexBundle,f,m=volumes(t),l=m) = assemblemassincidence(immersion(t),iterpts(t,f),iterable(t,m),iterable(t,l))
function assemblemassincidence(t::ImmersedTopology{N},f::F,m::V,l::T) where {N,F<:AbstractVector,V<:AbstractVector,T<:AbstractVector}
    np,n = totalnodes(t),Val(N)
    M,b = spzeros(np,np), zeros(np)
    for k ∈ 1:length(t)
        tk = t[k]
        assemblelocal!(M,mass(nothing,nothing,n),fiber(m)[k],tk)
        b[tk] .+= fiber(f)[tk]*fiber(l)[k]
    end
    return M,b
end

assemblemassload(t,f,m=volumes(t),l=m) = assemblemassincidence(t,iterpts(t,f)/sdims(t),m,l)
assemblemassload(tf::TensorField,m=volumes(base(tf)),l=m) = assemblemassload(base(tf),fiber(tf),m,l)

#mass(a,b,::Val{N}) where N = (ones(SMatrix{N,N,Int})+I)/Int(factorial(N+1)/factorial(N-1))
mass(a,b,::Val{N}) where N = (x=Submanifold(N)(∇);outer(x,x)+I)/Int(factorial(N+1)/factorial(N-1))
assemblemass(t,m=volumes(t)) = assembleglobal(mass,t,iterpts(t,m))

stiffness(c,g::Float64,::Val{2}) = (cg = c*g^2; Chain(Chain(cg,-cg),Chain(-cg,cg)))
stiffness(c,g,::Val{N}) where N = Chain{Submanifold(N),1}(map.(*,c,value(value(g)).⋅Ref(g)))
assemblestiffness(t,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
assemblestiffness(t::SimplexBundle,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(c isa Real ? immersion(t) : means(t),c),g)

# iterable(means(t),c) # mapping of c.(means(t))

#=function sonicstiffness(c,g,::Val{N}) where N
    A = zeros(MMatrix{N,N,typeof(c)})
    for i ∈ 1:N, j ∈ 1:N
        A[i,j] = c*g[i][1]^2+g[j][2]^2
    end
    return SMatrix{N,N,typeof(c)}(A)
end
assemblesonic(t,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(sonicstiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
# iterable(means(t),c) # mapping of c.(means(t))=#

#convection(b,g,::Val{N}) where N = ones(Values{N,Int})*column((b/N).⋅value(g))'
convection(b,g,::Val{N}) where N = outer(∇,Chain(Real.((b/N).⋅value(value(g)))))
assembleconvection(t,b,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(convection,t,m,b,g)

#SD(b,g,::Val) = (x=column(b.⋅value(g));x*x')
SD(b,g,::Val) = (x=Chain(column(b.⋅value(value(g))));outer(x,x))
assembleSD(t,b,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(SD,t,m,b,g)

function assembledivergence(t2e,m,g)
    nt,ne = length(m),totalnodes(t2e)
    D1,D2 = spzeros(nt,ne),spzeros(nt,ne)
    for k ∈ 1:nt
        edges,gm = immersion(t2e)[k],value(fiber(g)[k])*fiber(m)[k]
        D1[k,edges] .= getindex.(gm,1)
        D2[k,edges] .= getindex.(gm,2)
    end
    return D1,D2
end

function assemble(t,c=1,a=1,f=0,m=volumes(t),g=gradienthat(t,m))
    M,b = assemblemassload(t,f,typeof(a)<:Real && isone(a) ? m : fiber(a).*fiber(m),m)
    return assemblestiffness(t,c,m,g),M,b
end

assemblerobin(eκ::TensorField) = assemblerobin(base(eκ),fiber(eκ))
assemblerobin(e,κ=1e6) = assemblemassload(e,1,iterable(fiber(means(e)),fiber(κ)).*fiber(volumes(e)),0)
function assemblerobin(e,κ,gD,gN=0)
    a = fiber(means(e))
    v = fiber(volumes(e))
    m = fiber(iterable(a,κ))
    l = m.*iterable(a,fiber(gD)).+iterable(a,fiber(gN))
    return assemblemassload(e,1,m.*v,l.*v)
end

function solvepoisson(t,e,c,f,κ,gD=0,gN=0)
    m = volumes(t)
    b = assembleload(t,f,m)
    A = assemblestiffness(t,c,m)
    R,r = assemblerobin(e,κ,gD,gN)
    return TensorField(t,(A+R)\(b+r))
end

function poisson(t,e,c,a,f)
    A,M,b = assemble(t,c,a,f)
    ξ = solvedirichlet(A+M,b,immersion(e))
    return TensorField(t,ξ)
end

function solvetransportdiffusion(tf,eκ,c,δ,gD=0,gN=0)
    t,f,e,κ = base(tf),fiber(tf),base(eκ),fiber(eκ)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,c,m,g)
    b = means(immersion(t),f)
    C = assembleconvection(t,b,m,g)
    Sd = assembleSD(t,sqrt(δ)*b,m,g)
    R,r = assemblerobin(e,κ,gD,gN)
    return TensorField(t,(A+R-C'+Sd)\r)
end

function solvetransport(t,e,c,f=1,ϵ=0.1)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,ϵ,m,g)
    b = assembleload(t,f,m)
    C = assembleconvection(t,c,m,g)
    TensorField(t,solvedirichlet(A+C,b,e))
end

function adaptpoisson(g,pt,pe,c=1,a=0,f=1,κ=1e6,gD=0,gN=0)
    ϵ = 1.0
    while ϵ > 5e-5 && elements(pt) < 10000
        m = volumes(pt)
        h = gradienthat(pt,m)
        A,M,b = assemble(pt,c,a,f,m,h)
        ξ = solvedirichlet(A+M,b,immersion(pe))
        η = jumps(pt,c,a,f,ξ,m,h)
        ϵ = rms(η)
        println(Base.array_summary(stdout,immersion(pt),Cartan._axes(immersion(pt))),", ϵ=$ϵ, α=$(ϵ/maximum(η))")
        refinemesh!(g,pt,pe,select(η,ϵ),"regular")
    end
    return g,pt,pe
end

function dirichlet!(u)
    u[1] = u[end] = 0
    return u
end
function dirichlet!(u::AbstractMatrix)
    fiber(u)[1,:] .= fiber(u)[end,:] .= 0
    fiber(u)[:,1] .= fiber(u)[:,end] .= 0
    return u
end
function dirichlet!(u::AbstractArray{T,3} where T)
    fiber(u)[1,:,:] .= fiber(u)[end,:,:] .= 0
    fiber(u)[:,1,:] .= fiber(u)[:,end,:] .= 0
    fiber(u)[:,:,1] .= fiber(u)[:,:,end] .= 0
    return u
end
function dirichlet!(u::AbstractArray{T,4} where T)
    fiber(u)[1,:,:,:] .= fiber(u)[end,:,:,:] .= 0
    fiber(u)[:,1,:,:] .= fiber(u)[:,end,:,:] .= 0
    fiber(u)[:,:,1,:] .= fiber(u)[:,:,end,:] .= 0
    fiber(u)[:,:,:,1] .= fiber(u)[:,:,:,end] .= 0
    return u
end
function dirichlet!(u::AbstractArray{T,5} where T)
    fiber(u)[1,:,:,:,:] .= fiber(u)[end,:,:,:,:] .= 0
    fiber(u)[:,1,:,:,:] .= fiber(u)[:,end,:,:,:] .= 0
    fiber(u)[:,:,1,:,:] .= fiber(u)[:,:,end,:,:] .= 0
    fiber(u)[:,:,:,1,:] .= fiber(u)[:,:,:,end,:] .= 0
    fiber(u)[:,:,:,:,1] .= fiber(u)[:,:,:,:,end] .= 0
    return u
end

solvedirichlet(M,b,e::SimplexBundle) = solvedirichlet(M,b,vertices(e))
solvedirichlet(M,b,e::SimplexTopology) = solvedirichlet(M,b,vertices(e))
solvedirichlet(M,b,e::SimplexBundle,u) = solvedirichlet(M,b,vertices(e),u)
solvedirichlet(M,b,e::SimplexTopology,u) = solvedirichlet(M,b,vertices(e),u)
function solvedirichlet(A,b,fixed,boundary)
    neq = length(b) # number of equations
    free,ξ = interior(fixed,neq),zeros(eltype(b),neq)
    ξ[fixed] = boundary # set boundary condition
    ξ[free] = A[free,free]\(b[free]-A[free,fixed]*boundary)
    return ξ
end
function solvedirichlet(M,b,fixed)
    neq = length(b)
    free,ξ = interior(fixed,neq),zeros(eltype(b),neq)
    ξ[free] = M[free,free]\b[free]
    return ξ
end

to_TensorField(f::TensorField,u) = TensorField(f,u)
to_TensorField(f,u) = u

function reshapedirichlet(f::AbstractVector,u)
    N = length(f)
    uu = zeros(N)
    uu[2:N-1] .= u
    to_TensorField(f,uu)
end
function reshapedirichlet(f::AbstractMatrix,u)
    N,M = size(f)
    uu = zeros(N,M)
    uu[2:N-1,2:M-1] .= reshape(u,N-2,M-2)
    to_TensorField(f,uu)
end
function reshapedirichlet(f::AbstractArray{T,3} where T,u)
    N,M,O = size(f)
    uu = zeros(N,M,O)
    uu[2:N-1,2:M-1,2:O-1] .= reshape(u,N-2,M-2,O-2)
    to_TensorField(f,uu)
end
function reshapedirichlet(f::AbstractArray{T,4} where T,u)
    N,M,O,P = size(f)
    uu = zeros(N,M,O,P)
    uu[2:N-1,2:M-1,2:O-1,2:P-1] .= reshape(u,N-2,M-2,O-2,P-2)
    to_TensorField(f,uu)
end
function reshapedirichlet(f::AbstractArray{T,5} where T,u)
    N,M,O,P,Q = size(f)
    uu = zeros(N,M,O,P,Q)
    uu[2:N-1,2:M-1,2:O-1,2:P-1,2:Q-1] .= reshape(u,N-2,M-2,O-2,P-2,Q-2)
    to_TensorField(f,uu)
end

function reshapepolardirichlet(f::AbstractMatrix,u)
    N,M = size(f)
    uu = [zeros(1,M); reshape(u,N-1,M-1)[:,vcat(M-1,1:M-1)]]
    to_TensorField(f,uu)
end

function solvedirichlet(L,f::AbstractVector)
    N = length(f)
    reshapedirichlet(f,L\vec(fiber(f)[2:N-1]))
end
function solvedirichlet(L,f::AbstractMatrix)
    N,M = size(f)
    reshapedirichlet(f,L\vec(fiber(f)[2:N-1,2:M-1]))
end
function solvedirichlet(L,f::AbstractArray{T,3} where T)
    N,M,O = size(f)
    reshapedirichlet(f,L\vec(fiber(f)[2:N-1,2:M-1,2:O-1]))
end
function solvedirichlet(L,f::AbstractArray{T,4} where T)
    N,M,O,P = size(f)
    reshapedirichlet(f,L\vec(fiber(f)[2:N-1,2:M-1,2:O-1,2:P-1]))
end
function solvedirichlet(L,f::AbstractArray{T,5} where T)
    N,M,O,P,Q = size(f)
    reshapedirichlet(f,L\vec(fiber(f)[2:N-1,2:M-1,2:O-1,2:P-1,2:Q-1]))
end

#=function solvepolardirichlet(L,f::AbstractMatrix)
    N,M = size(f)
    u = reshape(L\vec(fiber(f)[2:N,2:M]),N-1,M-1)
    uu = [zeros(1,M); u[:,vcat(M-1,1:M-1)]]
    to_TensorField(f,uu/norm(u,Inf))
end=#
function solvepolardirichlet(L,f::AbstractMatrix)
    N,M = size(f)
    u = reshapepolardirichlet(f,L\vec(fiber(f)[2:N,2:M]))
    u/norm(fiber(u),Inf)
end


function solveiteration(L,f,u,solver=\)
    change = 1
    while change > 5eps()
        unew = solver(L,f.(u))
        change = norm(fiber(unew-u),Inf)
        u = unew
    end; return u
end

function helmholtz(f::AbstractVector,k=0)
    N = length(f)
    (ChebyshevMatrix(f)^2)[2:N-1,2:N-1] + k^2*I
end
function helmholtz(f::AbstractMatrix,k=0)
    N,M = size(f)
    x,y = split(points(f))
    D2X = (ChebyshevMatrix(x)^2)[2:N-1,2:N-1]
    D2Y = (ChebyshevMatrix(y)^2)[2:M-1,2:M-1]
    kron(I(M-2),D2X) + kron(D2Y,I(N-2)) + k^2*I
end

solvehelmholtz(f,k=0) = solvedirichlet(helmholtz(f,k),f)

function solvenonlinearhelmholtz(f,k,N::Int)
    nonlinearhelmholtz(f,k,TensorField(Chebyshev(N),zeros(N)))
end
function solvenonlinearhelmholtz(f,k,N::Int,M::Int)
    x,y = Chebyshev(N),Chebyshev(M)
    u = TensorField(ProductSpace{2}(x,y),zeros(N,M))
    nonlinearhelmholtz(f,k,u)
end
function solvenonlinearhelmholtz(f,k,u::TensorField)
    solveiteration(helmholtz(u,k),f,u,solvedirichlet)
end

function polarlaplacian(N=13,M=21)
    r = Chebyshev(2N)
    D = ChebyshevMatrix(r); D2 = D^2
    D1,D2 = D2[2:N,2:N],D2[2:N,2N-1:-1:N+1]
    E1,E2 = D[2:N,2:N],D[2:N,2N-1:-1:N+1]
    D2t,R = derivetoeplitz2(M-1),Diagonal(inv.(r[2:N]))
    M2 = Int((M-1)/2); Z = zeros(M2,M2)
    kron(I(M-1),D1+R*E1)+kron([Z I;I Z],D2+R*E2)+kron(D2t,R^2)
end

function biharmonic(v::AbstractVector,D=ChebyshevMatrix(points(v)))
    x = points(v)
    N = length(x)
    S = Diagonal(vcat(0,inv.(1.0.-x[2:N-1].^2),0))
    ((Diagonal(1.0.-x.^2)*D^4 - 8Diagonal(x)*D^3 - 12D^2)*S)[2:N-1,2:N-1]
end

function biharmonic(v::AbstractMatrix)
    N,M = size(v)
    x,y = split(points(v))
    DX,DY = ChebyshevMatrix(points(x)),ChebyshevMatrix(points(y)) # reverse x?
    D2X,D2Y = (DX^2)[2:N-1,2:N-1],(DY^2)[2:M-1,2:M-1]
    D4X,D4Y = biharmonic(x,DX),biharmonic(y,DY)
    kron(I(M-2),D4X) + kron(D4Y,I(N-2)) + 2kron(D2Y,I(N-2))*kron(I(M-2),D2X)
end

function orrsommerfeld(v,R=5772)
    N = length(v)
    x = points(v)
    D = ChebyshevMatrix(x)
    D2 = (D^2)[2:N-1,2:N-1]
    D4 = biharmonic(v,D)
    A = (D4-2D2+I(N-2))/R - 2im*I(N-2) - im*Diagonal(1.0.-x[2:N-1].^2)*(D2-I(N-2))
    B = D2 - I(N-2)
    return A,B
end

restwavemultiplier(k,t) = cos(t*Real(abs(k)))
wavemultiplier(k,t) = (ak = Real(abs(k)); sin(t*ak)/ak)
heatmultiplier(k,t) = exp(-t*Real(abs2(k)))
rieszmultiplier(k,t,s=2) = exp(-t*Real(abs2(k))^(s/2))
biharmonicmultiplier(k,t) = exp(-t*Real(abs2(k))^2)
schrodingermultiplier(k,t) = exp((-im/2)*t*Real(abs2(k)))
#heatkernel(u0,t,k=r2rspace(points(u0))) = idct(heatmultiplier.(k,t))

function rieszdirichlet end
function wavedirichlet end

heatperiodic(u0,t,k=rfftspace(points(u0))) = Cartan.irfft(Cartan.rfft(u0).*heatmultiplier.(k,t))
rieszperiodic(u0,t,s=2,k=rfftspace(points(u0))) = Cartan.irfft(Cartan.rfft(u0).*rieszmultiplier.(k,t,s))
biharmonicperiodic(u0,t,k=rfftspace(points(u0))) = Cartan.irfft(Cartan.rfft(u0).*biharmonicmultiplier.(k,t))
restwaveperiodic(u0,t,k=rfftspace(points(u0))) = Cartan.irfft(Cartan.rfft(u0).*restwavemultiplier.(k,t))
function waveperiodic(u0,u1,t,k=rfftspace(points(u0)))
    wm = wavemultiplier.(k,t); wm[1] = 0
    Cartan.irfft(Cartan.rfft(u0).*restwavemultiplier.(k,t)+Cartan.rfft(u1).*wm)
end
function fullwaveperiodic(u0,u1,t,k=rfftspace(points(u0)))
    wm = wavemultiplier.(k,t); wm[1] = t
    Cartan.irfft(Cartan.rfft(u0).*restwavemultiplier.(k,t)+Cartan.rfft(u1).*wm)
end
schrodingerperiodic(u0,t,k=fftspace(points(u0))) = Cartan.ifft(Cartan.fft(u0).*schrodingermultiplier.(k,t))
schrodingerperiodic(u0,t::AbstractVector,k=fftspace(points(u0))) = schrodingerperiodic(u0,TensorField(t),k)

function schrodingerperiodic(u0,t::TensorField,k=fftspace(points(u0)))
    data = zeros(size(u0)...,length(t))
    out = TensorField(base(u0)⊕base(t),data)
    assign!(out,1,u0)
    for i in 2:length(t)
        assign!(out,i,schrodingerperiodic(u0,fiber(t)[i],k))
    end
    return out
end

export wavedirichlet, wavemultiplier, schrodingerperiodic
export rieszdirichlet, rieszneumann, rieszperiodic, rieszmultiplier
export heatmultiplier, restwavemultiplier, biharmonicmultiplier

for fun ∈ (:heat,:restwave,:biharmonic)
    nfun,pfun,dfun = Symbol(fun,:neumann),Symbol(fun,:periodic),Symbol(fun,:dirichlet)
    @eval begin
        export $nfun, $pfun, $dfun
        function $dfun end
        $nfun(u0,t::AbstractVector,k=r2rspace(points(u0))) = $nfun(u0,TensorField(t),k)
        function $nfun(u0,t::TensorField,k=r2rspace(points(u0)))
            data = zeros(size(u0)...,length(t))
            out = TensorField(base(u0)⊕base(t),data)
            assign!(out,1,u0)
            for i in 2:length(t)
                assign!(out,i,$nfun(u0,fiber(t)[i],k))
            end
            return out
        end
        $pfun(u0,t::AbstractVector,k=rfftspace(points(u0))) = $pfun(u0,TensorField(t),k)
        function $pfun(u0,t::TensorField,k=rfftspace(points(u0)))
            data = zeros(size(u0)...,length(t))
            out = TensorField(base(u0)⊕base(t),data)
            assign!(out,1,u0)
            for i in 2:length(t)
                assign!(out,i,$pfun(u0,fiber(t)[i],k))
            end
            return out
        end
    end
end

for fun ∈ (:wave,:fullwave)
    nfun,pfun = Symbol(fun,:neumann),Symbol(fun,:periodic)
    @eval begin
        export $nfun, $pfun
        $nfun(u0,u1,t::AbstractVector,k=r2rspace(points(u0))) = $nfun(u0,u1,TensorField(t),k)
        function $nfun(u0,u1,t::TensorField,k=r2rspace(points(u0)))
            data = zeros(size(u0)...,length(t))
            out = TensorField(base(u0)⊕base(t),data)
            assign!(out,1,u0)
            for i in 2:length(t)
                assign!(out,i,$nfun(u0,u1,fiber(t)[i],k))
            end
            return out
        end
        $pfun(u0,u1,t::AbstractVector,k=rfftspace(points(u0))) = $pfun(u0,u1,TensorField(t),k)
        function $pfun(u0,u1,t::TensorField,k=rfftspace(points(u0)))
            data = zeros(size(u0)...,length(t))
            out = TensorField(base(u0)⊕base(t),data)
            assign!(out,1,u0)
            for i in 2:length(t)
                assign!(out,i,$pfun(u0,u1,fiber(t)[i],k))
            end
            return out
        end
    end
end

rieszneumann(u0,t::AbstractVector,s=2,k=r2rspace(points(u0))) = rieszneumann(u0,TensorField(t),s,k)
function rieszneumann(u0,t::TensorField,s=2,k=r2rspace(points(u0)))
    data = zeros(size(u0)...,length(t))
    out = TensorField(base(u0)⊕base(t),data)
    assign!(out,1,u0)
    for i in 2:length(t)
        assign!(out,i,rieszneumann(u0,fiber(t)[i],s,k))
    end
    return out
end
rieszperiodic(u0,t::AbstractVector,s=2,k=rfftspace(points(u0))) = rieszperiodic(u0,TensorField(t),s,k)
function rieszperiodic(u0,t::TensorField,s=2,k=rfftspace(points(u0)))
    data = zeros(size(u0)...,length(t))
    out = TensorField(base(u0)⊕base(t),data)
    assign!(out,1,u0)
    for i in 2:length(t)
        assign!(out,i,rieszperiodic(u0,fiber(t)[i],s,k))
    end
    return out
end

function solveheat(ic,f,κ,T)
    m,h = length(T),step(T)
    out = zeros(length(ic),m)
    out[:,1] = fiber(ic) # initial condition
    A = assemblestiffness(base(ic)) # assemble(p(t),1,f)
    M,b = assemblemassload(f).+assemblerobin(κ)
    LHS = M+h*A # time step
    for l ∈ 1:m-1
        out[:,l+1] = LHS\(M*out[:,l]+h*b); #l%10==0 && println(l*h)
    end
    TensorField(base(ic)⊕T,out)
end

function solvewave(pt,bc) # Crank-Nicolson
    Δt = step(base(bc).g.v[1])
    np = length(points(pt))
    sol = zeros(2np)
    out = zeros(np,length(base(bc).g))
    fixed = vertices(base(bc).s)
    A,M,b = assemble(pt,1,1,0)
    LHS = [M (Δt/-2)*M; (Δt/2)*A M] # Crank-Nicolson
    rhs = [M (Δt/2)*M; (Δt/-2)*A M]
    zkb = [zeros(np); Δt*b]
    for l ∈ 1:length(base(bc).g) # time loop
        sol .= LHS\(rhs*sol + zkb)
        sol[fixed] .= fiber(bc)[:,l] # set BC the ugly way
        out[:,l] .= sol[1:np]
    end
    return TensorField(pt⊕base(bc).g.v[1],out)
end

function assemblestokes(pt,ν=0.1,t2e=edgesindices(pt))
    nt,ne = length(immersion(pt)),length(t2e)
    m = volumes(pt)
    gCR = gradientCR(gradienthat(pt,m))
    A11 = ν*assemblestiffness(t2e,1,m,gCR)
    B1,B2 = assembledivergence(t2e,m,-gCR)
    [A11 spzeros(ne,ne) B1' spzeros(ne,1);
     spzeros(ne,ne) A11 B2' spzeros(ne,1);
     B1 B2 spzeros(nt,nt) fiber(m);
     spzeros(1,ne) spzeros(1,ne) fiber(m)' 0]
end

function solvestokes(pt,bc,ν=0.1,t2e=edgesindices(pt,base(bc)))
    nt,ne = length(immersion(pt)),length(t2e)
    LHS = assemblestokes(pt,ν,t2e)
    fixed = [subelements(bc); subelements(bc).+ne] # fixed totalnodes
    gvals = [getindex.(fiber(bc),1); getindex.(fiber(bc),2)] # nodal values of g
    ξ = solvedirichlet(LHS,zeros(2ne+nt+1),fixed,gvals)
    UV = Chain{Manifold(fibertype(bc))}.(ξ[1:ne],ξ[1+ne:2ne])
    P = ξ[2ne+1:2ne+nt]
    TensorField(pt,interp(fullimmersion(bc),UV)),TensorField(FaceBundle(pt),P)
end

function solvenavierstokes(pt,pe,inbc,outbc,ν=0.001,T=range(0,1,101),skip=1)
    p = fullcoordinates(pt)
    dt,nt = step(T),length(T)
    V = Cartan.varmanifold(3)(2,3)
    np = length(p)
    ins = vertices(inbc)
    inp = fiber(inbc) # inflow profile
    out = vertices(outbc) # totalnodes on outflow
    bnd = setdiff(vertices(pe),out) # remove outflow totalnodes from boundary totalnodes
    R = spzeros(np,np) # diagonal penalty matrix
    for i ∈ 1:length(out)
        j = out[i]
        R[j,j] = fiber(outbc)[i] # big weights
    end
    m = volumes(pt)
    g = gradienthat(pt,m)
    A = assemblestiffness(pt,1,m,g)
    M = assembleload(pt,1,m)
    Bx = assembleconvection(pt,Global{1}(Chain{V}(1,0)),m,g)
    By = assembleconvection(pt,Global{1}(Chain{V}(0,1)),m,g)
    B = Chain{V}.(Bx,By)
    νA,AR = ν*A,A+R
    dtiM = dt*inv.(M)
    UVs = Matrix{Chain{V,1,Float64,mdims(V)}}(undef,np,nt)
    UVs[:,1] = Chain{V,1,Float64}.(zeros(np),zeros(np)) # init velocity
    UVold = UVs[:,1]
    for l ∈ 1:nt-1
        C = assembleconvection(pt,means(pt,UVold),m,g)
        UVnew = UVold - ((νA+C)*UVold).*dtiM # compute tentative velocity
        UVnew[bnd] .*= 0 # no-slip boundary totalnodes
        UVnew[ins] += inp # inflow profile totalnodes
        P = (AR\(Bx*getindex.(UVnew,1)+By*getindex.(UVnew,2)))/-dt # solve PPE
        UVnew -= (B*P).*dtiM # update velocity
        if k==skip
            UVs[:,l+1] -= (B*P).*dtiM # update velocity
            k = 1
        else
            k += 1
        end
    end
    #=UVold = UVs[:,1]
    k = 1
    for l ∈ 1:nt-1
        C = assembleconvection(pt,means(pt,UVold),m,g)
        UVnew = UVold - ((νA+C)*UVold).*dtiM # compute tentative velocity
        UVnew[bnd] .*= 0 # no-slip boundary totalnodes
        UVnew[ins] += inp # inflow profile totalnodes
        P = (AR\(Bx*getindex.(UVnew,1)+By*getindex.(UVnew,2)))/-dt # solve PPE
        UVnew -= (B*P).*dtiM # update velocity
        UVold = copy(UVnew)
        if k==skip
            UVs[:,l+1] = UVnew # update velocity
            k = 1
        else
            k += 1
        end
    end=#
    return TensorField(pt⊕T,UVs)
end

function assembleelastic(f,μ,λ)
    t = immersion(f)
    ndof = 2length(f) # total number of DoFs
    K = spzeros(ndof,ndof) # stiffness
    M = spzeros(ndof,ndof) # mass
    F = zeros(ndof) # load
    m = volumes(base(f))
    g = gradienthat(base(f),m)
    for i ∈ 1:length(t) # assembly loop over subelements
        ti,mi = t[i],fiber(m)[i]
        fti = fiber(f)[ti]
        t2 = 2ti
        t1 = t2.-1
        MK = mass(nothing,nothing,Val(3))*mi
        assemblelocal!(K,elasticstrain(λ,μ,fiber(g)[i],Val(3)),mi,vcat(t1,t2))
        assemblelocal!(M,MK,t1)
        assemblelocal!(M,MK,t2)
        F[t1] .= value(MK⋅Chain(getindex.(fti,1)))
        F[t2] .= value(MK⋅Chain(getindex.(fti,2)))
    end
    return K,M,F
end

function solveelastic(f,e,E=1,ν=0.3)
    K,M,F = assembleelastic(f,Eν2Lame(E,ν)...)
    fixed = [2vertices(e).-1; 2vertices(e)] # boundary DoFs
    d = solvedirichlet(K,F,fixed,zeros(length(fixed)))
    TensorField(base(f),Chain{affinemanifold(2)}.(d[1:2:end],d[2:2:end]))
end

function assemblemaxwell(p,e,t,κ,μ,fhat,t2e=edgesindices(p(t)),signs=facetsigns(t))
    nt,ne = length(κ),length(e)
    A,b = spzeros(Complex{Float64},ne,ne),zeros(ne)
    m = volumes(p(t))
    g = gradienthat(p(t),m)
    l = volumes(p(e))
    for i ∈ 1:nt
        ed = immersion(t2e)[i]
        li = fiber(l)[ed].*signs[i]
        κi,mi,gi = fiber(κ)[i],fiber(m)[i],fiber(g)[i]
        assemblelocal!(A,maxwell(inv(mi*mi*μ),li,κi*κi,gi,Val(3)),mi,ed)
        b[ed] .+= Real.(fhat.⋅value(curl(value(gi)))).*(li*(mi/3))
    end
    return A,b
end

function solvemaxwell(κ,bc,μ=1,fhat=Chain{Cartan.varmanifold(3)(2,3)}(0,0))
    p,e,t = fullcoordinates(κ),fullimmersion(bc),fullimmersion(κ)
    pt = p(t)
    t2e = edgesindices(pt)#,p(e))
    signs = facetsigns(t)
    A,b = assemblemaxwell(p,e,t,κ,μ,fhat,t2e,signs)
    ξ = solvedirichlet(A,complex.(b),subelements(bc),fiber(bc))
    RI = interpnedelec(pt,ξ,t2e,signs)
    TensorField(pt,map.(real,fiber(RI))),TensorField(pt,map.(imag,fiber(RI)))
end

function assemblejacobianresidue(f,pe,u,Afcn,m,g,tiny=1e-8)
    pt,t = SimplexBundle(base(f)),immersion(f)
    # evaluate u, a, a', and f
    uc = means(t,fiber(u))
    a = Afcn.(uc)
    da = (Afcn.(uc.+tiny)-a)/tiny # da(u)/du
    gu = fiber(Cartan.gradient_2(pt,u,m,g)) # grad u
    # Assemble Jacobian and residual
    np = nodes(t)
    J,r = spzeros(np,np),zeros(np)
    for i ∈ 1:length(t)
        nodes = t[i]
        gi,mi,ai = fiber(g)[i],fiber(m)[i],fiber(a)[i]
        gug = Ref(value(fiber(gu)[i])).⋅value.(value(value(gi)))
        dagug = Chain{Cartan.varmanifold(3)}((da[i]/3)*gug)
        JK = Adapode.stiffness(ai,gi,Val(3))+Chain(dagug,dagug,dagug)
        Adapode.assemblelocal!(J,JK,mi,nodes)
        r[nodes] .+= ((fiber(f)[i]/3)*Values(1,1,1)-ai*gug)*mi
    end
    # enforce zero Dirichlet BC
    for n ∈ vertices(pe) # boundary nodes
        J[n,:] .= 0 # zero out row n of the Jacobian, J
        J[n,n] = 1 # set diagonal entry J[n,n] to 1
        r[n] = 0 # set residual entry r[n] to 0
    end
    return J,r
end

function solvenonlinearpoisson(f,pe,Afcn)
    pt = SimplexBundle(base(f))
    m = volumes(pt)
    g = gradienthat(pt,m)
    ξ = zeros(nodes(immersion(f)))
    for k ∈ list(1,5) # non-linear loop
        J,r = assemblejacobianresidue(f,pe,ξ,Afcn,m,g)
        d = J\r
        ξ += d # update solution by solving correction
        println("|d|=$(norm(d)), |r|=$(norm(r))")
    end
    TensorField(pt,ξ)
end

function solvebistable(ic,ϵ=0.01,T=StepRangeLen(0,0.1,101),f=u->u-u^3)
    dt = step(T)
    ξ = zeros(length(ic),length(T))
    ξ[:,1] = fiber(ic) # IC
    ξ_new = ξ[:,1]
    A,M = assemble(base(ic))
    for l ∈ 1:length(T)-1 # time loop
        for k ∈ list(1,3) # non-linear loop
            ξ_tmp = ξ_new
            ξ_new = (M+(dt*ϵ)*A)\(M*ξ[:,l]+dt*(M*f.(ξ_tmp)))
            #fixpterror = norm(ξ_tmp-ξ_new)
        end
        ξ[:,l+1] = ξ_new
    end
    TensorField(base(ic)⊕T,ξ)
end

function solvebistable_newton(ic,ϵ=0.01,T=StepRangeLen(0,0.1,101),f=u->u-u^3,df=u->1-3u^2)
    dt = step(T)
    np = length(ic)
    t = immersion(ic)
    ξ = zeros(length(ic),length(T))
    ξ[:,1] = fiber(ic) # IC
    ξ_new = ξ[:,1]
    m = volumes(base(ic))
    g = gradienthat(base(ic),m)
    A,M = assemble(base(ic),1,1,0,m,g)
    for l ∈ 1:length(T)-1 # time loop
        for k ∈ Cartan.list(1,3) # non-linear loop
            ξ_tmp = ξ_new
            ξ_tmp_mid = means(immersion(ic),ξ_tmp)
            Mdf,b = assemblemassload(base(ic),f.(ξ_tmp),df.(ξ_tmp_mid).*fiber(m),m)
            MA = M+(dt*ϵ)*A
            J = MA - dt*Mdf # Jacobian
            ρ = MA*ξ_new - M*ξ[:,l] - dt*b # residual
            ξ_new = ξ_tmp - J\ρ # Newton update
            #error = norm(ξ_tmp-ξ_new)
        end
        ξ[:,l+1] = ξ_new
    end
    TensorField(base(ic)⊕T,ξ)
end

function gradienthat(ip::Simplex) # ip is already inverse of the point Simplex
    TensorOperator(Chain{Manifold(ip)}(Cartan.affmanifold(2).(value(transpose(ip)))))
end

function assembleDIPG(pt,nbrs=neighbors(immersion(pt)))
    nt,np = elements(pt),totalnodes(pt)
    S = spzeros(np,np) # flux matrix
    P = spzeros(np,np) # penalty matrix
    edge2node = Values(Values(2,3),Values(1,3),Values(1,2))
    cmat = TensorOperator(Chain{Manifold(pt)}(Chain(2,0),Chain(1,1),Chain(0,2)))/2
    wvec = Values(1,4,1)/6 # Simpson's formula
    ah = Cartan.affinehull(pt)
    m = volumes(pt)
    ip = inv.(ah) # precompute
    ds,dn = trinormals(pt)
    val123 = Values(1,2,3)
    for i ∈ OneTo(nt)
        xyp = ah[i]
        ipp = ip[i]
        gp = gradienthat(ipp) # gradienthat on "plus" element
        ni = nbrs[i] # neighbors
        dsi,dni = ds[i],dn[i]
        for j ∈ val123 # loop over edges
            n = ni[j] # element neighbor
            n > i && continue # only assemble once on each edge
            iszero(n) && (n = i) # boundary?
            wxlen = wvec*dsi[j] # quadrature weight times edge length
            dnij = dni[j]
            exy = TensorOperator(Chain(xyp[edge2node[j]]))⋅cmat # xy coordinates of quadrature points on edge
            SE = zero(Chain{Submanifold(6),1,Chain{Submanifold(6),1,Float64}})
            PE = zero(Chain{Submanifold(6),1,Chain{Submanifold(6),1,Float64}})
            ipm = ip[n]
            gm = gradienthat(ipm) # gradienthat on "minus" element
            avgdn = Chain(vcat(value(dnij⋅gp),value(dnij⋅gm))/2) # average
            for q ∈ list(1,length(wvec)) # quadrature loop on edge
                jump = Chain(vcat(value(ipp⋅exy[q]),-value(ipm⋅exy[q]))) # jump
                PE += outer(jump*wvec[q],jump) # penalty divided by local length scale h_e
                SE += outer(jump*wxlen[q],avgdn)
            end
            if n==i # boundary
                dofs = val123.+3(i-1)
                assemblelocal!(P,PE,dofs)
                assemblelocal!(S,2SE,dofs) # no average on boundary
            else
                dofs = vcat(val123.+3(i-1),val123.+3(n-1))
                assemblelocal!(P,PE,dofs)
                assemblelocal!(S,SE,dofs)
            end
        end
    end
    return P,S
end

function solvepoissonDIPG(f,c,β,α=-1) # α = SIPG, β = penalty
    pt = base(f)
    m = volumes(pt)
    b = assembleload(pt,f,m)
    A = assemblestiffness(pt,c,m)
    P,S = assembleDIPG(pt)
    TensorField(pt,(A-S+α*S'+β*P)\b)
end

Eν2Lame(E,ν) = E/(2(1+ν)),E*ν/((1+ν)*(1-2ν))

elastic(μ,λ,::Val{3}) = DiagonalOperator(μ*Chain(2,2,1))+λ*TensorOperator(Chain(Chain(1,1,0),Chain(1,1,0),Chain(0,0,0)))
function strain(g,::Val{3})
    TensorOperator(Chain(
        Chain(g[1][1],0,g[1][2]),
        Chain(0,g[1][2],g[1][1]),
        Chain(g[2][1],0,g[2][2]),
        Chain(0,g[2][2],g[2][1]),
        Chain(g[3][1],0,g[3][2]),
        Chain(0,g[3][2],g[3][1])))
end
function elasticstrain(μ,λ,g,v::Val{3})
    D,BK = elastic(μ,λ,v),strain(g,v)
    return (transpose(BK)⋅D)⋅BK
end

function maxwell(mμ,l,λ,g,v::Val{3})
    n11,n22,n33,n12,n13,n23 = _nedelec(λ,g,v)
    m11 = (mμ+n11)*(l[1]*l[1])
    m22 = (mμ+n22)*(l[2]*l[2])
    m33 = (mμ+n33)*(l[3]*l[3])
    m12 = (mμ+n12)*(l[1]*l[2])
    m13 = (mμ+n13)*(l[1]*l[3])
    m23 = (mμ+n23)*(l[2]*l[3])
    Chain(Chain(m11,m12,m13),Chain(m12,m22,m23),Chain(m13,m23,m33))
end

function _nedelec(λ,g,v::Val{3})
    f = stiffness(λ,g,v)
    m11 = (f[3,3]-f[2,3]+f[2,2])/6
    m22 = (f[1,1]-f[1,3]+f[3,3])/6
    m33 = (f[2,2]-f[1,2]+f[1,1])/6
    m12 = (f[3,1]-f[3,3]-2f[2,1]+f[2,3])/12
    m13 = (f[3,2]-2f[3,1]-f[2,2]+f[2,1])/12
    m23 = (f[1,2]-f[1,1]-2f[3,2]+f[3,1])/12
    return m11,m22,m33,m12,m13,m23
end
function nedelec(λ,g,v::Val{3})
    m11,m22,m33,m12,m13,m23 = _nedelec(λ,g,v)
    Chain(Chain(m11,m12,m13),Chain(m12,m22,m23),Chain(m13,m23,m33))
end

function basisnedelec(p)
    M = Cartan.varmanifold(3); V = ↓(M)
    Chain{M,1}(
        Chain{V,1}(-p[2],p[1]),
        Chain{V,1}(-p[2],p[1]-1),
        Chain{V,1}(1-p[2],p[1]))
end

function interpnedelec(pt,ξ,t2e=edgesindices(pt),signs=facetsigns(pt))
    TensorField(pt,interp(immersion(pt),nedelecmean(pt,t2e,signs,fiber(ξ))))
end
function nedelecmean(t,t2e,signs,u)
    base = fiber(Grassmann.vectors(t))
    B = revrot.(base,revrot)./Real.(.∧(base))
    N = basisnedelec(Values(1,1)/3)
    x,y,z = columns(immersion(t2e)); X,Y,Z = columns(signs,1,3)
    (u[x].*X).*(B.⋅N[1]) + (u[y].*Y).*(B.⋅N[2]) + (u[z].*Z).*(B.⋅N[3])
end

function centroidvectors(t,m=means(t))
    p,nt = fullpoints(t),elements(t)
    V = Manifold(p)(2,3)
    c = Vector{Values{3,Chain{V,1,Float64,2}}}(undef,nt)
    δ = Vector{Values{3,Float64}}(undef,nt)
    ah = affinehull(t)
    for k ∈ 1:nt
        c[k] = V.(fiber(m)[k].-ah[k])
        δ[k] = value.(abs.(c[k]))
    end
    return c,δ
end

function jumps(t,c,a,f,u,m=volumes(t),g=gradienthat(t,m))
    N,np,nt = mdims(t),nodes(t),elements(t)
    η = zeros(nt)
    if N == 2
        fau = iterpts(t,fiber(f)).-fiber(a).*fiber(u)
        @threads for i ∈ 1:nt
            mi = fiber(m)[i]
            η[i] = mi*sqrt((fau[i]^2+fau[i+1]^2)*mi/2)
        end
    elseif N == 3
        ds,dn = trinormals(t) # ds.^1
        du,F,cols = Cartan.gradient_2(u,m,g),iterable(t,fiber(f)),columns(topology(t))
        fl = [-c*Real.(value(dn[k]).⋅fiber(du)[k]) for k ∈ 1:length(du)]
        intj = round.(adjacency(t,cols)/3)
        i,j,k = cols; x,y,z = getindex.(fl,1),getindex.(fl,2),getindex.(fl,3)
        jmps = sparse(j,k,x,np,np)+sparse(k,i,y,np,np)+sparse(i,j,z,np,np)
        jmps = abs.(intj.*abs.(jmps+jmps'))
        @threads for k = 1:nt
            tk,dsk = topology(t)[k],ds[k]
            η[k] = sqrt(((dsk[3]*jmps[tk[1],tk[2]])^2+(dsk[1]*jmps[tk[2],tk[3]])^2+(dsk[2]*jmps[tk[3],tk[1]])^2)/2)
        end
        η += [sqrt(norm(F[k].-a*fiber(u)[topology(t)[k]])/3fiber(m)[k]) for k ∈ 1:nt].*maximum.(ds)
    else
        throw(error("jumps on Manifold{$N} not defined"))
    end
    return TensorField(FaceBundle(t),η)
end

elementresiduals(f::TensorField{B,F,N,<:SimplexBundle} where {B,F,N}) = elementresiduals(means(f))
function elementresiduals(f::TensorField{B,F,N,<:FaceBundle} where {B,F,N})
    pt = base(f)
    TensorField(pt,maximum.(trilength(pt)).*sqrt.(fiber(volumes(pt)).*Real.(abs.(fiber(f)))))
end

function edgeresiduals(UV,E=1,ν=0.3,nbrs=neighbors(immersion(pt)))
    pt = base(UV)
    nt = elements(pt)
    rK = zeros(nt) # allocate edge residuals
    μ,λ = Eν2Lame(E,ν)
    m = volumes(pt)
    ds,dn = trinormals(pt)
    #dudx = fiber(gradient(UV))
    dudx1 = fiber(gradient_2(getindex.(UV,1)))
    dudx2 = fiber(gradient_2(getindex.(UV,2)))
    dudx = TensorOperator.(Chain{Manifold(dudx1)}.(dudx1,dudx2))
    for i ∈ OneTo(nt)
        r = 0 # sum of edge residuals sqrt(h)|0.5[n.sigma]|_dK
        dsi,dni,ni,dudxi = ds[i],dn[i],nbrs[i],dudx[i]
        h = maximum(dsi)
        for j ∈ Values(1,2,3) # loop over element edges
            n = ni[j] # element neighbour
            iszero(n) && continue # no neighbour, don't compute on boundary
            # iszero(n) ↦ should compute sqrt(h)|g_N-n.sigma|_dK
            Sp = stress(μ,λ,dudxi) # stress on element i
            Sm = stress(μ,λ,dudx[n]) # stress on neighbour
            jump = ((Sm-Sp)/2)⋅dni[j] # stress jump
            r += Real(abs(jump))*dsi[j]
        end
        rK[i] = sqrt(h)*sqrt(r)
    end
    return TensorField(FaceBundle(pt),rK)
end

function stress(μ,λ,dudx)
    divu = sum(value(diag(dudx)))
    ϵ = (dudx+transpose(dudx))/2
    (2μ)*ϵ + (λ*divu)*I
end

export shapeP1, shapeP2, gradientP1, gradientP2

shapeP1(rs::Values) = shapeP1(rs...)
shapeP1(r) = Chain(1-r,r)
shapeP1(r,s) = Chain(1-r-s,r,s)
shapeP1(r,s,t) = Chain(1-r-s-t,r,s,t)

gradientP1(rs::Values) = gradientP1(rs...)
function gradientP1(r)
    V = Cartan.affmanifold(1)
    TensorOperator(Chain(Chain{V}(-1),
        Chain{V}(1)))
end
function gradientP1(r,s)
    V = Cartan.affmanifold(2)
    TensorOperator(Chain(Chain{V}(-1,-1),
        Chain{V}(1,0),
        Chain{V}(0,1)))
end
function gradientP1(r,s,t)
    V = Cartan.affmanifold(3)
    TensorOperator(Chain(Chain{V}(-1,-1,-1),
        Chain{V}(1,0,0),
        Chain{V}(0,1,0),
        Chain{V}(0,0,1)))
end

shapeP2(rs::Values) = shapeP2(rs...)
function shapeP2(r,s)
    Chain(1-3r-3s+2r*r+4r*s+2s*s,
        2r*r-r,
        2s*s-s,
        4r*s,
        4s-4r*s-4s*s,
        4r-4r*r-4r*s)
end

gradientP2(rs::Values) = gradientP2(rs...)
function gradientP2(r,s)
    V = Cartan.affmanifold(2)
    TensorOperator(Chain(Chain{V}(-3+4r+4s,-3+4r+4s),
        Chain{V}(4r-1,0),
        Chain{V}(0,4s-1),
        Chain{V}(4s,4r),
        Chain{V}(-4s,4-4r-8s),
        Chain{V}(4-8r-4s,-4r)))
end

const GaussShape = Values(shapeP1.(Gauss[2][2]),shapeP2.(Gauss[4][2]))
const GaussGradient = Values(gradientP1.(Gauss[2][2]),gradientP2.(Gauss[4][2]))

function isoparametric(xy,dS)
    iJ,detJ = invdet(dS⋅transpose(xy))
    return iJ⋅dS,Real(detJ)
end

volumesPN(xy::Vector,dS::Values) = volumePN.(xy,Ref(dS))
volumesPN(xy::Vector,dS::TensorOperator) = volumePN.(xy,dS)
volumePN(xy::TensorOperator,dS::Values) = volumePN.(xy,dS)
volumePN(xy::TensorOperator,dS) = Real(det(dS⋅transpose(xy)))

export isoparametric, volumePN, volumesPN, assemblemassPN, assemblestiffnessPN

assemblemassP1(pt) = assemblemassPN(pt,Val(1))
assemblemassP2(pt) = assemblemassPN(pt,Val(2))
assemblemassPN(pt,N::Int) = assemblemassPN(pt,Val(N))
function assemblemassPN(pt,::Val{N}) where N
    qwgts = Adapode.Gauss[2N][1]/2 # quadrature rule
    S = Adapode.GaussShape[N] # quadrature shape
    dSrs = Adapode.GaussGradient[N] # quadrature gradient
    V,iter = Cartan.affmanifold(2),list(1,length(qwgts))
    dim = 3N
    np = nodes(pt) # number of nodes
    M = spzeros(np,np) # allocate mass matrix
    for i ∈ 1:elements(pt) # loop over elements
        ti = immersion(pt)[i] # node numbers
        xy = TensorOperator(Chain(V.(fullpoints(pt)[ti]))) #  node coordinates
        MK = zero(Chain{Submanifold(dim),1,Chain{Submanifold(dim),1,Float64}})
        for q ∈ iter # quadrature loop
            detJ = volumePN(xy,dSrs[q])
            wxarea = qwgts[q]*detJ # weight times area det(J), pre-divided by 2
            MK += outer(S[q],S[q]*wxarea) # compute and add integrand to MK
        end
        assemblelocal!(M,MK,ti)
    end
    return M#A,M,F
end

assemblestiffnessP1(pt) = assemblestiffnessPN(pt,Val(1))
assemblestiffnessP2(pt) = assemblestiffnessPN(pt,Val(2))
assemblestiffnessPN(pt,N::Int) = assemblestiffnessPN(pt,Val(N))
function assemblestiffnessPN(pt,::Val{N}) where N #,force)
    qwgts = Adapode.Gauss[2N][1]/2 # quadrature rule
    dSrs = Adapode.GaussGradient[N] # quadrature gradient
    V,iter = Cartan.affmanifold(2),list(1,length(qwgts))
    dim,vdim = 3N,Val(3N)
    np = nodes(pt) # number of nodes
    A = spzeros(np,np) # allocate stiffness matrix
    for i ∈ 1:elements(pt) # loop over elements
        ti = immersion(pt)[i] # node numbers
        xy = TensorOperator(Chain(V.(fullpoints(pt)[ti]))) #  node coordinates
        AK = zero(Chain{Submanifold(dim),1,Chain{Submanifold(dim),1,Float64}})
        for q ∈ iter # quadrature loop
            dSxy,detJ = isoparametric(xy,dSrs[q]) # quadrature gradient map
            wxarea = qwgts[q]*detJ # weight times area det(J), pre-divided by 2
            AK += stiffness(wxarea,dSxy,vdim) # element stiffness, wxarea*c, c=1
        end
        assemblelocal!(A,AK,ti)
    end
    return A#,M,F
end

function LagrangeP2(t::SimplexTopology)
    ei = edgesindices(t)
    ed = topology(ei).+Ref(Values(nodes(t),nodes(t),nodes(t))) # get element edges as nodes
    n = nodes(t)+nodes(ei)
    SimplexTopology(0,vcat.(topology(t),topology(ed)),OneTo(n),n)
end

shapemultilinear(rs::Values...) = shapemultilinear(rs...)
function shapemultilinear(r)
    Values(
        1-r,
        1+r)/2
end
function shapemultilinear(r,s)
    Values(
        (1-r)*(1-s),
        (1+r)*(1-s),
        (1+r)*(1+s),
        (1-r)*(1+s))/4
end
function shapemultilinear(r,s,t)
    Values(
        (1-r)*(1-s)*(1-t),
        (1+r)*(1-s)*(1-t),
        (1+r)*(1+s)*(1-t),
        (1-r)*(1+s)*(1-t),
        (1-r)*(1-s)*(1+t),
        (1+r)*(1-s)*(1+t),
        (1+r)*(1+s)*(1+t),
        (1-r)*(1+s)*(1+t))/8
end

gradientmultilinear(rs::Values...) = gradientmultilinear(rs...)
function gradientmultilinear(r)
    V = Cartan.affmanifold(1)
    Values(
        Chain{V}(-1),
        Chain{V}(+1))
end
function gradientmultilinear(r,s)
    V = Cartan.affmanifold(2)
    Values(
        Chain{V}(-1+s,-1+r),
        Chain{V}(+1-s,-1-r),
        Chain{V}(1+s,1+r),
        Chain{V}(-1-s,+1-r))
end
function gradientmultilinear(r,s,t)
    V = Cartan.affmanifold(3)
    Values(
        Chain{V}(-(1-s)*(1-t),-(1-r)*(1-t),-(1-r)*(1-s)),
        Chain{V}((1-s)*(1-t),-(1+r)*(1-t),-(1+r)*(1-s)),
        Chain{V}((1+s)*(1-t),(1+r)*(1-t),-(1+r)*(1+s)),
        Chain{V}(-(1+s)*(1-t),(1-r)*(1-t),-(1-r)*(1+s)),
        Chain{V}(-(1-s)*(1+t),-(1-r)*(1+t),(1-r)*(1-s)),
        Chain{V}((1-s)*(1+t),-(1+r)*(1+t),(1+r)*(1-s)),
        Chain{V}((1+s)*(1+t),(1+r)*(1+t),(1+r)*(1+s)),
        Chain{V}(-(1+s)*(1+t),(1-r)*(1+t),(1-r)*(1+s)))
end


