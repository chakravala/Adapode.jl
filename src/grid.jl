
#   This file is part of Adapode.jl
#   It is licensed under the AGPL license
#   Adapode Copyright (C) 2025 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

export helmholtz, solvehelmholtz, solvenonlinearhelmholtz, polarlaplacian
export reshapedirichlet, reshapepolardirichlet, solvepolardirichlet, orrsommerfeld
export solveiteration, dirichlet!, biharmonic

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

multiplierstep(multiplier,k,t) = (multiplier(k,t)-1)/t

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


