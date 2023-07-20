module Adapode

#   This file is part of Aadapode.jl
#   It is licensed under the GPL license
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
#   _______     __                          __
#  |   _   |.--|  |.---.-..-----..-----..--|  |.-----.
#  |       ||  _  ||  _  ||  _  ||  _  ||  _  ||  -__|
#  |___|___||_____||___._||   __||_____||_____||_____|
#                         |__|

using SparseArrays, LinearAlgebra
using AbstractTensors, DirectSum, Grassmann
import Grassmann: value, vector, valuetype, tangent
import Base: @pure
import AbstractTensors: Values, Variables, FixedVector
import AbstractTensors: Scalar, GradedVector, Bivector, Trivector

export Values, odesolve
export initmesh, pdegrad

export ElementFunction, IntervalMap, PlaneCurve, SpaceCurve, Surface, ParametricMap
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export RealFunction, ComplexMap, ComplexMapping, SpinorField, CliffordField
#export QuaternionField # PhasorField

abstract type TensorCategory end
Base.@pure iscategory(::TensorCategory) = true
Base.@pure iscategory(::Any) = false

struct TensorField{T,S,N} <: TensorCategory
    dom::T
    cod::Array{S,N}
end

#const ParametricMesh{T<:AbstractVector{<:Chain},S} = TensorField{T,S,1}
const ElementFunction{T<:AbstractVector,S<:Real} = TensorField{T,S,1}
const IntervalMap{T<:AbstractVector{<:Real},S} = TensorField{T,S,1}
const RealFunction{T<:AbstractVector{<:Real},S<:Real} = ElementFunction{T,S}
const PlaneCurve{T<:AbstractVector{<:Real},S<:Chain{V,G,R,2} where {V,G,R}} = TensorField{T,S,1}
const SpaceCurve{T<:AbstractVector{<:Real},S<:Chain{V,G,R,3} where {V,G,R}} = TensorField{T,S,1}
const Surface{T<:AbstractMatrix,S<:Real} = TensorField{T,S,2}
const ParametricMap{T<:AbstractArray{<:Real},S,N} = TensorField{T,S,N}
const ComplexMapping{T,S<:Complex,N} = TensorField{T,S,N}
const ComplexMap{T,S<:Couple,N} = TensorField{T,S,N}
const ScalarField{T,S<:Single,N} = TensorField{T,S,N}
const VectorField{T,S<:Chain{V,1} where V,N} = TensorField{T,S,N}
const SpinorField{T,S<:Spinor,N} = TensorField{T,S,N}
#const PhasorField{T,S<:Phasor,N} = TensorField{T,S,N}
#const QuaternionField{T,S<:Quaternion,N} = TensorField{T,S,N}
const CliffordField{T,S<:Multivector,N} = TensorField{T,S,N}
const BivectorField{T,S<:Chain{V,2} where V,N} = TensorField{T,S,N}
const TrivectorField{T,S<:Chain{V,3} where V,N} = TensorField{T,S,N}

parametric(dom::Vector{T},cod::Vector{S}) where {T,S} = TensorField{Vector{T},S,1}(dom,cod)
parametric(dom,fun::Function) = TensorField(collect(dom),fun.(dom))

function (m::IntervalMap{Vector{Float64}})(t)
    i = searchsortedfirst(m.dom,t)-1
    m.cod[i]+(t-m.dom[i])/(m.dom[i+1]-m.dom[i])*(m.cod[i+1]-m.cod[i])
end
function (m::IntervalMap{Vector{Float64}})(t::Vector,d=diff(m.cod)./diff(m.dom))
    [parametric(i,m,d) for i ∈ t]
end
function parametric(t,m,d=diff(m.cod)./diff(m.dom))
    i = searchsortedfirst(m.dom,t)-1
    m.cod[i]+(t-m.dom[i])*d[i]
end

centraldiff_fast(f::Vector,dt::Float64,l=length(f)) = [centraldiff_fast(i,f,l)/centraldiff_fast(i,dt,l) for i ∈ 1:l]
centraldiff_fast(f::Vector,dt::Vector,l=length(f)) = [centraldiff_fast(i,f,l)/dt[i] for i ∈ 1:l]
centraldiff_fast(f::Vector,l=length(f)) = [centraldiff_fast(i,f,l) for i ∈ 1:l]
function centraldiff_fast(i::Int,f::Vector{<:Chain},l=length(f))
    if isone(i) # 4f[2]-f[3]-3f[1]
        18f[2]-9f[3]+2f[4]-11f[1]
    elseif i==l # 3f[end]-4f[end-1]+f[end-2]
        11f[end]-18f[end-1]+9f[end-2]-2f[end-3]
    else
        f[i+1]-f[i-1]
    end
end
centraldiff_fast(dt::Float64,l::Int) = [centraldiff_fast(i,dt,l) for i ∈ 1:l]
centraldiff_fast(i::Int,dt::Float64,l::Int) = i∈(1,l) ? 6dt : 2dt
#centraldiff_fast(i::Int,dt::Float64,l::Int) = 2dt

centraldiff(f::Vector,dt::Float64,l=length(f)) = [centraldiff(i,f,l)/centraldiff(i,dt,l) for i ∈ 1:l]
centraldiff(f::Vector,dt::Vector,l=length(f)) = [centraldiff(i,f,l)/dt[i] for i ∈ 1:l]
centraldiffdiff(f,dt,l) = centraldiff(centraldiff(f,dt,l),dt,l)
centraldiffdiff(f::Vector,dt) = centraldiffdiff(f,dt,length(f))
centraldiff(f::Vector,l=length(f)) = [centraldiff(i,f,l) for i ∈ 1:l]
function centraldiff(i::Int,f::Vector,l::Int=length(f))
    if isone(i)
        18f[2]-9f[3]+2f[4]-11f[1]
    elseif i==l
        11f[end]-18f[end-1]+9f[end-2]-2f[end-3]
    elseif i==2
        6f[3]-f[4]-3f[2]-2f[1]
    elseif i==l-1
        3f[end-1]-6f[end-2]+f[end-3]+2f[end]
    else
        f[i-2]+8f[i+1]-8f[i-1]-f[i+2]
    end
end
centraldiff(dt::Float64,l::Int) = [centraldiff(i,dt,l) for i ∈ 1:l]
function centraldiff(i::Int,dt::Float64,l::Int)
    if i∈(1,2,l-1,l)
        6dt
    else
        12dt
    end
end

function centraldiff_fast(i::Int,h::Int,f::Vector{<:Chain},l=length(f))
    if isone(i) # 4f[2]-f[3]-3f[1]
        18f[2]-9f[3]+2f[4]-11f[1]
    elseif i==l # 3f[end]-4f[end-1]+f[end-2]
        11f[end]-18f[end-1]+9f[end-2]-2f[end-3]
    else
        (i-h<1)||(i+h)>l ? centraldiff_(i,h-1,f,l) : f[i+h]-f[i-h]
    end
end

qnumber(n,q) = (q^n-1)/(q-1)
qfactorial(n,q) = prod(cumsum([q^k for k ∈ 0:n-1]))
qbinomial(n,k,q) = qfactorial(n,q)/(qfactorial(n-k,q)*qfactorial(k,q))

richardson(k) = [(-1)^(j+k)*2^(j*(1+j))*qbinomial(k,j,4)/prod([4^n-1 for n ∈ 1:k]) for j ∈ k:-1:0]

export arclength, trapz, linetrapz
export centraldiff, tangent, unittangent, speed, normal, unitnormal

arclength(f::Vector) = sum(abs.(diff(f)))
function arclength(f::IntervalMap)
    int = cumsum(abs.(diff(f.cod)))
    pushfirst!(int,zero(eltype(int)))
    TensorField(f.dom,int)
end # trapz(speed(f))
function trapz(f::IntervalMap,d=diff(f.dom))
    int = (d/2).*cumsum(f.cod[2:end]+f.cod[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    return TensorField(f.dom,int)
end
function trapz(f::Vector,h::Float64)
    int = (h/2)*cumsum(f[2:end]+f[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    return int
end
function linetrapz(γ::IntervalMap,f::Function)
    trapz(TensorField(γ.dom,f.(γ.cod).⋅tangent(γ).cod))
end
function tangent(f::IntervalMap,dt=centraldiff(f.dom))
    TensorField(f.dom,centraldiff(f.cod,dt))
end
function unittangent(f::IntervalMap,d=centraldiff(f.dom),t=centraldiff(f.cod,d))
    TensorField(f.dom,t./abs.(t))
end
function speed(f::IntervalMap,d=centraldiff(f.dom),t=centraldiff(f.cod,d))
    TensorField(f.dom,abs.(t))
end
function normal(f::IntervalMap,d=centraldiff(f.dom),t=centraldiff(f.cod,d))
    TensorField(f.dom,centraldiff(t,d))
end
function unitnormal(f::IntervalMap,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,n./abs.(n))
end

export curvature, radius, osculatingplane, unitosculatingplane, binormal, unitbinormal, curvaturebivector, torsion, curvaturetrivector, trihedron, frenet, wronskian, bishoppolar, bishop, curvaturetorsion

function curvature(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    TensorField(f.dom,abs.(centraldiff(t./a,d))./a)
end
function radius(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    TensorField(f.dom,a./abs.(centraldiff(t./a,d)))
end
function osculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,t.∧n)
end
function unitosculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,(t./abs.(t)).∧(n./abs.(n)))
end
function binormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,.⋆(t.∧n))
end
function unitbinormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    TensorField(f.dom,.⋆((t./abs.(t)).∧(n./abs.(n))))
end
function curvaturebivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    a=abs.(t); ut=t./a
    TensorField(f.dom,abs.(centraldiff(ut,d))./a.*(ut.∧(n./abs.(n))))
end
function torsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    TensorField(f.dom,(b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
function curvaturetrivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a=abs.(t); ut=t./a
    TensorField(f.dom,((abs.(centraldiff(ut,d)./a).^2).*(b.∧centraldiff(n,d))./abs.(.⋆b).^2))
end
#torsion(f::TensorField,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),a=abs.(t)) = TensorField(f.dom,abs.(centraldiff(.⋆((t./a).∧(n./abs.(n))),d))./a)
function trihedron(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    Chain.(ut,un,.⋆(ut.∧un))
end
function frenet(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    centraldiff(Chain.(ut,un,.⋆(ut.∧un)),d)
end
function wronskian(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    f.cod.∧t.∧n
end

#???
function compare(f::TensorField)#???
    d = centraldiff(f.dom)
    t = centraldiff(f.cod,d)
    n = centraldiff(t,d)
    centraldiff(t./abs.(t)).-n./abs.(t)
end #????

function curvaturetorsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a = abs.(t)
    TensorField(f.dom,Chain.(value.(abs.(centraldiff(t./a,d))./a),getindex.((b.∧centraldiff(n,d))./abs.(.⋆b).^2,1)))
end

function bishoppolar(f::SpaceCurve,κ=value.(curvature(f).cod))
    κ = value.()
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    TensorField(f.dom,Chain.(κ,θ))
end
function bishop(f::SpaceCurve,κ=value.(curvature(f).cod))
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    TensorField(f.dom,Chain.(κ.*cos.(θ),κ.*sin.(θ)))
end
#bishoppolar(f::TensorField) = TensorField(f.dom,Chain.(value.(curvature(f).cod),getindex.(trapz(torsion(f)).cod,1)))
#bishop(f::TensorField,κ=value.(curvature(f).cod),θ=getindex.(trapz(torsion(f)).cod,1)) = TensorField(f.dom,Chain.(κ.*cos.(θ),κ.*sin.(θ)))

include("constants.jl")
include("element.jl")

mutable struct TimeStep{T}
    h::T
    hmin::T
    hmax::T
    emin::T
    emax::T
    e::T
    i::Int
    s::Int
    function TimeStep(h,hmin=1e-16,hmax=1e-4,emin=10^(log2(h)-3),emax=10^log2(h))
        checkstep!(new{typeof(h)}(h,hmin,hmax,emin,emax,(emin+emax)/2,1,0))
    end
end
function checkstep!(t)
    abs(t.h) < t.hmin && (t.h = copysign(t.hmin,t.h))
    abs(t.h) > t.hmax && (t.h = copysign(t.hmax,t.h))
    t.i < 1 && (t.i = 1)
    return t
end

function weights(c,fx)
    @inbounds cfx = c[1]*fx[1]
    for k ∈ 2:length(c)
        @inbounds cfx += c[k]*fx[k]
    end
    return cfx
end
@pure shift(::Val{m},::Val{k}=Val(1)) where {m,k} = Values{m,Int}(k:m+k-1)
@pure shift(M::Val{m},i) where m = ((shift(M,Val{0}()).+i).%(m+1)).+1
explicit(x,h,c,fx) = (l=length(c);x+weights(h*c,l≠length(fx) ? fx[shift(Val(l))] : fx))
explicit(x,h,c,fx,i) = (l=length(c);weights(h*c,l≠length(fx) ? fx[shift(Val(l),i)] : fx))
improved_heun(x,f,h) = (fx = f(x); x+(h/2)*(fx+f(x+h*fx)))

@pure butcher(::Val{N},::Val{A}=Val(false)) where {N,A} = A ? CBA[N] : CB[N]
@pure blength(n::Val{N},a::Val{A}=Val(false)) where {N,A} = Val(length(butcher(n,a))-A)
function butcher(x,f,h,v::Val{N}=Val(4),a::Val{A}=Val(false)) where {N,A}
    b = butcher(v,a)
    n = length(b)-A
    T = typeof(x)
    fx = T<:Vector ? FixedVector{n,T}(undef) : Variables{n,T}(undef)
    @inbounds fx[1] = f(x)
    for k ∈ 2:n
        @inbounds fx[k] = f(explicit(x,h,b[k-1],fx))
    end
    return fx
end
explicit(x,f,h,b::Val=Val(4)) = explicit(x,h,butcher(b)[end],butcher(x,f,h,b))
explicit(x,f,h,::Val{1}) = x+h*f(x)

function multistep!(x,f,fx,t,K::Val{k}=Val(4),::Val{PC}=Val(false)) where {k,PC}
    @inbounds fx[((t.s+k-1)%(k+1))+1] = f(x)
    @inbounds explicit(x,t.h,PC ? CAM[k] : CAB[k],fx,t.s)
end
function predictcorrect(x,f,fx,t,k::Val{m}=Val(4)) where m
    iszero(t.s) && initsteps!(x,f,fx,t)
    xi = x[t.i]
    p = xi + multistep!(xi,f,fx,t,k)
    t.s = (t.s%(m+1))+1; t.i += 1
    c = xi + multistep!(p,f,fx,t,k,Val(true))
end
function predictcorrect(x,f,fx,t,::Val{1})
    xi = x[t.i]
    t.i += 1
    xi+t.h*f(xi+t.h*f(xi))
end

function initsteps(x0,t,tmin,tmax)
    x = Vector{typeof(x0)}(undef,Int(round((tmax-tmin)*2^-log2(t.h)))+1)
    x[1] = x0
    return x
end
function initsteps(x0,t,tmin,tmax,f,m::Val{o},B=Val(4)) where o
    initsteps(x0,t,tmin,tmax), Variables{o+1,typeof(x0)}(undef)
end

function initsteps!(x,f,fx,t,B=Val(4))
    m = length(fx)-2
    xi = x[t.i]
    for j ∈ 1:m
        @inbounds fx[j] = f(xi)
        xi = explicit(xi,f,t.h,B)
        x[t.i+j] = xi
    end
    t.i += m
end

function explicit!(x,f,t,B=Val(5))
    resize!(x,t.i,10000)
    @inbounds xi = x[t.i]
    fx,b = butcher(xi,f,t.h,B,Val(true)),butcher(B,Val(true))
    t.i += 1
    @inbounds x[t.i] = explicit(xi,t.h,b[end-1],fx)
    @inbounds t.e = maximum(abs.(t.h*value(b[end]⋅fx)))
end

function predictcorrect!(x,f,fx,t,B::Val{m}=Val(4)) where m
    resize!(x,t.i+m,10000)
    iszero(t.s) && initsteps!(x,f,fx,t)
    @inbounds xi = x[t.i]
    p = xi + multistep!(xi,f,fx,t,B)
    t.s = (t.s%(m+1))+1
    c = xi + multistep!(p,f,fx,t,B,Val(true))
    t.i += 1
    @inbounds x[t.i] = c
    t.e = maximum(abs.(value(c-p)./value(c)))
end
function predictcorrect!(x,f,fx,t,::Val{1})
    xi = x[t.i]
    p = xi + t.h*f(xi)
    c = xi + t.h*f(p)
    t.i += 1
    resize!(x,t.i,10000)
    @inbounds x[t.i] = c
    t.e = maximum(abs.(value(c-p)./value(c)))
end

odesolve(f,x0,tmin,tmax,tol,m,o) = odesolve(f,x0,∂,tol,Val(m),Val(o))
function odesolve(f,x0,tmin=0,tmax=2π,tol=15,::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    t = TimeStep(2.0^-tol)
    if m == 0 # Improved Euler, Heun's Method
        x = initsteps(x0,t,tmin,tmax)
        for i ∈ 2:length(x)
            @inbounds x[i] = improved_heun(x[i-1],f,t.h)
        end
    elseif m == 1 # Singlestep
        x = initsteps(x0,t,tmin,tmax)
        for i ∈ 2:length(x)
            @inbounds x[i] = explicit(x[i-1],f,t.h,B)
        end
    elseif m == 2 # Adaptive Singlestep
        x = initsteps(x0,t,tmin,tmax)
        while timeloop!(x,t,tmax)
            explicit!(x,f,t,B)
        end
    elseif m == 3 # Multistep
        x,fx = initsteps(x0,t,tmin,tmax,f,B)
        for i ∈ o+1:length(x)
            @inbounds x[i] = predictcorrect(x,f,fx,t,B)
        end
    else # Adaptive Multistep
        x,fx = initsteps(x0,t,tmin,tmax,f,B)
        while timeloop!(x,t,tmax,B)
            predictcorrect!(x,f,fx,t,B)
        end
    end
    return x
end

function timeloop!(x,t,tmax,::Val{m}=Val(1)) where m
    if t.e < t.emin
        t.h *= 2
        t.i -= Int(floor(m/2))
        t.s = 0
    end
    if t.e > t.emax
        t.h /= 2
        t.i -= Int(ceil(m/2))
        t.s = 0
    end
    iszero(t.s) && checkstep!(t)
    d = tmax-time(x[t.i])
    d ≤ t.h && (t.h = d)
    done = d ≤ t.hmax
    done && truncate!(x,t.i-1)
    return !done
end

time(x) = x[1]

Base.resize!(x,i,h) = length(x)<i+1 && resize!(x,i+h)
truncate!(x,i) = length(x)>i+1 && resize!(x,i)

show_progress(x,t,b) = t.i%75000 == 11 && println(time(x[t.i])," out of ",b)

end # module
