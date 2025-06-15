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

using SparseArrays, LinearAlgebra, Grassmann, Cartan, Requires
import Grassmann: value, vector, valuetype, tangent, list
import Grassmann: Values, Variables, FixedVector
import Grassmann: Scalar, GradedVector, Bivector, Trivector
import Base: @pure, OneTo
import Cartan: resize, resize_lastdim!, extract, assign!, geodesic

export Values, odesolve, odesolve2
export initmesh, pdegrad

export ElementFunction, IntervalMap, PlaneCurve, SpaceCurve, SurfaceGrid, ScalarGrid
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export RealFunction, ComplexMap, SpinorField, CliffordField
export MeshFunction, GradedField, QuaternionField # PhasorField
export Section, FiberBundle, AbstractFiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype
export ProductSpace, RealRegion, Interval, Rectangle, Hyperrectangle, ⧺, ⊕

include("constants.jl")
include("element.jl")

export AbstractIntegrator, StepIntegrator, AdaptiveIntegrator
export EulerHeunIntegrator, ExplicitIntegrator, ExplicitAdaptor
export MultistepIntegrator, MultistepAdaptor

abstract type AbstractIntegrator end
abstract type StepIntegrator <: AbstractIntegrator end
abstract type AdaptiveIntegrator <: AbstractIntegrator end

struct EulerHeunIntegrator <: StepIntegrator
    tol::Float64
    skip::Int
    geo::Bool
end

struct ExplicitIntegrator{o} <: StepIntegrator
    tol::Float64
    skip::Int
    geo::Bool
end

struct ExplicitAdaptor{o} <: AdaptiveIntegrator
    tol::Float64
    skip::Int
    geo::Bool
end

struct MultistepIntegrator{o} <: StepIntegrator
    tol::Float64
    skip::Int
    geo::Bool
end

struct MultistepAdaptor{o} <: AdaptiveIntegrator
    tol::Float64
    skip::Int
    geo::Bool
end

AbstractIntegrator(tol=15,int::AbstractIntegrator=ExplicitIntegrator{4}) = int(tol)
AbstractIntegrator(tol,m,o=4) = AbstractIntegrator(tol,Val(m),Val(o))
function AbstractIntegrator(tol,M::Val{m},B::Val{o}=Val(4)) where {m,o}
    int = if m == 0
        EulerHeunIntegrator(tol)
    elseif m == 1
        ExplicitIntegrator{o}(tol)
    elseif m == 2
        ExplicitAdaptor{o}(tol)
    elseif m == 3
        MultistepIntegrator{o}(tol)
    elseif m == 4
        MultistepAdaptor{o}(tol)
    end
end

EulerHeunIntegrator(tol,skip=1) = EulerHeunIntegrator(tol,skip,false)
EulerHeunIntegrator(tol::Int,skip::Int=1,geo=false) = EulerHeunIntegrator(2.0^-tol,skip,geo)
for fun ∈ (:ExplicitIntegrator,:ExplicitAdaptor,:MultistepIntegrator,:MultistepAdaptor)
    @eval begin
        $fun{o}(tol,skip=1) where o = $fun{o}(tol,skip,false)
        $fun{o}(tol::Int,skip::Int=1,geo=false) where o = $fun{o}(2.0^-tol,skip,geo)
    end
end

mutable struct TimeStep{T}
    h::T # step
    hmin::T # min step
    hmax::T # max step
    emin::T
    emax::T
    e::T
    i::Int
    s::Int
    function TimeStep(h,hmin=1e-16,hmax=h>1e-4 ? h : 1e-4,emin=10^(log2(h)-3),emax=10^log2(h))
        checkstep!(new{typeof(h)}(h,hmin,hmax,emin,emax,(emin+emax)/2,1,0))
    end
end
TimeStep(I::AbstractIntegrator) = TimeStep(I.tol)

Base.step(t::TimeStep) = t.h

function checkstep!(t)
    abs(step(t)) < t.hmin && (t.h = copysign(t.hmin,step(t)))
    abs(step(t)) > t.hmax && (t.h = copysign(t.hmax,step(t)))
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
@pure shift(::Val{m},::Val{k}=Val(1)) where {m,k} = Values{m,Int}(list(k,m+k-1))
#@pure shift(M::Val{m},i) where m = ((shift(M,Val{0}()).+i).%m).+1
@pure shift(::Val{m},l::Val,i) where m = ((shift(l,Val(0)).+i).%m).+1
function explicit(x,h,c,fx)
    l = length(c)
    fiber(x)+weights(h*c,l≠length(fx) ? fx[shift(Val(l))] : fx)
end
function explicit(x,h,c,fx,i)
    l,m = length(c),length(fx)
    weights(h*c,fx[shift(Val(m),Val(l),i+(m-l))])
end
function heun(x,f::Function,h)
    hfx = h*localfiber(f(x))
    fiber(x)+(hfx+h*fiber(f((point(x)+h)↦(fiber(x)+hfx))))/2
end
function heun(x,f::TensorField,h)
    hfx = h*localfiber(f(x))
    fiber(x)+(hfx+h*localfiber(f((point(x)+h)↦(fiber(x)+hfx))))/2
end
function heun2(x,f::TensorField,t)
    fx = f[t.i]
    hfx = step(t)*fx
    fiber(x)+(hfx+step(t)*localfiber(f(point(fx)↦(fiber(x)+hfx))))/2
end

@pure butcher(::Val{N},::Val{A}=Val(false)) where {N,A} = A ? CBA[N] : CB[N]
@pure blength(n::Val{N},a::Val{A}=Val(false)) where {N,A} = Val(length(butcher(n,a))-A)
function butcher(x::Section{B,F},f,h,v::Val{N}=Val(4),a::Val{A}=Val(false)) where {N,A,B,F}
    b = butcher(v,a)
    n = length(b)-A
    # switch to isbits(F)
    fx = F<:AbstractArray ? FixedVector{n,F}(undef) : Variables{n,F}(undef)
    @inbounds fx[1] = localfiber(f(x))
    for k ∈ 2:n
        @inbounds fx[k] = localfiber(f((point(x)+h*sum(b[k-1]))↦explicit(x,h,b[k-1],fx)))
    end
    return fx
end
explicit(x,f::Function,h,b::Val=Val(4)) = explicit(x,h,butcher(b)[end],butcher(x,f,h,b))
explicit(x,f::Function,h,::Val{1}) = fiber(x)+h*localfiber(f(x))
explicit(x,f::TensorField,h,b::Val=Val(4)) = explicit(x,h,butcher(b)[end],butcher(x,f,h,b))
explicit(x,f::TensorField,h,::Val{1}) = fiber(x)+h*localfiber(f(x))

function multistep!(x,f,fx,t,::Val{k}=Val(4),::Val{PC}=Val(false)) where {k,PC}
    fx[t.s] = localfiber(f(x))
    explicit(x,t.h,PC ? CAM[k] : CAB[k],fx,t.s)
end # more accurate compared with CAB[k] methods
function predictcorrect(x,f,fx,t,k::Val{m}=Val(4)) where m
    iszero(t.s) && initsteps!(x,f,fx,t)
    xti = typeof(x)<:LocalTensor ? x : extract(x,t.i)
    xi,tn = fiber(xti),point(xti)+step(t)
    xn = multistep!(xti,f,fx,t,k)
    t.s = (t.s%(m+1))+1; t.i += 1
    xn = multistep!(tn↦(xi+xn),f,fx,t,k,Val(true))
    return xi + xn
end
function predictcorrect(x,f,fx,t,::Val{1})
    xti = extract(x,t.i)
    t.i += 1
    h = step(t)
    fiber(xti)+h*localfiber(f((point(xti)+h)↦(fiber(xti)+h*localfiber(f(xti)))))
end

initsteps(x0,t,tmax,m) = initsteps(init(x0),t,tmax,m)
initsteps(x0,t,tmax,f,m,B) = initsteps(init(x0),t,tmax,f,m,B)
function initsteps(x0::Section,t,tmax,::Val{m}) where m
    tmin,f0 = base(x0),fiber(x0)
    n = Int(round((tmax-tmin)/step(t)))+1
    t = m ? (tmin:step(t):tmax) : Vector{typeof(t.h)}(undef,n)
    (!m) && (t[1] = tmin)
    x = Array{fibertype(fibertype(x0)),ndims(f0)+1}(undef,size(f0)...,m ? length(t) : n)
    assign!(x,1,fiber(f0))
    return TensorField(ndims(f0) > 0 ? base(f0)×t : t,x)
end
function initsteps(x0::Section,t,tmax,f,m,B::Val{o}=Val(4)) where o
    initsteps(x0,t,tmax,m), Variables{o+1,fibertype(x0)}(undef)
end

function multistep2!(x,f,fx,t,::Val{k}=Val(4),::Val{PC}=Val(false)) where {k,PC}
    @inbounds fx[t.s] = localfiber(f(x))
    @inbounds explicit(x,step(t),PC ? CAM[k] : CAB[k-1],fx,t.s)
end
function predictcorrect2(x,f,fx,t,k::Val{m}=Val(4)) where m
    iszero(t.s) && initsteps!(x,f,fx,t)
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),point(xti)+step(t)
    xn = multistep2!(xti,f,fx,t,k)
    t.s = (t.s%m)+1; t.i += 1
    xn = multistep2!(tn↦(xi+xn),f,fx,t,k,Val(true))
    t.s = (t.s%m)+1
    return xi + xn
end
function predictcorrect2(x,f,fx,t,::Val{1})
    @inbounds xti = x[t.i]
    t.i += 1
    fiber(xti)+step(t)*localfiber(f((point(xti)+step(t))↦xti))
end

initsteps2(x0,t,tmax,f,m,B) = initsteps2(init(x0),t,tmax,f,m,B)
function initsteps2(x0::Section,t,tmax,f,m,B::Val{o}=Val(4)) where o
    initsteps(x0,t,tmax,m), Variables{o,fibertype(x0)}(undef)
end

function initsteps!(x::LocalTensor,f,fx,t,B::Val=Val(4))
    m = length(fx)-2
    xi = x
    for j ∈ 1:m
        @inbounds fx[j] = localfiber(f(xi))
        xi = (point(xi)+step(t)) ↦ explicit(xi,f,step(t),B)
    end
    t.s = 1+m
    t.i += m
end
function initsteps!(x,f,fx,t,B::Val=Val(4))
    m = length(fx)-2
    xi = extract(x,t.i)
    for j ∈ 1:m
        @inbounds fx[j] = localfiber(f(xi))
        xi = (point(xi)+step(t)) ↦ explicit(xi,f,step(t),B)
        assign!(x,t.i+j,xi)
    end
    t.s = 1+m
    t.i += m
end

function explicit!(x,f,t,B=Val(5))
    resize!(x,t.i,10000)
    xti = extract(x,t.i)
    xi = fiber(xti)
    fx,b = butcher(xti,f,step(t),B,Val(true)),butcher(B,Val(true))
    t.i += 1
    assign!(x,t.i,(point(xti)+step(t)) ↦ explicit(xti,step(t),b[end-1],fx))
    @inbounds t.e = maximum(abs.(step(t)*value(b[end]⋅fx)))
end

function predictcorrect!(x,f,fx,t,B::Val{m}=Val(4)) where m
    resize!(x,t.i+m,10000)
    iszero(t.s) && initsteps!(x,f,fx,t)
    xti = extract(x,t.i)
    xi,tn = fiber(xti),point(xti)+step(t)
    p = xi + multistep!(xti,f,fx,t,B)
    t.s = (t.s%(m+1))+1
    c = xi + multistep!(tn↦p,f,fx,t,B,Val(true))
    t.i += 1
    assign!(x,t.i,tn ↦ c)
    t.e = maximum(abs.(value(c-p)./value(c)))
end
function predictcorrect!(x,f,fx,t,::Val{1})
    xti = extract(x,t.i)
    xi,tn = fiber(xti),point(xti)+step(t)
    p = xi + step(t)*localfiber(f(xti))
    c = xi + step(t)*localfiber(f(tn↦p))
    t.i += 1
    resize!(x,t.i,10000)
    assign!(x,t.i,tn ↦ c)
    t.e = maximum(abs.(value(c-p)./value(c)))
end

init(x0) = 0.0 ↦ x0
init(x0::Section) = x0

export Flow, FlowIntegral, InitialCondition, integrator

struct Flow{F}
    f::F
    t::Float64
end

Flow(f::Flow) = f
Flow(f) = Flow(f,2π)
system(Φ::Flow) = Φ.f
duration(Φ::Flow) = Φ.t

(Φ::Flow)(x0,i=ExplicitIntegrator{4}(2^-15,0)) = odesolve(InitialCondition(Φ,x0),i)
(Φ::Flow)(x0::LocalTensor,i=ExplicitIntegrator{4}(2^-15,0)) = Flow(system(Φ),duration(Φ)+base(x0))(fiber(x0),i)

function (Φ::Flow)(x0::TensorField,i=ExplicitIntegrator{4}(2^-11,0))
    ϕ = Flow(t -> TensorField(base(fiber(t)),Φ.f.(fiber(fiber(t)))),duration(Φ))
    odesolve(InitialCondition(ϕ,x0),i)
end

struct FlowIntegral{F,I<:AbstractIntegrator}
    Φ::Flow{F}
    i::I
    FlowIntegral(Φ::Flow{F},i::I=ExplicitIntegrator{4}(2^-11)) where {F,I<:AbstractIntegrator} = new{F,I}(Φ,i)
end

FlowIntegral(f,tmax,i=ExplicitIntegrator{4}(2^-11)) = FlowIntegral(Flow(f,tmax))
FlowIntegral(f) = FlowIntegral(f,2π)

Flow(Φ::FlowIntegral) = Φ.Φ
system(Φ::FlowIntegral) = system(Φ.Φ)
duration(Φ::FlowIntegral) = duration(Φ.Φ)
integrator(Φ::FlowIntegral) = Φ.i

(Φ::FlowIntegral)(x0) = Flow(Φ)(x0,integrator(Φ))

struct InitialCondition{F,X}
    Φ::Flow{F}
    x0::X
    InitialCondition(Φ::Flow{F},x0::X) where {F,X} = new{F,X}(Φ,x0)
end

InitialCondition(f,x0,tmax) = InitialCondition(Flow(f,tmax),x0)
InitialCondition(f,x0) = InitialCondition(f,x0,2π)

Flow(ic::InitialCondition) = ic.Φ
system(ic::InitialCondition) = system(Flow(ic))
duration(ic::InitialCondition) = duration(Flow(ic))
parameter(ic::InitialCondition) = ic.x0
(I::AbstractIntegrator)(ic::InitialCondition) = odesolve(ic,I)

odesolve(f,x0,tmax,tol,m,o=4) = odesolve(f,x0,tmax,tol,Val(m),Val(o))
function odesolve(f,x0,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    odesolve(InitialCondition(f,x0,tmax),AbstractIntegrator(tol,M,B))
end
function odesolve(ic::InitialCondition,I::EulerHeunIntegrator)
    t = TimeStep(I)
    x = initsteps(parameter(ic),t,duration(ic),Val(true))
    for i ∈ 2:size(x)[end]
        assign!(x,i,heun(extract(x,i-1),system(ic),step(t)))
    end
    return x
end
function odesolve(ic::InitialCondition,I::ExplicitIntegrator{o}) where o
    t,B = TimeStep(I),Val(o)
    stp = step(t)
    if iszero(I.skip) # don't allocate
        xi = init(parameter(ic))
        n = Int(round((duration(ic)-point(xi))/stp))
        for i ∈ 2:abs(n)+1
            xi = LocalTensor(point(xi)+sign(n)*stp,explicit(xi,system(ic),sign(n)*stp,B))
        end
        return xi
    elseif isone(I.skip) # full allocations
        x = initsteps(parameter(ic),t,duration(ic),Val(true))
        for i ∈ 2:size(x)[end]
            assign!(x,i,explicit(extract(x,i-1),system(ic),stp,B))
        end
        return x
    else # skip some allocations
        x = initsteps(parameter(ic),t,duration(ic),Val(true))
        skip = list(1,I.skip)
        for i ∈ 2:size(x)[end]
            xi = extract(x,i-1)
            for i ∈ skip
                xi = LocalTensor(point(xi)+stp,explicit(xi,system(ic),stp,B))
            end
            assign!(x,i,fiber(xi))
        end
        return x
    end
end
function odesolve(ic::InitialCondition,I::ExplicitAdaptor{o}) where o
    t,B = TimeStep(I),Val(o)
    x = initsteps(parameter(ic),t,duration(ic),Val(false))
    while timeloop!(x,t,duration(ic))
        explicit!(x,system(ic),t,B)
    end
    return resize(x)
end
function odesolve(ic::InitialCondition,I::MultistepIntegrator{o}) where o
    t,B = TimeStep(I),Val(o)
    stp = step(t)
    if iszero(I.skip) # don't allocate
        xi = init(parameter(ic))
        fx = Variables{o+1,fibertype(xi)}(undef)
        n = Int(round((duration(ic)-point(xi))/stp))
        pxi = point(xi)+(o-1)*sign(n)*stp
        for i ∈ o+1:abs(n)+1 # o+1 changed to o
            xi = LocalTensor(pxi+sign(n)*stp,predictcorrect(xi,system(ic),fx,t,B))
            pxi = point(xi)
        end
        return xi
    elseif isone(I.skip) # full allocations
        x,fx = initsteps(parameter(ic),t,duration(ic),system(ic),Val(true),B)
        for i ∈ o+1:size(x)[end] # o+1 changed to o
            assign!(x,i,predictcorrect(x,system(ic),fx,t,B))
        end
        return x
    else # skip some allocations
        x,fx = initsteps(parameter(ic),t,duration(ic),system(ic),Val(true),B)
        skip = list(1,I.skip)
        for i ∈ o+1:size(x)[end] # o+1 changed to o
            xi = extract(x,i-1)
            for i ∈ skip
                xi = LocalTensor(point(xi)+stp,predictcorrect(x,system(ic),fx,t,B))
            end
            assign!(x,i,fiber(xi))
        end
        return x
    end
end
function odesolve(ic::InitialCondition,I::MultistepAdaptor{o}) where o
    t,B = TimeStep(I),Val(o)
    x,fx = initsteps(parameter(ic),t,duration(ic),system(ic),Val(false),B)
    while timeloop!(x,t,duration(ic),B) # o+1 fix??
        predictcorrect!(x,system(ic),fx,t,B)
    end
    return resize(x)
end

#integrange
#integadapt

odesolve2(f,x0,tmax,tol,m,o=4) = odesolve2(f,x0,tmax,tol,Val(m),Val(o))
function odesolve2(f,x0,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    odesolve2(InitialCondition(f,x0,tmax),AbstractIntegrator(tol,M,B))
end
odesolve2(ic::InitialCondition,I::ExplicitIntegrator) = odesolve(ic,I)
odesolve2(ic::InitialCondition,I::ExplicitAdaptor) = odesolve(ic,I)
odesolve2(ic::InitialCondition,I::MultistepAdaptor) = odesolve(ic,I)
function odesolve2(ic::InitialCondition,I::MultistepIntegrator{o}) where o
    t,B = TimeStep(I),Val(o)
    x,fx = initsteps2(parameter(ic),t,duration(ic),system(ic),Val(true),B)
    for i ∈ (isone(o) ? 2 : o):size(x)[end] # o+1 changed to o
        assign!(x,i,predictcorrect2(x,system(ic),fx,t,B))
    end
    return x
end

function integrate(f::TensorField,x,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    x0,t = init(x),TimeStep(2.0^-tol)
    if m == 0 # Improved Euler, Heun's Method
        x = initsteps(x0,t,tmax,Val(true))
        for i ∈ 2:length(x)
            assign!(x,i,heun(extract(x,i-1),f,step(t)))
        end
    elseif m == 3 # Multistep
        x,fx = initsteps(x0,t,tmax,f,Val(true),B)
        for i ∈ o+1:length(x)
            assign!(x,i,predictcorrect(x,f,fx,t,B))
        end
    end
    return resize(x)
end

export geosolve, geodesic, GeodesicCondition, Geodesic

geodesic(Γ,x0v0) = InitialCondition(geodesic(Γ),x0v0)
geodesic(Γ,x0v0,tmax::AbstractReal) = InitialCondition(geodesic(Γ),x0v0,tmax)
geodesic(Γ,x0,v0) = GeodesicCondition(Γ,x0,v0,2π)
geodesic(Γ,x0,v0,tmax::AbstractReal) = InitialCondition(geodesic(Γ),Chain(x0,v0),tmax)
const Geodesic,GeodesicCondition = geodesic,geodesic,geodesic

geosolve(ic::InitialCondition,i::AbstractIntegrator) = getindex.(odesolve(ic,i),1)
geosolve(Γ,x0,v0,tmax,tol,m,o=4) = geosolve(Γ,x0,v0,tmax,tol,Val(m),Val(o))
function geosolve(Γ,x0,v0,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    getindex.(odesolve(geodesic(Γ),Chain(x0,v0),tmax,tol,M,B),1)
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
    d = tmax-point(x[t.i])
    d ≤ step(t) && (t.h = d)
    done = d ≤ t.hmax
    done && truncate!(x,t.i-1)
    return !done
end

time(x) = x[1]

Base.resize!(x,i,h) = length(x)<i+1 && resize_lastdim!(x,i+h)
truncate!(x,i) = length(x)>i+1 && resize_lastdim!(x,i)

show_progress(x,t,b) = t.i%75000 == 11 && println(point(x[t.i])," out of ",b)

end # module
