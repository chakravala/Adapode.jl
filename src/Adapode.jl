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
using AbstractTensors, DirectSum, Grassmann, TensorFields, Requires
import Grassmann: value, vector, valuetype, tangent
import Base: @pure
import AbstractTensors: Values, Variables, FixedVector
import AbstractTensors: Scalar, GradedVector, Bivector, Trivector

export Values, odesolve
export initmesh, pdegrad

export ElementFunction, IntervalMap, PlaneCurve, SpaceCurve, GridSurface, GridParametric
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export RealFunction, ComplexMap, ComplexMapping, SpinorField, CliffordField
export MeshFunction, GradedField, QuaternionField # PhasorField
export Section, FiberBundle, AbstractFiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype

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
explicit(x,h,c,fx) = (l=length(c);fiber(x)+weights(h*c,l≠length(fx) ? fx[shift(Val(l))] : fx))
explicit(x,h,c,fx,i) = (l=length(c);weights(h*c,l≠length(fx) ? fx[shift(Val(l),i)] : fx))
improved_heun(x,f,h) = (fx = f(x); fiber(x)+(h/2)*(fx+f((base(x)+h)↦(fiber(x)+h*fx))))

@pure butcher(::Val{N},::Val{A}=Val(false)) where {N,A} = A ? CBA[N] : CB[N]
@pure blength(n::Val{N},a::Val{A}=Val(false)) where {N,A} = Val(length(butcher(n,a))-A)
function butcher(x::Section{B,F},f,h,v::Val{N}=Val(4),a::Val{A}=Val(false)) where {N,A,B,F}
    b = butcher(v,a)
    n = length(b)-A
    fx = F<:Vector ? FixedVector{n,F}(undef) : Variables{n,F}(undef)
    @inbounds fx[1] = f(x)
    for k ∈ 2:n
        @inbounds fx[k] = f((base(x)+h*sum(b[k-1]))↦explicit(x,h,b[k-1],fx))
    end
    return fx
end
explicit(x,f,h,b::Val=Val(4)) = explicit(x,h,butcher(b)[end],butcher(x,f,h,b))
explicit(x,f,h,::Val{1}) = fiber(x)+h*f(x)

function multistep!(x,f,fx,t,K::Val{k}=Val(4),::Val{PC}=Val(false)) where {k,PC}
    @inbounds fx[((t.s+k-1)%(k+1))+1] = f(x)
    @inbounds explicit(x,t.h,PC ? CAM[k] : CAB[k],fx,t.s)
end
function predictcorrect(x,f,fx,t,k::Val{m}=Val(4)) where m
    iszero(t.s) && initsteps!(x,f,fx,t)
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),base(xti)+t.h
    p = xi + multistep!(xti,f,fx,t,k)
    t.s = (t.s%(m+1))+1; t.i += 1
    xi+multistep!(tn↦p,f,fx,t,k,Val(true))
end
function predictcorrect(x,f,fx,t,::Val{1})
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),base(xti)+t.h
    t.i += 1
    xi+t.h*f(tn↦(xi+t.h*f(xti)))
end

initsteps(x0::Section,t,tmin,tmax,m) = initsteps(fiber(x0),t,tmin,tmax,m)
initsteps(x0::Section,t,tmin,tmax,f,m,B) = initsteps(fiber(x0),t,tmin,tmax,f,m,B)
function initsteps(x0,t,tmin,tmax,::Val{m}) where m
    n = Int(round((tmax-tmin)*2^-log2(t.h)))+1
    t = m ∈ (0,1,3) ? (tmin:t.h:tmax) : Vector{typeof(t.h)}(undef,n)
    m ∉ (0,1,3) && (t[1] = tmin)
    x = Vector{typeof(x0)}(undef,n)
    x[1] = x0
    return TensorField(t,x)
end
function initsteps(x0,t,tmin,tmax,f,m,B::Val{o}=Val(4)) where o
    initsteps(x0,t,tmin,tmax,m), Variables{o+1,typeof(x0)}(undef)
end

function initsteps!(x,f,fx,t,B=Val(4))
    m = length(fx)-2
    @inbounds xi = x[t.i]
    for j ∈ 1:m
        @inbounds fx[j] = f(xi)
        xi = (base(xi)+t.h) ↦ explicit(xi,f,t.h,B)
        x[t.i+j] = xi
    end
    t.i += m
end

function explicit!(x,f,t,B=Val(5))
    resize!(x,t.i,10000)
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti)
    fx,b = butcher(xti,f,t.h,B,Val(true)),butcher(B,Val(true))
    t.i += 1
    @inbounds x[t.i] = (base(xti)+t.h) ↦ explicit(xti,t.h,b[end-1],fx)
    @inbounds t.e = maximum(abs.(t.h*value(b[end]⋅fx)))
end

function predictcorrect!(x,f,fx,t,B::Val{m}=Val(4)) where m
    resize!(x,t.i+m,10000)
    iszero(t.s) && initsteps!(x,f,fx,t)
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),base(xti)+t.h
    p = xi + multistep!(xti,f,fx,t,B)
    t.s = (t.s%(m+1))+1
    c = xi + multistep!(tn↦p,f,fx,t,B,Val(true))
    t.i += 1
    @inbounds x[t.i] = tn ↦ c
    t.e = maximum(abs.(value(c-p)./value(c)))
end
function predictcorrect!(x,f,fx,t,::Val{1})
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),base(xti)+t.h
    p = xi + t.h*f(xti)
    c = xi + t.h*f(tn↦p)
    t.i += 1
    resize!(x,t.i,10000)
    @inbounds x[t.i] = tn ↦ c
    t.e = maximum(abs.(value(c-p)./value(c)))
end

odesolve(f,x0,tmin,tmax,tol,m,o) = odesolve(f,x0,∂,tol,Val(m),Val(o))
function odesolve(f,x0,tmin=0,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    t = TimeStep(2.0^-tol)
    if m == 0 # Improved Euler, Heun's Method
        x = initsteps(x0,t,tmin,tmax,M)
        for i ∈ 2:length(x)
            @inbounds x[i] = improved_heun(x[i-1],f,t.h)
        end
    elseif m == 1 # Singlestep
        x = initsteps(x0,t,tmin,tmax,M)
        for i ∈ 2:length(x)
            @inbounds x[i] = explicit(x[i-1],f,t.h,B)
        end
    elseif m == 2 # Adaptive Singlestep
        x = initsteps(x0,t,tmin,tmax,M)
        while timeloop!(x,t,tmax)
            explicit!(x,f,t,B)
        end
    elseif m == 3 # Multistep
        x,fx = initsteps(x0,t,tmin,tmax,f,M,B)
        for i ∈ o+1:length(x)
            @inbounds x[i] = predictcorrect(x,f,fx,t,B)
        end
    else # Adaptive Multistep
        x,fx = initsteps(x0,t,tmin,tmax,f,M,B)
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

function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        Makie.lines(t::SpaceCurve;args...) = Makie.lines(t.cod;color=value.(speed(t).cod),args...)
        Makie.lines(t::PlaneCurve;args...) = Makie.lines(t.cod;color=value.(speed(t).cod),args...)
        Makie.lines(t::RealFunction;args...) = Makie.lines(t.dom,getindex.(value.(t.cod),1);color=value.(speed(t).cod),args...)
        Makie.linesegments(t::SpaceCurve;args...) = Makie.linesegments(t.cod;color=value.(speed(t).cod),args...)
        Makie.linesegments(t::PlaneCurve;args...) = Makie.linesegments(t.cod;color=value.(speed(t).cod),args...)
        Makie.linesegments(t::RealFunction;args...) = Makie.linesegments(t.dom,getindex.(value.(t.cod),1);color=value.(speed(t).cod),args...)
        Makie.volume(t::GridParametric{<:Chain,<:Hyperrectangle};args...) = Makie.volume(t.dom.v[1],t.dom.v[2],t.dom.v[3],t.cod;args...)
        Makie.volumeslices(t::GridParametric{<:Chain,<:Hyperrectangle};args...) = Makie.volumeslices(t.dom.v[1],t.dom.v[2],t.dom.v[3],t.cod;args...)
        Makie.surface(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.surface(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.contour(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.contour(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.contourf(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.contourf(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.contour3d(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.contour3d(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.heatmap(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.heatmap(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.wireframe(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.wireframe(t.dom.v[1],t.dom.v[2],t.cod;args...)
        #Makie.spy(t::GridSurface{<:Chain,<:RealRegion};args...) = Makie.spy(t.dom.v[1],t.dom.v[2],t.cod;args...)
        Makie.streamplot(f::Function,t::Rectangle;args...) = Makie.streamplot(f,t.v[1],t.v[2];args...)
        Makie.streamplot(f::Function,t::Hyperrectangle;args...) = Makie.streamplot(f,t.v[1],t.v[2],t.v[3];args...)
        Makie.streamplot(m::TensorField{R,<:RealRegion,<:Real} where R;args...) = Makie.streamplot(tangent(m);args...)
        Makie.streamplot(m::TensorField{R,<:ChainBundle,<:Real} where R,dims...;args...) = Makie.streamplot(tangent(m),dims...;args...)
        Makie.streamplot(m::VectorField{R,<:ChainBundle} where R,dims...;args...) = Makie.streamplot(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
        Makie.streamplot(m::VectorField{R,<:RealRegion} where R;args...) = Makie.streamplot(p->Makie.Point(m(Chain(p.data...))),domain(m).v...;args...)
        Makie.arrows(t::VectorField{<:Chain,<:Rectangle};args...) = Makie.arrows(t.dom.v[1],t.dom.v[2],getindex.(t.cod,1),getindex.(t.cod,2);args...)
        Makie.arrows!(t::VectorField{<:Chain,<:Rectangle};args...) = Makie.arrows!(t.dom.v[1],t.dom.v[2],getindex.(t.cod,1),getindex.(t.cod,2);args...)
        Makie.arrows(t::VectorField{<:Chain,<:Hyperrectangle};args...) = Makie.arrows(Makie.Point.(domain(t))[:],Makie.Point.(codomain(t))[:];args...)
        Makie.arrows!(t::VectorField{<:Chain,<:Hyperrectangle};args...) = Makie.arrows!(Makie.Point.(domain(t))[:],Makie.Point.(codomain(t))[:];args...)
        Makie.arrows(t::Rectangle,f::Function;args...) = Makie.arrows(t.v[1],t.v[2],f;args...)
        Makie.arrows!(t::Rectangle,f::Function;args...) = Makie.arrows!(t.v[1],t.v[2],f;args...)
        Makie.arrows(t::Hyperrectangle,f::Function;args...) = Makie.arrows(t.v[1],t.v[2],t.v[3],f;args...)
        Makie.arrows!(t::Hyperrectangle,f::Function;args...) = Makie.arrows!(t.v[1],t.v[2],t.v[3],f;args...)
        Makie.arrows(t::MeshFunction;args...) = Makie.arrows(points(domain(t)),codomain(t);args...)
        Makie.arrows!(t::MeshFunction;args...) = Makie.arrows!(points(domain(t)),codomain(t);args...)
        Makie.mesh(t::ElementFunction;args...) = Makie.mesh(domain(t);color=codomain(t),args...)
        Makie.mesh!(t::ElementFunction;args...) = Makie.mesh!(domain(t);color=codomain(t),args...)
        Makie.mesh(t::MeshFunction;args...) = Makie.mesh(domain(t);color=codomain(t),args...)
        Makie.mesh!(t::MeshFunction;args...) = Makie.mesh!(domain(t);color=codomain(t),args...)
        Makie.wireframe(t::ElementFunction;args...) = Makie.wireframe(value(domain(t));color=codomain(t),args...)
        Makie.wireframe!(t::ElementFunction;args...) = Makie.wireframe!(value(domain(t));color=codomain(t),args...)
    end
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" begin
        UnicodePlots.lineplot(t::PlaneCurve;args...) = UnicodePlots.lineplot(getindex.(t.cod,1),getindex.(t.cod,2);args...)
        UnicodePlots.lineplot(t::RealFunction;args...) = UnicodePlots.lineplot(t.dom,getindex.(value.(t.cod),1);args...)
        UnicodePlots.contourplot(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.contourplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.surfaceplot(t.dom.v[1][2:end-1],t.dom.v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.spy(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.spy(t.cod;args...)
        UnicodePlots.heatmap(t::GridSurface{<:Chain,<:RealRegion};args...) = UnicodePlots.heatmap(t.cod;args...)
    end
end

end # module
