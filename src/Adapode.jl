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
import Base: @pure

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
    hfx = h*f(x)
    fiber(x)+(hfx+h*f((point(x)+h)↦(fiber(x)+hfx)))/2
end
function heun(x,f::TensorField,t)
    fx = f[t.i]
    hfx = t.h*fx
    fiber(x)+(hfx+t.h*f(point(fx)↦(fiber(x)+hfx)))/2
end

@pure butcher(::Val{N},::Val{A}=Val(false)) where {N,A} = A ? CBA[N] : CB[N]
@pure blength(n::Val{N},a::Val{A}=Val(false)) where {N,A} = Val(length(butcher(n,a))-A)
function butcher(x::Section{B,F},f,h,v::Val{N}=Val(4),a::Val{A}=Val(false)) where {N,A,B,F}
    b = butcher(v,a)
    n = length(b)-A
    fx = F<:Vector ? FixedVector{n,F}(undef) : Variables{n,F}(undef)
    @inbounds fx[1] = f(x)
    for k ∈ 2:n
        @inbounds fx[k] = f((point(x)+h*sum(b[k-1]))↦explicit(x,h,b[k-1],fx))
    end
    return fx
end
explicit(x,f::Function,h,b::Val=Val(4)) = explicit(x,h,butcher(b)[end],butcher(x,f,h,b))
explicit(x,f::Function,h,::Val{1}) = fiber(x)+h*f(x)

function multistep!(x,f,fx,t,::Val{k}=Val(4),::Val{PC}=Val(false)) where {k,PC}
    fx[t.s] = f(x)
    explicit(x,t.h,PC ? CAM[k] : CAB[k],fx,t.s)
end # more accurate compared with CAB[k] methods
function predictcorrect(x,f,fx,t,k::Val{m}=Val(4)) where m
    iszero(t.s) && initsteps!(x,f,fx,t)
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),point(xti)+t.h
    xn = multistep!(xti,f,fx,t,k)
    t.s = (t.s%(m+1))+1; t.i += 1
    xn = multistep!(tn↦(xi+xn),f,fx,t,k,Val(true))
    return xi + xn
end
function predictcorrect(x,f,fx,t,::Val{1})
    @inbounds xti = x[t.i]
    t.i += 1
    fiber(xti)+t.h*f((point(xti)+t.h)↦(fiber(xti)+t.h*f(xti)))
end

initsteps(x0,t,tmax,m) = initsteps(init(x0),t,tmax,m)
initsteps(x0,t,tmax,f,m,B) = initsteps(init(x0),t,tmax,f,m,B)
function initsteps(x0::Section,t,tmax,::Val{m}) where m
    tmin = base(x0)
    n = Int(round((tmax-tmin)*2^-log2(t.h)))+1
    t = m ∈ (0,1,3) ? (tmin:t.h:tmax) : Vector{typeof(t.h)}(undef,n)
    m ∉ (0,1,3) && (t[1] = tmin)
    x = Vector{fibertype(x0)}(undef,m ∈ (0,1,3) ? length(t) : n)
    x[1] = fiber(x0)
    return TensorField(t,x)
end
function initsteps(x0::Section,t,tmax,f,m,B::Val{o}=Val(4)) where o
    initsteps(x0,t,tmax,m), Variables{o+1,fibertype(x0)}(undef)
end

function multistep2!(x,f,fx,t,::Val{k}=Val(4),::Val{PC}=Val(false)) where {k,PC}
    @inbounds fx[t.s] = f(x)
    @inbounds explicit(x,t.h,PC ? CAM[k] : CAB[k-1],fx,t.s)
end
function predictcorrect2(x,f,fx,t,k::Val{m}=Val(4)) where m
    iszero(t.s) && initsteps!(x,f,fx,t)
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),point(xti)+t.h
    xn = multistep2!(xti,f,fx,t,k)
    t.s = (t.s%m)+1; t.i += 1
    xn = multistep2!(tn↦(xi+xn),f,fx,t,k,Val(true))
    t.s = (t.s%m)+1
    return xi + xn
end
function predictcorrect2(x,f,fx,t,::Val{1})
    @inbounds xti = x[t.i]
    t.i += 1
    fiber(xti)+t.h*f((point(xti)+t.h)↦xti)
end

initsteps2(x0,t,tmax,f,m,B) = initsteps2(init(x0),t,tmax,f,m,B)
function initsteps2(x0::Section,t,tmax,f,m,B::Val{o}=Val(4)) where o
    initsteps(x0,t,tmax,m), Variables{o,fibertype(x0)}(undef)
end

function initsteps!(x,f,fx,t,B=Val(4))
    m = length(fx)-2
    @inbounds xi = x[t.i]
    for j ∈ 1:m
        @inbounds fx[j] = f(xi)
        xi = (point(xi)+t.h) ↦ explicit(xi,f,t.h,B)
        x[t.i+j] = xi
    end
    t.s = 1+m
    t.i += m
end

function explicit!(x,f,t,B=Val(5))
    resize!(x,t.i,10000)
    @inbounds xti = x[t.i]
    xi = fiber(xti)
    fx,b = butcher(xti,f,t.h,B,Val(true)),butcher(B,Val(true))
    t.i += 1
    @inbounds x[t.i] = (point(xti)+t.h) ↦ explicit(xti,t.h,b[end-1],fx)
    @inbounds t.e = maximum(abs.(t.h*value(b[end]⋅fx)))
end

function predictcorrect!(x,f,fx,t,B::Val{m}=Val(4)) where m
    resize!(x,t.i+m,10000)
    iszero(t.s) && initsteps!(x,f,fx,t)
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),point(xti)+t.h
    p = xi + multistep!(xti,f,fx,t,B)
    t.s = (t.s%(m+1))+1
    c = xi + multistep!(tn↦p,f,fx,t,B,Val(true))
    t.i += 1
    @inbounds x[t.i] = tn ↦ c
    t.e = maximum(abs.(value(c-p)./value(c)))
end
function predictcorrect!(x,f,fx,t,::Val{1})
    @inbounds xti = x[t.i]
    xi,tn = fiber(xti),point(xti)+t.h
    p = xi + t.h*f(xti)
    c = xi + t.h*f(tn↦p)
    t.i += 1
    resize!(x,t.i,10000)
    @inbounds x[t.i] = tn ↦ c
    t.e = maximum(abs.(value(c-p)./value(c)))
end

init(x0) = 0.0 ↦ x0
init(x0::Section) = x0

odesolve(f,x0,tmax,tol,m,o) = odesolve(f,x0,tmax,tol,Val(m),Val(o))
function odesolve(f,x0,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    t = TimeStep(2.0^-tol)
    if m == 0 # Improved Euler, Heun's Method
        x = initsteps(x0,t,tmax,M)
        for i ∈ 2:length(x)
            @inbounds x[i] = heun(x[i-1],f,t.h)
        end
    elseif m == 1 # Singlestep
        x = initsteps(x0,t,tmax,M)
        for i ∈ 2:length(x)
            @inbounds x[i] = explicit(x[i-1],f,t.h,B)
        end
    elseif m == 2 # Adaptive Singlestep
        x = initsteps(x0,t,tmax,M)
        while timeloop!(x,t,tmax)
            explicit!(x,f,t,B)
        end
    elseif m == 3 # Multistep
        x,fx = initsteps(x0,t,tmax,f,M,B)
        for i ∈ o+1:length(x) # o+1 changed to o
            @inbounds x[i] = predictcorrect(x,f,fx,t,B)
        end
    else # Adaptive Multistep
        x,fx = initsteps(x0,t,tmax,f,M,B)
        while timeloop!(x,t,tmax,B) # o+1 fix??
            predictcorrect!(x,f,fx,t,B)
        end
    end
    return x
end

#integrange
#integadapt

odesolve2(f,x0,tmax,tol,m,o) = odesolve2(f,x0,tmax,tol,Val(m),Val(o))
function odesolve2(f,x0,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    t = TimeStep(2.0^-tol)
    if m == 1 # Singlestep
        x = initsteps(x0,t,tmax,M)
        for i ∈ 2:length(x)
            @inbounds x[i] = explicit(x[i-1],f,t.h,B)
        end
    elseif m == 2 # Adaptive Singlestep
        x = initsteps(x0,t,tmax,M)
        while timeloop!(x,t,tmax)
            explicit!(x,f,t,B)
        end
    elseif m == 3 # Multistep
        x,fx = initsteps2(x0,t,tmax,f,M,B)
        for i ∈ (isone(o) ? 2 : o):length(x) # o+1 changed to o
            @inbounds x[i] = predictcorrect2(x,f,fx,t,B)
        end
    else # Adaptive Multistep
        x,fx = initsteps(x0,t,tmax,f,M,B) # o+1 fix?
        while timeloop!(x,t,tmax,B)
            predictcorrect!(x,f,fx,t,B)
        end
    end
    return x
end

function integrate(f::TensorField,x,tmax=2π,tol=15,M::Val{m}=Val(1),B::Val{o}=Val(4)) where {m,o}
    x0,t = init(x),TimeStep(2.0^-tol)
    if m == 0 # Improved Euler, Heun's Method
        x = initsteps(x0,t,tmax,M)
        for i ∈ 2:length(x)
            @inbounds x[i] = heun(x[i-1],f,t.h)
        end
    elseif m == 3 # Multistep
        x,fx = initsteps(x0,t,tmax,f,M,B)
        for i ∈ o+1:length(x)
            @inbounds x[i] = predictcorrect(x,f,fx,t,B)
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
    d = tmax-point(x[t.i])
    d ≤ t.h && (t.h = d)
    done = d ≤ t.hmax
    done && truncate!(x,t.i-1)
    return !done
end

time(x) = x[1]

Base.resize!(x,i,h) = length(x)<i+1 && resize!(x,i+h)
truncate!(x,i) = length(x)>i+1 && resize!(x,i)

show_progress(x,t,b) = t.i%75000 == 11 && println(point(x[t.i])," out of ",b)

end # module
