module Adapode

#   This file is part of Aadapode.jl. It is licensed under the GPL license
#   Adapode Copyright (C) 2019 Michael Reed

using StaticArrays, SparseArrays
using AbstractTensors, DirectSum, Grassmann
import Grassmann: value, vector, valuetype
import Base: @pure

export SVector, odesolve
export initmesh, pdegrad

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
    r::Bool
    function TimeStep(h,hmin=1e-16,hmax=1e-4,emin=10^(log2(h)-3),emax=10^log2(h))
        checkhsize!(new{typeof(h)}(h,hmin,hmax,emin,emax,(emin+emax)/2,1,0,true))
    end
end
function checkhsize!(t)
    abs(t.h) < t.hmin && (t.h = copysign(t.hmin,t.h))
    abs(t.h) > t.hmax && (t.h = copysign(t.hmax,t.h))
    t.i < 1 && (t.i = 1)
    return t
end

@pure shift(::Val{m},::Val{k}=Val{1}()) where {m,k} = SVector{m,Int}(k:m+k-1)
@pure shift(M::Val{m},i) where {m,a} = ((shift(M,Val{0}()).+i).%(m+1)).+1
function explicit(x,h,c,fx)
    l=length(c)
    x+h*c⋅(l≠length(fx) ? fx[shift(Val{l}())] : fx)
end
function explicit(x,h,c,fx,i)
    l=length(c)
    h*c⋅(l≠length(fx) ? fx[shift(Val{l}(),i)] : fx)
end
improved_heun(x,f,h) = (fx = f(x); x+(h/2)*(fx+f(x+h*fx)))

@pure butcher(::Val{N},::Val{A}=Val{false}()) where {N,A} = A ? CBA[N] : CB[N]
@pure blength(n::Val{N},a::Val{A}=Val{false}()) where {N,A} = Val{length(butcher(n,a))-A}()
function butcher(x,f,h,v::Val{N}=Val{4}(),a::Val{A}=Val{false}()) where {N,A}
    b = butcher(v,a)
    n = length(b)-A
    fx = MVector{n,typeof(x)}(undef)
    fx[1] = f(x)
    for k ∈ 2:n
        fx[k] = f(explicit(x,h,b[k-1],fx))
    end
    return fx
end
explicit(x,f,h,b::Val=Val{4}()) = explicit(x,h,butcher(b)[end],butcher(x,f,h,b))
explicit(x,f,h,::Val{1}) = x+h*f(x)

function multistep!(x,f,h,fx,i,K::Val{k}=Val{4}(),::Val{PC}=Val{false}()) where {k,PC}
    fx[((i+k-1)%(k+1))+1] = f(x)
    explicit(x,h,PC ? CAM[k] : CAB[k],fx,i)
end
function predictcorrect(x,f,t,fx,k::Val{m}=Val{4}()) where m
    p = x + multistep!(x,f,t.h,fx,t.s,k)
    t.s = (t.s%(m+1))+1
    x + multistep!(p,f,t.h,fx,t.s,k,Val{true}())
end
predictcorrect(x,f,h,fx,::Val{1}) = x+h*f(x+h*f(x))

function initsteps(x0,t,tmin,tmax)
    x = Vector{typeof(x0)}(undef,Int(round((tmax-tmin)*2^-log2(t.h)))+1)
    x[1] = x0
    return x
end
function initsteps(x0,t,tmin,tmax,f,m::Val{o},B=Val{4}()) where o
    x,fx = initsteps(x0,t,tmin,tmax), MVector{o+1,typeof(x0)}(undef)
    t.r && initsteps!(x,f,fx,t,B)
    return x, fx
end

function initsteps!(x,f,fx,t,B=Val{4}())
    m = length(fx)-2
    xi = x[t.i]
    for j ∈ 1:m
        fx[j] = f(xi)
        xi = explicit(xi,f,t.h,B)
        x[t.i+j] = xi
    end
    t.s = 0
    t.r = false
    t.i += m
end

function explicit!(x,f,t,B=Val{5}())
    resize_array!(x,t.i,10000)
    xi = x[t.i]
    fx,b = butcher(xi,f,t.h,B,Val{true}()),butcher(B,Val{true}())
    t.i += 1
    x[t.i] = explicit(xi,t.h,b[end-1],fx)
    t.e = maximum(abs.(t.h*value(b[end]⋅fx)))
end

function predictcorrect!(x,f,fx,t,B::Val{m}=Val{4}()) where m
    resize_array!(x,t.i+m,10000)
    t.r && initsteps!(x,f,fx,t)
    xi = x[t.i]
    p = xi + multistep!(xi,f,t.h,fx,t.s,B)
    t.s = (t.s%(m+1))+1
    c = xi + multistep!(p,f,t.h,fx,t.s,B,Val{true}())
    t.i += 1
    x[t.i] = c
    t.e = maximum(abs.(value(c-p)./value(c)))
end

odesolve(f,x0,∂,tol,m,o) = odesolve(f,x0,∂,tol,Val{m}(),Val{o}())
function odesolve(f,x0,tmin=0,tmax=2π,tol=15,::Val{m}=Val{1}(),B::Val{o}=Val{4}()) where {m,o}
    itmax = 5^(tol+5)
    t = TimeStep(2.0^-tol)
    if m == 0 # Improved Euler, Heun's Method
        x = initsteps(x0,t,tmin,tmax)
        for i ∈ 2:length(x)
            x[i] = improved_heun(x[i-1],f,t.h)
        end
    elseif m == 1 # Singlestep
        x = initsteps(x0,t,tmin,tmax)
        for i ∈ 2:length(x)
            x[i] = explicit(x[i-1],f,t.h,B)
        end
    elseif m == 2 # Adaptive Singlestep
        x = initsteps(x0,t,tmin,tmax)
        while timeloop!(x,t,tmax)
            explicit!(x,f,t,B)
        end
    elseif m == 3 # Multistep
        x,fx = initsteps(x0,t,tmin,tmax,f,B)
        for i ∈ o+1:length(x)
            x[i] = predictcorrect(x[i-1],f,t,fx,B)
        end
    else # Adaptive Multistep
        x,fx = initsteps(x0,t,tmin,tmax,f,B)
        while timeloop!(x,t,tmax,B)
            predictcorrect!(x,f,fx,t,B)
        end
    end
    return x
end

function timeloop!(x,t,tmax,::Val{m}=Val{1}()) where m
    if t.e < t.emin
        t.h *= 2
        t.i -= Int(floor(m/2))
        t.r = true
    end
    if t.e > t.emax
        t.h /= 2
        t.i -= Int(ceil(m/2))
        t.r = true
    end
    t.r && checkhsize!(t)
    d = tmax-x[t.i][1]
    if d ≤ t.h
        println("final step: $(t.i)")
        t.h = d
    end
    done = d ≤ t.hmax
    done && truncate_array!(x,t.i-1)
    return !done
end

resize_array!(x,i,h) = length(x)<i+1 && resize!(x,i+h)
truncate_array!(x,i) = length(x)>i+1 && resize!(x,i)

show_progress(x,t,b) = t.i%75000 == 11 && println(x[t.i][1]," out of ",b)

end # module
