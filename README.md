<p align="center">
  <img src="./docs/src/assets/logo.png" alt="DirectSum.jl"/>
</p>

# Adapode.jl

*Adaptive multistep numerical ODE solver based on [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) element assembly*

[![DOI](https://zenodo.org/badge/223493781.svg)](https://zenodo.org/badge/latestdoi/223493781)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://grassmann.crucialflow.com/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://grassmann.crucialflow.com/dev)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build Status](https://travis-ci.org/chakravala/Adapode.jl.svg?branch=master)](https://travis-ci.org/chakravala/Adapode.jl)

This Julia project originally started as a FORTRAN 95 project called [adapode](https://github.com/chakravala/adapode) and evolved with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) and [Cartan.jl](https://github.com/chakravala/Cartan.jl).

```julia
using Grassmann, Adapode, Makie
x0 = Chain(10.0,10.0,10.0)
Lorenz(σ,r,b) = x -> Chain(
	σ*(x[2]-x[1]),
	x[1]*(r-x[3])-x[2],
	x[1]*x[2]-b*x[3])
lines(odesolve(Lorenz(10.0,28.0,8/3),x0))
```
Supported ODE solvers include:
explicit Euler,
Heun's method (improved Euler),
Midpoint 2nd order RK,
Kutta's 3rd order RK,
classical 4th order Runge-Kuta,
adaptive Heun-Euler,
adaptive Bogacki-Shampine RK23,
adaptive Fehlberg RK45,
adaptive Cash-Karp RK45,
adaptive Dormand-Prince RK45,
multistep Adams-Bashforth-Moulton 2nd,3rd,4th,5th order,
adaptive multistep ABM 2nd,3rd,4th,5th order.

It is possible to work with L2 projection on a mesh with
```julia
L2Projector(t,f;args...) = mesh(t,color=\(assemblemassfunction(t,f)...);args...)
L2Projector(initmesh(0:1/5:1)[3],x->x[2]*sin(x[2]))
L2Projector(initmesh(0:1/5:1)[3],x->2x[2]*sin(2π*x[2])+3)
```

Partial differential equations can also be assembled with various additional methods:
```julia
function solvepoisson(t,e,c,f,κ,gD=0,gN=0)
    m = volumes(t)
    b = assemblefunction(t,f,m)
    A = assemblestiffness(t,c,m)
    R,r = assemblerobin(e,κ,gD,gN)
    return TensorField(t,(A+R)\(b+r))
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
function solvetransport(t,e,c,ϵ=0.1)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,ϵ,m,g)
    b = assembleload(t,m)
    C = assembleconvection(t,c,m,g)
    return TensorField(t,solvedirichlet(A+C,b,e))
end
```
Such modular methods can work with a `TensorField` of any dimension.
The following examples are based on trivially generated 1 dimensional domains:
```Julia
function BackwardEulerHeat1D()
    x,m = 0:0.01:1,100; p,e,t = initmesh(x)
    T = range(0,0.5,length=m+1) # time grid
    ξ = 0.5.-abs.(0.5.-x) # initial condition
    A = assemblestiffness(p(t)) # assemble(p(t),1,2x)
    M,b = assemblemassfunction(p(t),2x).+assemblerobin(p(e),1e6,0,0)
    h = Float64(T.step); LHS = M+h*A # time step
    for l ∈ 1:m
        ξ = LHS\(M*ξ+h*b); l%10==0 && println(l*h)
    end
    lines(TensorField(p(t),ξ))
end
function PoissonAdaptive(g,p,e,t,c=1,a=0,f=1)
    ϵ = 1.0
    pt,pe = p(t),p(e)
    while ϵ > 5e-5 && length(t) < 10000
        m = volumes(pt)
        h = gradienthat(pt,m)
        A,M,b = assemble(pt,c,a,f,m,h)
        ξ = solvedirichlet(A+M,b,pe)
        η = jumps(pt,c,a,f,ξ,m,h)
        display(lines(TensorField(pt,ξ)))
        if typeof(g)<:AbstractRange
            #scatter!(p,ξ,markersize=0.01)
        else
            #wireframe!(t,color=(:red,0.6),linewidth=3)
        end
        ϵ = sqrt(norm(η)^2/length(η))
        println(t,", ϵ=$ϵ, α=$(ϵ/maximum(η))"); sleep(0.5)
        refinemesh!(g,p,e,t,select(η,ϵ),"regular")
    end
    return g,p,e,t
end
PoissonAdaptive(refinemesh(0:0.25:1)...,1,0,x->exp(-100abs2(x[2]-0.5)))
```
More general problems for finite element boundary value problems are also enabled by mesh representations imported from external sources and managed by `Cartan` via `Grassmann` algebra.
These methods can automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.
