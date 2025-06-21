<p align="center">
  <img src="./docs/src/assets/logo.png" alt="DirectSum.jl"/>
</p>

# Adapode.jl

*Adaptive P/ODE numerics with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) element TensorField assembly*

[![DOI](https://zenodo.org/badge/223493781.svg)](https://zenodo.org/badge/latestdoi/223493781)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cartan.crucialflow.com)
[![PDF 2021](https://img.shields.io/badge/PDF-2021-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=differential-geometric-algebra-2021.pdf)
[![PDF 2025](https://img.shields.io/badge/PDF-2025-blue.svg)](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf)
[![Gitter](https://badges.gitter.im/Grassmann-jl/community.svg)](https://gitter.im/Grassmann-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build Status](https://travis-ci.org/chakravala/Adapode.jl.svg?branch=master)](https://travis-ci.org/chakravala/Adapode.jl)

This project originally started as a FORTRAN 95 project called [adapode](https://github.com/chakravala/adapode) and evolved with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) and [Cartan.jl](https://github.com/chakravala/Cartan.jl).

*Cartan.jl* introduces a pioneering unified numerical framework for comprehensive differential geometric algebra, purpose-built for the formulation and solution of partial differential equations on manifolds with non-trivial topological structure and [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) algebra.
Written in Julia, [Cartan.jl](https://github.com/chakravala/Cartan.jl) unifies differential geometry, geometric algebra, and tensor calculus with support for fiber product topology; enabling directly executable generalized treatment of geometric PDEs over grids, meshes, and simplicial decompositions.

The system supports intrinsic formulations of differential operators (including the exterior derivative, codifferential, Lie derivative, interior product, and Hodge star) using a coordinate-free algebraic language grounded in Grassmann-Cartan multivector theory.
Its core architecture accomodates numerical representations of fiber bundles, product manifolds, and submanifold immersion, providing native support for PDE models defined on structured or unstructured domains.

*Cartan.jl* integrates naturally with simplex-based finite element exterior calculus, allowing for geometrical discretizations of field theories and conservation laws.
With its synthesis of symbolic abstraction and numerical execution, *Cartan.jl* empowers researchers to develop PDE models that are simultaneously founded in differential geometry, algebraically consistent, and computationally expressive, opening new directions for scientific computing at the interface of geometry, algebra, and analysis.

```
   _______     __                          __
  |   _   |.--|  |.---.-..-----..-----..--|  |.-----.
  |       ||  _  ||  _  ||  _  ||  _  ||  _  ||  -__|
  |___|___||_____||___._||   __||_____||_____||_____|
                         |__|
```
developed by [chakravala](https://github.com/chakravala) with [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) and [Cartan.jl](https://github.com/chakravala/Cartan.jl)

For `GridBundle` initialization it is typical to invoke a combination of `ProductSpace` and `QuotientTopology`, while optional Julia packages extend `SimplexBundle` initialization, such as
[Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl),
[GeometryBasics.jl](https://github.com/JuliaGeometry/GeometryBasics.jl),
[Delaunay.jl](https://github.com/eschnett/Delaunay.jl),
[QHull.jl](https://github.com/JuliaPolyhedra/QHull.jl),
[MiniQhull.jl](https://github.com/gridap/MiniQhull.jl),
[Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl),
[TetGen.jl](https://github.com/JuliaGeometry/TetGen.jl),
[MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl),
[FlowGeometry.jl](https://github.com/chakravala/FlowGeometry.jl).

## Ordinary differential equations

*Example* (Lorenz). Observe vector fields by integrating streamlines
```julia
using Grassmann, Cartan, Adapode, Makie # GLMakie
Lorenz(s,r,b) = x -> Chain(
    s*(x[2]-x[1]), x[1]*(r-x[3])-x[2], x[1]*x[2]-b*x[3])
p = TensorField(ProductSpace(-40:0.2:40,-40:0.2:40,10:0.2:90))
vf = Lorenz(10.0,60.0,8/3).(p) # pick Lorenz parameters, apply
streamplot(vf,gridsize=(10,10)) # visualize vector field
```
ODE solvers in the `Adapode` package are built on `Cartan`, providing both Runge-Kutta and multistep methods with optional adaptive time stepping.
```julia
fn,x0 = Lorenz(10.0,28.0,8/3),Chain(10.0,10.0,10.0)
ic = InitialCondition(fn,x0,2pi) # tmax = 2pi
lines(odesolve(ic,MultistepIntegrator{4}(2^-15)))
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

When there is a Levi-Civita connection with zero torsion related to a `metrictensor`, then there exist Christoffel symbols of the `secondkind`.
In particular, these can be expressed in terms of the `metrictensor` `g` to express local geodesic differential equations for Riemannian geometry.

*Example*. `using Grassmann, Cartan, Adapode, Makie # GLMakie`
```julia
torus(x) = Chain(
    (2+0.5cos(x[1]))*cos(x[2]),
    (2+0.5cos(x[1]))*sin(x[2]),
    0.5sin(x[1]))
tor = torus.(TorusParameter(60,60))
tormet = surfacemetric(tor) # intrinsic metric
torcoef = secondkind(tormet) # Christoffel symbols
ic = geodesic(torcoef,Chain(1.0,1.0),Chain(1.0,sqrt(2)),10pi)
sol = geosolve(ic,ExplicitIntegrator{4}(2^-7)) # Runge-Kutta
lines(torus.(sol))
```
```julia
totalarclength(sol) # apparent length of parameter path
@basis MetricTensor([1 1; 1 1]) # abstract non-Euclidean V
solm = TensorField(tormet(sol),Chain{V}.(value.(fiber(sol))))
totalarclength(solm) # 2D estimate totalarclength(torus.(sol))
totalarclength(torus.(sol)) # compute in 3D Euclidean metric
lines(solm) # parametric curve can have non-Euclidean metric
lines(arclength(solm)); lines!(arclength(sol))
```

*Example* (Klein geodesic). General `ImmersedTopology` are supported
```julia
klein(x) = klein(x[1],x[2]/2)
function klein(v,u)
    x = cos(u)*(-2/15)*(3cos(v)-30sin(u)+90sin(u)*cos(u)^4-
        60sin(u)*cos(u)^6+5cos(u)*cos(v)*sin(u))
    y = sin(u)*(-1/15)*(3cos(v)-3cos(v)*cos(u)^2-
        48cos(v)*cos(u)^4+48cos(v)*cos(u)^6-
        60sin(u)+5cos(u)*cos(v)*sin(u)-
        5cos(v)*sin(u)*cos(u)^3-80cos(v)*sin(u)*cos(u)^5+
        80cos(v)*sin(u)*cos(u)^7)
    z = sin(v)*(2/15)*(3+5cos(u)*sin(u))
    Chain(x,y,z)
end # too long paths over QuotientTopology can stack overflow
kle = klein.(KleinParameter(100,100))
klecoef = secondkind(surfacemetric(kle))
ic = geodesic(klecoef,Chain(1.0,1.0),Chain(1.0,2.0),2pi)
lines(geosolve(ic,ExplicitIntegrator{4}(2^-7)));wireframe(kle)
```

*Example* (Upper half plane). Intrinsic hyperbolic Lobachevsky metric
```julia
halfplane(x) = TensorOperator(Chain(
    Chain(Chain(0.0,inv(x[2])),Chain(-inv(x[2]),0.0)),
    Chain(Chain(-inv(x[2]),0.0),Chain(0.0,-inv(x[2])))))
z1 = geosolve(halfplane,Chain(1.0,1.0),Chain(1.0,2.0),10pi,7)
z2 = geosolve(halfplane,Chain(1.0,0.1),Chain(1.0,2.0),10pi,7)
z3 = geosolve(halfplane,Chain(1.0,0.5),Chain(1.0,2.0),10pi,7)
z4 = geosolve(halfplane,Chain(1.0,1.0),Chain(1.0,1.0),10pi,7)
z5 = geosolve(halfplane,Chain(1.0,1.0),Chain(1.0,1.5),10pi,7)
lines(z1); lines!(z2); lines!(z3); lines!(z4); lines!(z5)
```

*Example* (da Rios). The `Cartan` abstractions enable easily integrating
```julia
start(x) = Chain(cos(x),sin(x),cos(1.5x)*sin(1.5x)/5)
x1 = start.(TorusParameter(180));
darios(t,dt=tangent(fiber(t))) = hodge(wedge(dt,tangent(dt)))
sol = odesolve(darios,x1,1.0,2^-11)
mesh(sol,normalnorm)
```

## Finite element methods

*Example* (L2 Projector). It is possible to work with L2 projection on a mesh with
```julia
L2Projector(t,f;args...) = mesh(t,color=\(assemblemassload(t,f)...);args...)
L2Projector(initmesh(0:1/5:1)[3],x->x[2]*sin(x[2]))
L2Projector(initmesh(0:1/5:1)[3],x->2x[2]*sin(2π*x[2])+3)
```

*Example* (Eigenmodes of disk). Enabled with `assemble` for stiffness and mass matrix from `Adapode`:
```julia
using Grassmann, Cartan, Adapode, MATLAB, Makie # GLMakie
pt,pe = initmesh("circleg","hmax"=>0.1) # MATLAB circleg mesh
A,M = assemble(pt,1,1,0) # stiffness & mass matrix assembly
using KrylovKit # provides general eigsolve
yi,xi = geneigsolve((A,M),10,:SR;krylovdim=100) # maxiter=100
amp = TensorField.(Ref(pt),xi./3) # solutions amplitude
mode = TensorField.(graphbundle.(amp),xi) # make 3D surface
mesh(mode[7]); wireframe!(pt) # figure modes are 4,5,7,8,6,9
```

To build on the `FiberBundle` functionality of `Cartan`, the numerical analysis package `Adapode` is being developed to provide extra utilities for finite element method assemblies.
Poisson equation syntax or transport equations with finite element methods can be expressed in terms of methods like `volumes` using exterior products or `gradienthat` by applying the exterior algebra principles discussed.
Global `Grassmann` element assembly problems involve applying geometric algebra locally per element basis and combining it with a global manifold topology.

```julia
function solvepoisson(t,e,c,f,κ,gD=0,gN=0)
    m = volumes(t)
    b = assembleload(t,f,m)
    A = assemblestiffness(t,c,m)
    R,r = assemblerobin(e,κ,gD,gN)
    return TensorField(t,(A+R)\(b+r))
end
```
```Julia
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
```
```Julia
function solvetransport(t,e,c,f=1,ϵ=0.1)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,ϵ,m,g)
    b = assembleload(t,f,m)
    C = assembleconvection(t,c,m,g)
    return TensorField(t,solvedirichlet(A+C,b,e))
end
```
More general problems for finite element boundary value problems are also enabled by mesh representations imported into `Cartan` from external sources and computationally operated on in terms of `Grassmann` algebra.
Many of these methods automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.
Further advanced features such as `DiscontinuousTopology` have been implemented and the `LagrangeTopology` variant of `SimplexTopology` is being used in research.
```Julia
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
```
```Julia
pt,pe = initmesh(0:0.01:1)
triangle(x) = 0.5-abs(0.5-x[2])
bt = solveheat(triangle.(pt),(x->2x[2]).(pt),(x->1e6).(pe),range(0,0.6,101))
```
```Julia
function adaptpoisson(g,pt,pe,c=1,a=0,f=1,κ=1e6,gD=0,gN=0)
    ϵ = 1.0
    while ϵ > 5e-5 && elements(pt) < 10000
        m = volumes(pt)
        h = gradienthat(pt,m)
        A,M,b = assemble(pt,c,a,f,m,h)
        ξ = solvedirichlet(A+M,b,immersion(pe))
        η = jumps(pt,c,a,f,ξ,m,h)
        ϵ = rms(η)
        println("ϵ=$ϵ, α=$(ϵ/maximum(η))")
        refinemesh!(g,pt,pe,select(η,ϵ),"regular")
    end
    return g,pt,pe
end
```
```Julia
adaptpoisson(refinemesh(0:0.25:1)...,1,0,x->exp(-100abs2(x[2]-0.5)))

```

*Example* (Heatflow around airfoil).
`FlowGeometry` builds on `Cartan` to provide NACA airfoil shapes, and `Adapode` can solve transport diffusion.
```julia
using Grassmann, Cartan, Adapode, FlowGeometry, MATLAB, Makie # GLMakie
pt,pe = initmesh(decsg(NACA"6511"),"hmax"=>0.1)
tf = solvepoisson(pt,pe,1,0,
    x->(x[2]>3.49 ? 1e6 : 0.0),0,x->(x[2]<-1.49 ? 1.0 : 0.0))
function kappa(z); x = base(z)
    if x[2]<-1.49 || sqrt((x[2]-0.5)^2+x[3]^2)<0.51
        1e6
    else
        x[2]>3.49 ? fiber(z)[1] : 0.0
    end
end
gtf = -gradient(tf)
kf = kappa.(gtf(immersion(pe)))
tf2 = solvetransportdiffusion(gtf,kf,0.01,1/50,
    x->(sqrt((x[2]-0.5)^2+x[3]^2)<0.7 ? 1.0 : 0.0))
wireframe(pt)
streamplot(gtf,-0.3..1.3,-0.2..0.2)
mesh(tf2)
```

*Example*. Most finite element methods `using Grassmann, Cartan` automatically generalize to higher dimension manifolds with e.g. tetrahedra,
and the author has contributed to packages such as *Triangulate.jl*, *TetGen.jl*.
```julia
using Grassmann, Cartan, Adapode, FlowGeometry, MiniQhull, TetGen, Makie # GLMakie
ps = sphere(sphere(∂(delaunay(PointCloud(sphere())))))
pt,pe = tetrahedralize(cubesphere(),"vpq1.414a0.1";
    holes=[TetGen.Point(0.0,0.0,0.0)])
tf = solvepoisson(pt,pe,1,0,
    x->(x[2]>1.99 ? 1e6 : 0.0),0,x->(x[2]<-1.99 ? 1.0 : 0.0))
streamplot(-gradient(tf),-1.1..1.1,-1.1..1.1,-1.1..1.1,
    gridsize=(10,10,10))
wireframe!(ps)
```

## References

* Michael Reed, [Foundatons of differential geometric algebra](https://vixra.org/abs/2304.0228) (2021)
* Michael Reed, [Differential geometric algebra: compute using Grassmann.jl and Cartan.jl](https://www.dropbox.com/sh/tphh6anw0qwija4/AAACiaXig5djrLVAKLPFmGV-a/Geometric-Algebra?preview=grassmann-cartan-2025.pdf) (2025)

