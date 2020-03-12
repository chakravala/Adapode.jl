# Adapode.jl

Adaptive multistep numerical ODE solver based on [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) element assembly.

This Julia project originally started as a FORTRAN 95 project called [adapode](https://github.com/chakravala/adapode).

```julia
using Grassmann, Adapode, Makie
basis"4"; x0 = 10.0v2+10.0v3+10.0v4
Lorenz(x::Chain{V}) where V = Chain{V,1}(SVector{4,Float64}(
	1.0,
	10.0(x[3]-x[2]),
	x[2]*(28.0-x[4])-x[3],
	x[2]*x[3]-(8/3)*x[4]))
lines(Point.((V(2,3,4)).(odesolve(Lorenz,x0))))
```

Partial differential equations can also be assembled with various additional methods:
```julia
PoissonSolver(p,e,t,c,f,κ,gD=1,gN=0) = mesh(t,color=solvepoisson(t,e,c,f,κ,gD,gN))
function solvepoisson(t,e,c,f,κ,gD=0,gN=0)
    m = detsimplex(t)
    b = assemblefunction(t,f,m)
    A = assemblestiffness(t,c,m)
    R,r = assemblerobin(e,κ,gD,gN)
    return (A+R)\(b+r)
end
function BackwardEulerHeat1D()
    x,m = 0:0.01:1,100; p,e,t = initmesh(x)
    T = range(0,0.5,length=m+1) # time grid
    ξ = 0.5.-abs.(0.5.-x) # initial condition
    A = assemblestiffness(t) # assemble(t,1,2x)
    M,b = assemblemassfunction(t,2x).+assemblerobin(e,1e6,0,0)
    h = Float64(T.step); LHS = M+h*A # time step
    for l ∈ 1:m
        ξ = LHS\(M*ξ+h*b); l%10==0 && println(l*h)
    end
    mesh(t,color=ξ)
end
```
More general problems for finite element boundary value problems are also enabled by mesh representations imported from external sources. These methods can automatically generalize to higher dimensional manifolds and is compatible with discrete differential geometry.
