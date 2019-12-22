# Adapode.jl

Adaptive multistep numerical ODE solver based on [Grassmann.jl](https://github.com/chakravala/Grassmann.jl) tensor fields.

This Julia project originally started as a FORTRAN 95 project called [adapode](https://github.com/chakravala/adapode).

```julia
using Grassmann, Adapode, Makie
basis"4"; x0 = 10.0v2+10.0v3+10.0v4
lines(Point.((V(2,3,4)).(odesolve(Lorenz,x0))))
```
