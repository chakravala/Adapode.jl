

using Grassmann, Adapode, Makie
x0 = Chain(10.0,10.0,10.0)

Lorenz(σ,r,b) = x -> Chain(
	σ*(x[2]-x[1]),
	x[1]*(r-x[3])-x[2],
	x[1]*x[2]-b*x[3])
lines(odesolve(Lorenz(10.0,28.0,8/3),Chain(10.0,10.0,10.0)))
lines(odesolve(Lorenz(10.0,60.0,8/3),Chain(10.0,10.0,10.0)))

DiskDynamo(a,b,c) = x -> Chain(
    a*(x[2]-x[1]),
    x[3]*x[1]-x[2],
    b-x[1]*x[2]-c*x[3])
lines(odesolve(DiskDynamo(14.625,1.0,5.0),Chain(10.0,10.0,10.0)))

Rossler(a,b,c) = x -> Chain(
    -(x[2]+x[3]),
    x[1]+a*x[2],
    b+x[3]*(x[1]-c))
lines(odesolve(Rossler(1/5,1/5,2.4),x0))
lines(odesolve(Rossler(1/5,1/5,3.5),x0))
lines(odesolve(Rossler(1/5,1/5,4.0),x0))
lines(odesolve(Rossler(1/5,1/5,4.23),x0))
lines(odesolve(Rossler(1/5,1/5,4.3),x0))
lines(odesolve(Rossler(1/5,1/5,5.0),x0))
lines(odesolve(Rossler(1/5,1/5,5.7),x0))
lines(odesolve(Rossler(0.0,0.0,12.0),x0))
lines(odesolve(Rossler(0.0,0.0,25.0),x0))
lines(odesolve(Rossler(0.343,1.82,9.75),x0))

ChemicalKinetics(a1,a2,a3,a4,a5,k1,k2) = x -> Chain(
    x[1]*(a1-k1*x[1]-x[3]-x[2])+k2*x[2]*x[2]+a3,
    x[2]*(x[1]-k2*x[2]-a5)+a2,
    x[3]*(a4-x[1]-k5*x[3])+a3)
lines(odesolve(ChemicalKinetics(30.0,0.01,0.01,16.5,10.0,0.25,0.001,0.5),x0))

Rossler4(a,b,c,d) = x -> Chain(
    -(x[2]+x[3]),
    x[1]+a*x[2]+x[4],
    b+x[3]*x[1],
    d*x[4]-c*x[3])
lines(odesolve(Rossler4(1/4,3.0,0.5,0.05),x0))


