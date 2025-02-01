using Grassmann, Cartan
using Adapode, Test

x0 = Chain(10.0,10.0,10.0)
Lorenz(x) = Chain(
	10.0(x[2]-x[1]),
	x[1]*(28.0-x[3])-x[2],
	x[1]*x[2]-(8/3)*x[3])
for k ∈ 0:4
    @test (odesolve(Lorenz,x0,2π,7,Val(k),Val(4)); true)
end
pt,pe = Cartan.initmesh(0:1/5:1)
@test (\(assemblemassload(pt,x->x[2]*sin(x[2]))...); true)
@test (\(assemblemassload(pt,x->2x[2]*sin(2π*x[2])+3)...); true)
@test begin # BackwardEulerHeat1D
    x,m = 0:0.01:1,100; pt,pe = Cartan.initmesh(x)
    T = range(0,0.5,length=m+1) # time grid
    ξ = 0.5.-abs.(0.5.-x) # initial condition
    A = assemblestiffness(pt) # assemble(t,1,2x)
    M,b = assemblemassload(pt,2x).+assemblerobin(pe,1e6,0,0)
    h = Float64(T.step); LHS = M+h*A # time step
    for l ∈ 1:m
        ξ = LHS\(M*ξ+h*b)
    end
    true
end
@test (adaptpoisson(refinemesh(0:0.25:1)...,1,0,x->exp(-100abs2(x[2]-0.5))); true)
