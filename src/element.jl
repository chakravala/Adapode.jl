
#   This file is part of Adapode.jl
#   It is licensed under the AGPL license
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

export assemble, assembleglobal, assemblestiffness, assembleconvection, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assembledivergence
export assemblemassincidence, asssemblemasstotalnodes, assembletotalnodes
export assembleelastic, assemblestokes, assemblemaxwell
export assembleload, assemblemassload, assemblerobin, edges, edgesindices, neighbors
export solvepoisson, solvetransport, solvetransportdiffusion, solvemaxwell
export solveelastic, solvenonlinearpoisson, solvebistable, solvebistable_newton
export solvedirichlet, adaptpoisson, solveheat, solvewave, solvestokes, solvenavierstokes
export gradienthat, gradientCR, gradient, interp, nedelec, nedelecmean, jumps
export submesh, detsimplex, iterable, callable, value, edgelengths, laplacian
export boundary, interior, trilength, trinormals, incidence, degrees
import Grassmann: norm, column, columns
using Base.Threads

import Cartan: points, pointset, edges, iterpts, iterable, callable, revrot
import Cartan: gradienthat, laplacian, gradient, assemblelocal!

trilength(rc) = value.(abs.(value(rc)))
function trinormals(t)
    c = curls(t)
    ds = trilength.(c)
    V = Manifold(c); V2 = ↓(V)
    dn = [Chain{V,1}(revrot.(V2.(value(c[k]))./-ds[k])) for k ∈ 1:length(c)]
    return ds,dn
end

gradientCR(t,m) = gradientCR(gradienthat(t,m))
gradientCR(g) = gradientCR.(g)
function gradientCR(g::Chain{V}) where V
    Chain{V,1}(g.⋅Values(
        Chain{V,1}(-1,1,1),
        Chain{V,1}(1,-1,1),
        Chain{V,1}(1,1,-1)))
end

assembleglobal(M,t::SimplexFrameBundle,m=volumes(t),c=1,g=0) = assembleglobal(M,immersion(t),iterable(t,m),iterable(t,c),iterable(t,g))
function assembleglobal(M,t::SimplexTopology{N},m::T,c::C,g::F) where {N,T<:AbstractVector,C<:AbstractVector,F<:AbstractVector}
    np,v = totalnodes(t),Val(N)
    A = spzeros(np,np)
    for k ∈ 1:length(t)
        assemblelocal!(A,M(c[k],g[k],v),m[k],value(t[k]))
    end
    return A
end

import Cartan: weights, degrees, assembleincidence, incidence

assemblemassincidence(t::SimplexFrameBundle,f,m=volumes(t),l=m) = assemblemassincidence(immersion(t),iterpts(t,f),iterable(t,m),iterable(t,l))
function assemblemassincidence(t::SimplexTopology{N},f::F,m::V,l::T) where {N,F<:AbstractVector,V<:AbstractVector,T<:AbstractVector}
    np,n = totalnodes(t),Val(N)
    M,b = spzeros(np,np), zeros(np)
    for k ∈ 1:length(t)
        tk = value(t[k])
        assemblelocal!(M,mass(nothing,nothing,n),m[k],tk)
        b[tk] .+= f[tk]*l[k]
    end
    return M,b
end

assemblemassload(t,f,m=volumes(t),l=m) = assemblemassincidence(t,iterpts(t,f)/sdims(t),m,l)
assemblemassload(tf::TensorField,m=volumes(base(tf)),l=m) = assemblemassload(base(tf),fiber(tf),m,l)

import Cartan: assembleload, interp, pretni

#mass(a,b,::Val{N}) where N = (ones(SMatrix{N,N,Int})+I)/Int(factorial(N+1)/factorial(N-1))
mass(a,b,::Val{N}) where N = (x=Submanifold(N)(∇);outer(x,x)+I)/Int(factorial(N+1)/factorial(N-1))
assemblemass(t,m=volumes(t)) = assembleglobal(mass,t,iterpts(t,m))

stiffness(c,g::Float64,::Val{2}) = (cg = c*g^2; Chain(Chain(cg,-cg),Chain(-cg,cg)))
stiffness(c,g,::Val{N}) where N = Chain{Submanifold(N),1}(map.(*,c,value(g).⋅Ref(g)))
assemblestiffness(t,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
assemblestiffness(t::SimplexFrameBundle,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(stiffness,t,m,iterable(c isa Real ? immersion(t) : means(t),c),g)

# iterable(means(t),c) # mapping of c.(means(t))

#=function sonicstiffness(c,g,::Val{N}) where N
    A = zeros(MMatrix{N,N,typeof(c)})
    for i ∈ 1:N, j ∈ 1:N
        A[i,j] = c*g[i][1]^2+g[j][2]^2
    end
    return SMatrix{N,N,typeof(c)}(A)
end
assemblesonic(t,c=1,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(sonicstiffness,t,m,iterable(c isa Real ? t : means(t),c),g)
# iterable(means(t),c) # mapping of c.(means(t))=#

#convection(b,g,::Val{N}) where N = ones(Values{N,Int})*column((b/N).⋅value(g))'
convection(b,g,::Val{N}) where N = outer(∇,Chain(Real.((b/N).⋅value(g))))
assembleconvection(t,b,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(convection,t,m,b,g)

#SD(b,g,::Val) = (x=column(b.⋅value(g));x*x')
SD(b,g,::Val) = (x=Chain(column(b.⋅value(g)));outer(x,x))
assembleSD(t,b,m=volumes(t),g=gradienthat(t,m)) = assembleglobal(SD,t,m,b,g)

function assembledivergence(t2e,m,g)
    nt,ne = length(m),totalnodes(t2e)
    D1,D2 = spzeros(nt,ne),spzeros(nt,ne)
    for k ∈ 1:nt
        edges,gm = immersion(t2e)[k],value(g[k])*m[k]
        D1[k,edges] .= getindex.(gm,1)
        D2[k,edges] .= getindex.(gm,2)
    end
    return D1,D2
end

function assemble(t,c=1,a=1,f=0,m=volumes(t),g=gradienthat(t,m))
    M,b = assemblemassload(t,f,typeof(a)<:Real && isone(a) ? m : a.*m,m)
    return assemblestiffness(t,c,m,g),M,b
end

assemblerobin(eκ::TensorField) = assemblerobin(base(eκ),fiber(eκ))
assemblerobin(e,κ=1e6) = assemblemassload(e,1,iterable(means(e),κ).*volumes(e),0)
function assemblerobin(e,κ,gD,gN=0)
    a = means(e)
    v = volumes(e)
    m = iterable(a,κ)
    l = m.*iterable(a,gD).+iterable(a,gN)
    return assemblemassload(e,1,m.*v,l.*v)
end

function solvepoisson(t,e,c,f,κ,gD=0,gN=0)
    m = volumes(t)
    b = assembleload(t,f,m)
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

function solvetransport(t,e,c,f=1,ϵ=0.1)
    m = volumes(t)
    g = gradienthat(t,m)
    A = assemblestiffness(t,ϵ,m,g)
    b = assembleload(t,f,m)
    C = assembleconvection(t,c,m,g)
    TensorField(t,solvedirichlet(A+C,b,e))
end

function adaptpoisson(g,p,e,t,c=1,a=0,f=1,κ=1e6,gD=0,gN=0)
    ϵ = 1.0
    while ϵ > 5e-5 && length(t) < 10000
        m = volumes(t)
        h = gradienthat(t,m)
        A,M,b = assemble(t,c,a,f,m,h)
        ξ = solvedirichlet(A+M,b,e)
        η = jumps(t,c,a,f,ξ,m,h)
        ϵ = sqrt(norm(η)^2/length(η))
        println(t,", ϵ=$ϵ, α=$(ϵ/maximum(η))")
        refinemesh!(g,p,e,t,select(η,ϵ),"regular")
    end
    return g,p,e,t
end

#solvedirichlet(M,b,e::ChainBundle) = solvedirichlet(M,b,pointset(e))
#solvedirichlet(M,b,e::ChainBundle,u) = solvedirichlet(M,b,pointset(e),u)
solvedirichlet(M,b,e::SimplexFrameBundle) = solvedirichlet(M,b,vertices(e))
solvedirichlet(M,b,e::SimplexTopology) = solvedirichlet(M,b,vertices(e))
solvedirichlet(M,b,e::SimplexFrameBundle,u) = solvedirichlet(M,b,vertices(e),u)
solvedirichlet(M,b,e::SimplexTopology,u) = solvedirichlet(M,b,vertices(e),u)
function solvedirichlet(A,b,fixed,boundary)
    neq = length(b) # number of equations
    free,ξ = interior(fixed,neq),zeros(eltype(b),neq)
    ξ[fixed] = boundary # set boundary condition
    ξ[free] = A[free,free]\(b[free]-A[free,fixed]*boundary)
    return ξ
end
function solvedirichlet(M,b,fixed)
    neq = length(b)
    free,ξ = interior(fixed,neq),zeros(eltype(b),neq)
    ξ[free] = M[free,free]\b[free]
    return ξ
end

import Cartan: interior

solvehomogenous(e,M,b) = solvedirichlet(M,b,e)
export solvehomogenous, solveboundary
const solveboundary = solvedirichlet # deprecate
const edgelengths = volumes # deprecate
const boundary = pointset # deprecate

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

function solvewave(pt,bc) # Crank-Nicolson
    Δt = step(base(bc).g.v[1])
    np = length(points(pt))
    sol = zeros(2np)
    out = zeros(np,length(base(bc).g))
    fixed = vertices(base(bc).s)
    A,M,b = assemble(pt,1,1,0)
    LHS = [M (Δt/-2)*M; (Δt/2)*A M] # Crank-Nicolson
    rhs = [M (Δt/2)*M; (Δt/-2)*A M]
    zkb = [zeros(np); Δt*b]
    for l ∈ 1:length(base(bc).g) # time loop
        sol .= LHS\(rhs*sol + zkb)
        sol[fixed] .= fiber(bc)[:,l] # set BC the ugly way
        out[:,l] .= sol[1:np]
    end
    return TensorField(pt⊕base(bc).g.v[1],out)
end

function assemblestokes(pt,ν=0.1,t2e=edgesindices(pt))
    nt,ne = length(immersion(pt)),length(t2e)
    m = volumes(pt)
    gCR = gradientCR(gradienthat(pt,m))
    A11 = ν*assemblestiffness(t2e,1,m,gCR)
    B1,B2 = assembledivergence(t2e,m,-gCR)
    #=B1,B2 = spzeros(nt,ne),spzeros(nt,ne)
    for i ∈ 1:nt
        edges,gCRi = immersion(t2e)[i],value(gCR[i])*(-m[i])
        B1[i,edges] .= getindex.(gCRi,1)
        B2[i,edges] .= getindex.(gCRi,2)
    end=#
    [A11 spzeros(ne,ne) B1' spzeros(ne,1);
     spzeros(ne,ne) A11 B2' spzeros(ne,1);
     B1 B2 spzeros(nt,nt) m;
     spzeros(1,ne) spzeros(1,ne) m' 0]
end

function solvestokes(pt,bc,ν=0.1,t2e=edgesindices(pt,base(bc)))
    nt,ne = length(immersion(pt)),length(t2e)
    LHS = assemblestokes(pt,ν,t2e)
    fixed = [subelements(bc); subelements(bc).+ne] # fixed totalnodes
    gvals = [getindex.(fiber(bc),1); getindex.(fiber(bc),2)] # nodal values of g
    ξ = solvedirichlet(LHS,zeros(2ne+nt+1),fixed,gvals)
    UV = Chain{Manifold(fibertype(bc))}.(ξ[1:ne],ξ[1+ne:2ne])
    P = ξ[2ne+1:2ne+nt]
    TensorField(pt,interp(fullimmersion(bc),UV)),TensorField(pt,interp(pt,P))
end

function solvenavierstokes(p,e,t,inbc,outbc,ν=0.001,T=range(0,1,101))
    dt,nt = step(T),length(T)
    V = Cartan.varmanifold(3)(2,3)
    np = length(p)
    ins = vertices(inbc)
    inp = fiber(inbc) # inflow profile
    out = vertices(outbc) # totalnodes on outflow
    bnd = setdiff(vertices(e),out) # remove outflow totalnodes from boundary totalnodes
    R = spzeros(np,np) # diagonal penalty matrix
    for i ∈ 1:length(out)
        j = out[i]
        R[j,j] = fiber(outbc)[i] # big weights
    end
    m = volumes(p(t))
    g = gradienthat(p(t),m)
    A = assemblestiffness(p(t),1,m,g)
    M = assembleload(p(t),1,m)
    Bx = assembleconvection(p(t),Global{1}(Chain{V}(1,0)),m,g)
    By = assembleconvection(p(t),Global{1}(Chain{V}(0,1)),m,g)
    B = Chain{V}.(Bx,By)
    νA,AR = ν*A,A+R
    dtiM = dt*inv.(M)
    UVs = Matrix{Chain{V,1,Float64,mdims(V)}}(undef,np,nt)
    UVs[:,1] = Chain{V,1,Float64}.(zeros(np),zeros(np)) # init velocity
    for l ∈ 1:nt-1
        C = assembleconvection(p(t),means(t,UVs[:,l]),m,g)
        UVs[:,l+1] = UVs[:,l] - ((νA+C)*UVs[:,l]).*dtiM # compute tentative velocity
        UVs[bnd,l+1] .*= 0 # no-slip boundary totalnodes
        UVs[ins,l+1] += inp # inflow profile totalnodes
        P = (AR\(Bx*getindex.(UVs[:,l+1],1)+By*getindex.(UVs[:,l+1],2)))/-dt # solve PPE
        UVs[:,l+1] -= (B*P).*dtiM # update velocity
    end
    return TensorField(p(t)⊕T,UVs)
end

function assembleelastic(f,μ,λ)
    t = immersion(f)
    ndof = 2length(f) # total number of DoFs
    K = spzeros(ndof,ndof) # stiffness
    M = spzeros(ndof,ndof) # mass
    F = zeros(ndof) # load
    m = volumes(base(f))
    g = gradienthat(base(f),m)
    for i ∈ 1:length(t) # assembly loop over subelements
        ti,mi = t[i],m[i]
        fti = fiber(f)[ti]
        dofs1 = 2ti
        dofs2 = dofs1.-1
        MK = mass(nothing,nothing,Val(3))*mi
        assemblelocal!(K,elasticstrain(λ,μ,g[i],Val(3)),mi,Values(dofs1...,dofs2...))
        assemblelocal!(M,MK,dofs1)
        assemblelocal!(M,MK,dofs2)
        F[dofs1] .= value(MK⋅Chain(getindex.(fti,1)))
        F[dofs2] .= value(MK⋅Chain(getindex.(fti,2)))
    end
    return K,M,F
end

function solveelastic(f,e,E=1,ν=0.3)
    K,M,F = assembleelastic(f,Eν2Lame(E,ν)...)
    fixed = [2vertices(e).-1; 2vertices(e)] # boundary DoFs
    d = solvedirichlet(K,F,fixed,zeros(length(fixed)))
    TensorField(base(f),Chain{Cartan.varmanifold(3)(2,3)}.(d[1:2:end],d[2:2:end]))
end

function assemblemaxwell(p,e,t,κ,μ,fhat,t2e=edgesindices(p(t)),signs=facetsigns(t))
    nt,ne = length(κ),length(e)
    A,b = spzeros(Complex{Float64},ne,ne),zeros(ne)
    m = volumes(p(t))
    g = gradienthat(p(t),m)
    l = volumes(p(e))
    for i ∈ 1:nt
        ed = immersion(t2e)[i]
        li = l[ed].*signs[i]
        κi = fiber(κ)[i]
        assemblelocal!(A,maxwell(inv(m[i]*m[i]*μ),li,κi*κi,g[i],Val(3)),m[i],ed)
        b[ed] .+= Real.(fhat.⋅value(curl(g[i]))).*(li*(m[i]/3))
    end
    return A,b
end

function solvemaxwell(κ,bc,μ=1,fhat=Chain{Cartan.varmanifold(3)(2,3)}(0,0))
    p = fullcoordinates(κ)
    t = fullimmersion(κ)
    e = fullimmersion(bc)
    t2e = edgesindices(p(t))#,p(e))
    signs = facetsigns(t)
    A,b = assemblemaxwell(p,e,t,κ,μ,fhat,t2e,signs)
    ξ = solvedirichlet(A,complex.(b),subelements(bc),fiber(bc))
    RI = TensorField(p(t),interp(t,nedelecmean(p(t),t2e,signs,ξ)))
    TensorField(p(t),map.(real,fiber(RI))),TensorField(p(t),map.(imag,fiber(RI)))
end

function assemblejacobianresidue(f,e,u,Afcn,m,g,tiny=1e-8)
    pt = SimplexFrameBundle(base(f))
    # evaluate u, a, a', and f
    uc = means(immersion(f),u)
    a = Afcn.(uc)
    da = (Afcn.(uc.+tiny)-a)/tiny # da(u)/du
    gu = Cartan.gradient_2(pt,u,m,g) # grad u
    # Assemble Jacobian and residual
    np = nodes(immersion(f))
    J,r = spzeros(np,np),zeros(np)
    for i ∈ 1:length(immersion(f))
        nodes = t[i]
        gug = Ref(value(fiber(gu)[i])).⋅value.(value(g[i]))
        dagug = Chain{Cartan.varmanifold(3)}((da[i]/3)*gug)
        JK = Adapode.stiffness(a[i],g[i],Val(3))+Chain(dagug,dagug,dagug)
        Adapode.assemblelocal!(J,JK,m[i],nodes)
        r[nodes] .+= ((fiber(f)[i]/3)*Values(1,1,1)-a[i]*gug)*m[i]
    end
    # enforce zero Dirichlet BC
    for n ∈ vertices(e) # boundary nodes
        J[n,:] .= 0 # zero out row n of the Jacobian, J
        J[n,n] = 1 # set diagonal entry J[n,n] to 1
        r[n] = 0 # set residual entry r[n] to 0
    end
    return J,r
end

function solvenonlinearpoisson(f,e,Afcn)
    pt = SimplexFrameBundle(base(f))
    m = volumes(pt)
    g = gradienthat(pt,m)
    ξ = zeros(nodes(immersion(f)))
    for k ∈ 1:5 # non-linear loop
        J,r = assemblejacobianresidue(f,e,ξ,Afcn,m,g)
        d = J\r
        ξ = ξ + d # update solution by solving correction
        println("|d|=$(norm(d)), |r|=$(norm(r))")
    end
    TensorField(p(t),ξ)
end

function solvebistable(ic,ϵ=0.01,T=StepRangeLen(0,0.1,101),f=u->u-u^3)
    dt = step(T)
    ξ = zeros(length(ic),length(T))
    ξ[:,1] = fiber(ic) # IC
    ξ_new = ξ[:,1]
    A,M = assemble(base(ic))
    for l ∈ 1:length(T)-1 # time loop
        for k ∈ list(1,3) # non-linear loop
            ξ_tmp = ξ_new
            ξ_new = (M+(dt*ϵ)*A)\(M*ξ[:,l]+dt*(M*f.(ξ_tmp)))
            #fixpterror = norm(ξ_tmp-ξ_new)
        end
        ξ[:,l+1] = ξ_new
    end
    TensorField(base(ic)⊕T,ξ)
end

function solvebistable_newton(ic,ϵ=0.01,T=StepRangeLen(0,0.1,101),f=u->u-u^3,df=u->1-3u^2)
    dt = step(T)
    np = length(ic)
    t = immersion(ic)
    ξ = zeros(length(ic),length(T))
    ξ[:,1] = fiber(ic) # IC
    ξ_new = ξ[:,1]
    m = volumes(base(ic))
    g = gradienthat(base(ic),m)
    A,M = assemble(base(ic),1,1,0,m,g)
    for l ∈ 1:length(T)-1 # time loop
        for k ∈ Cartan.list(1,3) # non-linear loop
            ξ_tmp = ξ_new
            ξ_tmp_mid = means(immersion(ic),ξ_tmp)
            Mdf,b = assemblemassload(base(ic),f.(ξ_tmp),df.(ξ_tmp_mix).*m,m)
            MA = M+(dt*ϵ)*A
            J = MA - dt*Mdf # Jacobian
            ρ = MA*ξ_new - M*ξ[:,l] - dt*b # residual
            ξ_new = ξ_tmp - J\ρ # Newton update
            #error = norm(ξ_tmp-ξ_new)
        end
        ξ[:,l+1] = ξ_new
    end
    TensorField(base(ic)⊕T,ξ)
end

import Cartan: facesindices, edgesindices, neighbor, neighbors, centroidvectors

Eν2Lame(E,ν) = E/(2(1+ν)),E*ν/((1+ν)*(1-2ν))

elastic(μ,λ,::Val{3}) = DiagonalOperator(μ*Chain(2,2,1))+λ*TensorOperator(Chain(Chain(1,1,0),Chain(1,1,0),Chain(0,0,0)))
function strain(g,::Val{3})
    TensorOperator(Chain(
        Chain(g[1][1],0,g[1][2]),
        Chain(0,g[1][2],g[1][1]),
        Chain(g[2][1],0,g[2][2]),
        Chain(0,g[2][2],g[2][1]),
        Chain(g[3][1],0,g[3][2]),
        Chain(0,g[3][2],g[3][1])))
end
function elasticstrain(μ,λ,g,v::Val{3})
    D,BK = elastic(μ,λ,v),strain(g,v)
    return (transpose(BK)⋅D)⋅BK
end

function maxwell(mμ,l,λ,g,v::Val{3})
    n11,n22,n33,n12,n13,n23 = _nedelec(λ,g,v)
    m11 = (mμ+n11)*(l[1]*l[1])
    m22 = (mμ+n22)*(l[2]*l[2])
    m33 = (mμ+n33)*(l[3]*l[3])
    m12 = (mμ+n12)*(l[1]*l[2])
    m13 = (mμ+n13)*(l[1]*l[3])
    m23 = (mμ+n23)*(l[2]*l[3])
    Chain(Chain(m11,m12,m13),Chain(m12,m22,m23),Chain(m13,m23,m33))
end

function _nedelec(λ,g,v::Val{3})
    f = stiffness(λ,g,v)
    m11 = (f[3,3]-f[2,3]+f[2,2])/6
    m22 = (f[1,1]-f[1,3]+f[3,3])/6
    m33 = (f[2,2]-f[1,2]+f[1,1])/6
    m12 = (f[3,1]-f[3,3]-2f[2,1]+f[2,3])/12
    m13 = (f[3,2]-2f[3,1]-f[2,2]+f[2,1])/12
    m23 = (f[1,2]-f[1,1]-2f[3,2]+f[3,1])/12
    return m11,m22,m33,m12,m13,m23
end
function nedelec(λ,g,v::Val{3})
    m11,m22,m33,m12,m13,m23 = _nedelec(λ,g,v)
    Chain(Chain(m11,m12,m13),Chain(m12,m22,m23),Chain(m13,m23,m33))
end

function basisnedelec(p)
    M = Cartan.varmanifold(3); V = ↓(M)
    Chain{M,1}(
        Chain{V,1}(-p[2],p[1]),
        Chain{V,1}(-p[2],p[1]-1),
        Chain{V,1}(1-p[2],p[1]))
end

function nedelecmean(t,t2e,signs,u)
    base = Grassmann.vectors(t)
    B = revrot.(base,revrot)./Real.(.∧(base))
    N = basisnedelec(Values(1,1)/3)
    x,y,z = columns(immersion(t2e)); X,Y,Z = columns(signs,1,3)
    (u[x].*X).*(B.⋅N[1]) + (u[y].*Y).*(B.⋅N[2]) + (u[z].*Z).*(B.⋅N[3])
end

function jumps(t,c,a,f,u,m=volumes(t),g=gradienthat(t,m))
    N,np,nt = mdims(Manifold(t)),length(points(t)),length(t)
    η = zeros(nt)
    if N == 2
        fau = iterable(points(t),f).-a*u
        @threads for i ∈ 1:nt
            η[i] = m[i]*sqrt((fau[i]^2+fau[i+1]^2)*m[i]/2)
        end
    elseif N == 3
        ds,dn = trinormals(t) # ds.^1
        du,F,cols = gradient(t,u,m,g),iterable(t,f),columns(t)
        fl = [-c*column(value(dn[k]).⋅du[k]) for k ∈ 1:length(du)]
        intj = round.(adjacency(t,cols)/3)
        i,j,k = cols; x,y,z = getindex.(fl,1),getindex.(fl,2),getindex.(fl,3)
        jmps = sparse(j,k,x,np,np)+sparse(k,i,y,np,np)+sparse(i,j,z,np,np)
        jmps = abs.(intj.*abs.(jmps+jmps'))
        @threads for k = 1:nt
            tk,dsk = t[k],ds[k]
            η[k] = sqrt(((dsk[3]*jmps[tk[1],tk[2]])^2+(dsk[1]*jmps[tk[2],tk[3]])^2+(dsk[2]*jmps[tk[3],tk[1]])^2)/2)
        end
        η += [sqrt(norm(F[k].-a*u[value(t[k])])/3m[k]) for k ∈ 1:nt].*maximum.(ds)
    else
        throw(error("jumps on Manifold{$N} not defined"))
    end
    return η
end
