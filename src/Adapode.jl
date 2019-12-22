module Adapode

#   This file is part of Aadapode.jl. It is licensed under the GPL license
#   Adapode Copyright (C) 2019 Michael Reed

using StaticArrays, Grassmann
import Grassmann: value, vector, valuetype

export Lorenz, odesolve

function Lorenz(t::T) where T<:TensorAlgebra{V} where V
    t,x = promote_type(valuetype(t),Float64),Chain(vector(t))
    Chain{V,1,t}(SVector(1.0,10.0(x[3]-x[2]),x[2]*(28.0-x[4])-x[3],x[2]*x[3]-(8/3)*x[4]))
end

include("constants.jl")

expl_euler(F,x,h) = x+h*F(x)
impl_euler(F,x,h) = x+h*F(x+h*F(x)) # to be implemented
improved_heun(F,x,h) = (Fx = F(x); x+(h/2)*(Fx+F(x+h*Fx)))

function adaptive_method(F,x,h,m,b,c,ce)
    K = explicit_k(F,x,h,b,m)
    p = c[1]*K[1]
    s = ce[1]*K[1]
    for r ∈ 2:m
        p += c[r]*K[r]
        s += ce[r]*K[r]
    end
    return x + h*p, maximum(abs.(value(h*s)))
end

function general_method(F,x,h,m,b,c)
    K = explicit_k(F,x,h,b,m)
    p = c[1]*K[1]
    for r ∈ 2:m
        p += c[r]*K[r]
    end
    return x + h*p
end

function explicit_k(F,x,h,b,m)
    out = Vector(undef,m)
    out[1] = F(x)
    for r ∈ 2:m
        z = 0
        for s ∈ 1:r-1
            z += b[r,s]*out[s]
        end
        out[r] = F(x+h*z)
    end
    return out
end

function predictor_corrector!(F,x,h,m,Fx,cp,cc)
    order = length(cp)
    mp1 = (m%(order+1))+1
    Fx[m] = F(x)
    s = 0
    for k ∈ 1:order
        j = ((m-k+order+1)%(order+1))+1
        s += cp[k]*Fx[j]
    end
    y = x+h*s
    Fx[mp1] = F(y)
    s = 0
    for k ∈ 1:order
        j = ((mp1-k+order+1)%(order+1))+1
        s += cc[k]*Fx[j]
    end
    xout = x+h*s
    return xout, maximum(abs.(value(xout-y)./value(xout))), mp1
end

function odesolve(F,x0::T,B=(0,2π),tol=15,mode=15) where T<:TensorAlgebra{V} where V
    a,b = B
    h = 2.0^-tol
    N = Int(round((b-a)*2^tol))
    eqc = ndims(V)
    emax = 1e1^-tol
    emin = 1e1^(-tol-3)
    hmin = 1e-16
    hmax = 1e-4
    itmax = 5^(tol+5)
    Y = Vector{Chain{V,1,Float64,eqc}}(undef,N+1)
    Y[1] = x0 # Set initial conditions
    if mode ≤ 3
        if mode == 1 # Explicit Euler Method
            for i ∈ 1:N
                Y[i+1] = expl_euler(F,Y[i],h)
            end
        elseif mode == 2 # Implicit Euler Method
            for i ∈ 1:N
                Y[i+1] = impl_euler(F,Y[i],h)
            end
        elseif mode == 3 # Improved Euler, Heun's Method
            for i ∈ 1:N
                Y[i+1] = improved_heun(F,Y[i],h)
            end
        end
    elseif mode < 10 # Single Step
        m,brs,c = if mode == 4 # Midpoint 2nd Order RK Method
            midpoint2nd()
        elseif mode == 5 # Kutta's 3rd Order RK Method
            kutta3rd()
        elseif mode == 6 # Classical 4th Order RK Method
            classical4th()
        end
        for i ∈ 1:N
            Y[i+1] = general_method(F,Y[i],h,m,brs,c)
        end
    elseif mode < 20 # Adaptive Single Step
        m,brs,c,ce = if mode == 11 # Adaptive Heun-Euler
            adaptive_heun_euler()
        elseif mode == 12 # Adaptive BOgacki-Shampine RK23 Method
            bogacki_shampine_rk23()
        elseif mode == 13 # Adaptive Fehlberg RK45 Method
            fehlberg_rk45()
        elseif mode == 14 # Adaptive Cash-Karp RK45 Method
            cash_karp_rk45()
        elseif mode == 15 # Adaptive Dormand-Prince RK45 Method
            dormand_prince_rk45()
        end
        i = 0
        iflag = 1
        h = checkhsize(h,hmin,hmax)
        while i ≤ itmax
            i = i + 1
            x = Y[i]
            h,iflag = checkfinalstep(x[1],b,h,iflag)
            resize_array!(Y,i,10000)
            Y[i+1],e = adaptive_method(F,x,h,m,brs,c,ce)
            iflag == 0 && break
            e < emin && (h *= 2)
            if e > emax
                h /= 2
                i -= 1
            end
            h = checkhsize(h,hmin,hmax)
            show_progress(x,i,b)
        end
        truncate_array!(Y,i)
    elseif mode < 30 # Multistep
        m,brs,c = classical4th() # Classical RK4 Initializer
        mm,cp,cc,Fx = if mode == 22 # Adams-Bashorth-Moulton 2nd Order
            abm2nd(eqc)
        elseif mode == 23 # Adams-Bashorth-Moulton 3rd Order
            abm3rd(eqc)
        elseif mode == 24 # Adams-Bashorth-Moulton 4th Order
            abm4th(eqc)
        elseif mode == 25 # Adams-Bashorth-Moulton 5th Order
            abm5th(eqc)
        end
        for i ∈ 1:mm-1
            x = Y[i]
            Y[i+1] = general_method(F,x,h,m,brs,c)
            Fx[i] = F(x)
        end
        q = mm
        for i ∈ mm:N
            x = Y[i]
            Y[i+1],e,q = predictor_corrector!(F,x,h,q,Fx,cp,cc)
        end
    else # Adaptive Multistep
        m,brs,c = classical4th() # Classical RK4 Initializer
        mm,cp,cc,Fx = if mode == 32 # Adaptive Adams-Bashorth-Moulton 2nd Order
            abm2nd(eqc)
        elseif mode == 33 # Adaptive Adams-Bashorth-Moulton 3rd Order
            abm3rd(eqc)
        elseif mode == 34 # Adaptive Adams-Bashorth-Moulton 4th Order
            abm4th(eqc)
        elseif mode == 35 # Adaptive Adams-Bashorth-Moulton 5th Order
            abm5th(eqc)
        end
        i = 0
        iflag = 1
        initialize = 0
        h = checkhsize(h,hmin,hmax)
        while i ≤ itmax
            i += 1
            x = Y[i]
            h,iflag = checkfinalstep(x[1],b,h,iflag)
            resize_array!(Y,i+mm,10000)
            if initialize == 0
                for j ∈ 1:mm-1
                    Y[i+j] = general_method(F,x,h,m,brs,c)
                    Fx[j] = F(x)
                    x = Y[i+j]
                end
                q = mm
                initialize = 1
                i += mm - 1
            end
            Y[i+1],e,q = predictor_corrector!(F,x,h,q,Fx,cp,cc)
            iflag == 0 && break
            if e < emin
                h *= 2
                i -= Int(ceil(mm/2))
                initialize = 0
            end
            if e > emax
                h /= 2
                i -= Int(ceil(mm/2))
                initialize = 0
            end
            h = checkhsize(h,hmin,hmax)
            show_progress(x,i,b)
        end
        truncate_array!(Y,i)
    end
    return Y
end

function checkhsize(h,hmin,hmax)
    abs(h) < hmin && (return copysign(hmin,h))
    abs(h) > hmax && (return copysign(hmax,h))
    return h
end

resize_array!(Y,i,h) = length(Y)<i+1 && resize!(Y,i+h)
truncate_array!(Y,i) = length(Y)>i+1 && resize!(Y,i)

function checkfinalstep(a,b,h,iflag)
    d = abs(b-a)
    if d ≤ abs(h)
        iflag = 0
        h = copysign(d,h)
    end
    return h,iflag
end

show_progress(x,i,b) = i%75000 == 11 && println(x[1]," out of ",b)

end # module
