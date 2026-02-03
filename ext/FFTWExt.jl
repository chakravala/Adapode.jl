module FFTWExt

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

using Grassmann, Cartan, Adapode
isdefined(Adapode, :Requires) ? (import Adapode: FFTW) : (using FFTW)

import Adapode: heatneumann, heatdirichlet, rieszneumann, rieszdirichlet
import Adapode: biharmonicneumann, biharmonicdirichlet
import Adapode: restwaveneumann, restwavedirichlet
import Adapode: waveneumann, wavedirichlet

heatneumann(u0,t,k=r2rspace(points(u0))) = idct(dct(u0).*heatmultiplier.(k,t))
function heatdirichlet(u0,t,ω=r2rspace(points(u0),FFTW.RODFT10))
    FFTW.r2r((FFTW.r2r(u0, FFTW.RODFT10)/prod(2.0.*size(u0))).*heatmultiplier.(ω,t),FFTW.RODFT01)
end

rieszneumann(u0,t,s,k=r2rspace(points(u0))) = idct(dct(u0).*rieszmultiplier.(k,t,s))
function rieszdirichlet(u0,t,s,ω=r2rspace(points(u0),FFTW.RODFT10))
    FFTW.r2r((FFTW.r2r(u0, FFTW.RODFT10)/prod(2.0.*size(u0))).*rieszmultiplier.(ω,t,s),FFTW.RODFT01)
end
rieszdirichlet(u0,t::AbstractVector,s,k=r2rspace(points(u0),FFTW.RODFT10)) = rieszdirichlet(u0,TensorField(t),s,k)
function rieszdirichlet(u0,t::TensorField,s,k=r2rspace(points(u0),FFTW.RODFT10))
    data = zeros(size(u0)...,length(t))
    out = TensorField(base(u0)⊕base(t),data)
    Adapode.assign!(out,1,u0)
    for i in 2:length(t)
        Adapode.assign!(out,i,rieszdirichlet(u0,fiber(t)[i],s,k))
    end
    return out
end

restwaveneumann(u0,t,k=r2rspace(points(u0))) = idct(dct(u0).*restwavemultiplier.(k,t))
function restwavedirichlet(u0,t,ω=r2rspace(points(u0),FFTW.RODFT10))
    FFTW.r2r((FFTW.r2r(u0, FFTW.RODFT10)/prod(2.0.*size(u0))).*restwavemultiplier.(ω,t),FFTW.RODFT01)
end

function waveneumann(u0,u1,t,k=r2rspace(points(u0)))
    wm = wavemultiplier.(k,t); wm[1] = 0
    idct(dct(u0).*restwavemultiplier.(k,t)+dct(u1).*wm)
end
function fullwaveneumann(u0,u1,t,k=r2rspace(points(u0)))
    wm = wavemultiplier.(k,t); wm[1] = t
    idct(dct(u0).*restwavemultiplier.(k,t)+dct(u1).*wm)
end
function wavedirichlet(u0,u1,t,ω=r2rspace(points(u0),FFTW.RODFT10))
    A = (FFTW.r2r(u0, FFTW.RODFT10)/prod(2.0.*size(u0))).*restwavemultiplier.(ω,t)
    B = (FFTW.r2r(u1, FFTW.RODFT10)/prod(2.0.*size(u1))).*wavemultiplier.(ω,t)
    FFTW.r2r(A+B,FFTW.RODFT01)
end
wavedirichlet(u0,u1,t::AbstractVector,k=r2rspace(points(u0),FFTW.RODFT10)) = wavedirichlet(u0,u1,TensorField(t),k)
function wavedirichlet(u0,u1,t::TensorField,k=r2rspace(points(u0),FFTW.RODFT10))
    data = zeros(size(u0)...,length(t))
    out = TensorField(base(u0)⊕base(t),data)
    Adapode.assign!(out,1,u0)
    for i in 2:length(t)
        Adapode.assign!(out,i,wavedirichlet(u0,u1,fiber(t)[i],k))
    end
    return out
end

for fun ∈ (:heat,:restwave,:biharmonic)
    nfun,pfun,dfun = Symbol(fun,:neumann),Symbol(fun,:periodic),Symbol(fun,:dirichlet)
    @eval begin
        $dfun(u0,t::AbstractVector,k=r2rspace(points(u0),FFTW.RODFT10)) = $dfun(u0,TensorField(t),k)
        function $dfun(u0,t::TensorField,k=r2rspace(points(u0),FFTW.RODFT10))
            data = zeros(size(u0)...,length(t))
            out = TensorField(base(u0)⊕base(t),data)
            Adapode.assign!(out,1,u0)
            for i in 2:length(t)
                Adapode.assign!(out,i,$dfun(u0,fiber(t)[i],k))
            end
            return out
        end
    end
end

end # module
