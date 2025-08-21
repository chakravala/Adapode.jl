module MakieExt

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
isdefined(Adapode, :Requires) ? (import Adapode: Makie) : (using Makie)

Makie.lines(X::Function,xi::Vector{<:Chain},t=1;args...) = Makie.lines(FlowIntegral(X,t),xi;args...)
Makie.lines(X::VectorField,xi::Vector{<:Chain},t=1;args...) = Makie.lines(FlowIntegral(X,t),xi;args...)
function Makie.lines(ϕ::FlowIntegral,xi::Vector{<:Chain};args...)
    out = Makie.lines(ϕ(xi[1]),args...)
    display(out)
    for i ∈ 2:length(xi)
        Makie.lines!(ϕ(xi[i]),args...)
    end
    return out
end

Makie.lines!(X::Function,xi::Vector{<:Chain},t=1;args...) = Makie.lines(FlowIntegral(X,t),xi;args...)
Makie.lines!(X::VectorField,xi::Vector{<:Chain},t=1;args...) = Makie.lines(FlowIntegral(X,t),xi;args...)
function Makie.lines!(ϕ::FlowIntegral,xi::Vector{<:Chain};args...)
    Makie.lines!(ϕ(xi[1]),args...)
    for i ∈ 2:length(xi)
        Makie.lines!(ϕ(xi[i]),args...)
    end
end

Makie.lines(X::Function,t::AbstractCurve,n::Int=7;args...) = Makie.lines(Flow(X,0.2),t,n;args...)
Makie.lines(X::VectorField,t::AbstractCurve,n::Int=7;args...) = Makie.lines(Flow(X,0.2),t,n;args...)
function Makie.lines(ϕ::Flow,t::AbstractCurve,n::Int=7;args...)
    out = Makie.lines(t,args...)
    display(out)
    ϕt = t
    for i ∈ 1:n
        ϕt = ϕ(ϕt)
        Makie.lines!(fiber(ϕt),args...)
    end
    return out
end

Makie.lines!(X::Function,t::AbstractCurve,n::Int=7;args...) = Makie.lines(X,t,n;args...)
Makie.lines!(X::VectorField,t::AbstractCurve,n::Int=7;args...) = Makie.lines(X,t,n;args...)
function Makie.lines!(ϕ::Flow,t::AbstractCurve,n::Int=7;args...)
    Makie.lines!(t,args...)
    ϕt = t
    for i ∈ 1:n
        ϕt = ϕ(ϕt)
        Makie.lines!(fiber(ϕt),args...)
    end
end

function Makie.lines(X,t::Components,n::Int=7;args...)
    out = Makie.lines(X,t[1],n;args...)
    display(out)
    for i ∈ 2:length(t)
        Makie.lines!(X,t[i],n;args...)
    end
    return out
end
function Makie.lines!(X,t::Components,n::Int=7;args...)
    Makie.lines!(X,t[1],n;args...)
    for i ∈ 2:length(t)
        Makie.lines!(X,t[i],n;args...)
    end
end

end # module
