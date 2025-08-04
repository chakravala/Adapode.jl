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

Makie.lines!(ϕ::FlowIntegral,xi::Vector{<:Chain};args...) = Makie.lines(ϕ,xi,Makie.lines!;args...)
Makie.lines!(X::Function,xi::Vector{<:Chain},t::Float64=1.0;args...) = Makie.lines(FlowIntegral(X,t),xi,Makie.lines!;args...)
Makie.lines!(X::VectorField,xi::Vector{<:Chain},t::Float64=1.0;args...) = Makie.lines(FlowIntegral(X,t),xi,Makie.lines!;args...)
Makie.lines(X::Function,xi::Vector{<:Chain},t::Float64=1.0,lines=Makie.lines;args...) = Makie.lines(FlowIntegral(X,t),xi,lines;args...)
Makie.lines(X::VectorField,xi::Vector{<:Chain},t::Float64=1.0,lines=Makie.lines;args...) = Makie.lines(FlowIntegral(X,t),xi,lines;args...)
function Makie.lines(ϕ::FlowIntegral,xi::Vector{<:Chain},lines=Makie.lines;args...)
    display(lines(ϕ(xi[1]),args...))
    for i ∈ 2:length(xi)
        Makie.lines!(ϕ(xi[i]),args...)
    end
end
Makie.lines!(ϕ::Flow,t::AbstractCurve,n::Int=7;args...) = Makie.lines(ϕ,t,n,Makie.lines!;args...)
Makie.lines!(X::Function,t::AbstractCurve,n::Int=7;args...) = Makie.lines(X,t,n,Makie.lines!;args...)
Makie.lines!(X::VectorField,t::AbstractCurve,n::Int=7;args...) = Makie.lines(X,t,n,Makie.lines!;args...)
Makie.lines(X::Function,t::AbstractCurve,n::Int=7,lines=Makie.lines;args...) = Makie.lines(Flow(X,0.2),t,n,lines;args...)
Makie.lines(X::VectorField,t::AbstractCurve,n::Int=7,lines=Makie.lines;args...) = Makie.lines(Flow(X,0.2),t,n,lines;args...)
function Makie.lines(ϕ::Flow,t::AbstractCurve,n::Int=7,lines=Makie.lines;args...)
    display(lines(t,args...))
    ϕt = t
    for i ∈ 1:n
        ϕt = ϕ(ϕt)
        Makie.lines!(fiber(ϕt),args...)
    end
end

end # module
