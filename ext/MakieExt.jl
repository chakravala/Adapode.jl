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

for (fun,fun!) ∈ ((:(Makie.lines),:(Makie.lines!)),(:(Cartan.graylines),:(Cartan.graylines!)))
    @eval begin
        $fun(X::Function,xi::Vector{<:Chain},t=1;args...) = $fun(FlowIntegral(X,t),xi;args...)
        $fun(X::VectorField,xi::Vector{<:Chain},t=1;args...) = $fun(FlowIntegral(X,t),xi;args...)
        function $fun(ϕ::FlowIntegral,xi::Vector{<:Chain};args...)
            out = $fun(ϕ(xi[1]);args...)
            display(out)
            for i ∈ 2:length(xi)
                $fun!(ϕ(xi[i]);args...)
            end
            return out
        end

        $fun!(X::Function,xi::Vector{<:Chain},t=1;args...) = $fun!(FlowIntegral(X,t),xi;args...)
        $fun!(X::VectorField,xi::Vector{<:Chain},t=1;args...) = $fun!(FlowIntegral(X,t),xi;args...)
        function $fun!(ϕ::FlowIntegral,xi::Vector{<:Chain};args...)
            $fun!(ϕ(xi[1]);args...)
            for i ∈ 2:length(xi)
                $fun!(ϕ(xi[i]);args...)
            end
        end

        $fun(X::Function,t::AbstractCurve,n::Int=7;args...) = $fun(Flow(X,0.2),t,n;args...)
        $fun(X::VectorField,t::AbstractCurve,n::Int=7;args...) = $fun(Flow(X,0.2),t,n;args...)
        function $fun(ϕ::Flow,t::AbstractCurve,n::Int=7;args...)
            out = $fun(t;args...)
            display(out)
            ϕt = t
            for i ∈ 1:n
                ϕt = ϕ(ϕt)
                $fun!(fiber(ϕt);args...)
            end
            return out
        end

        $fun!(X::Function,t::AbstractCurve,n::Int=7;args...) = $fun!(Flow(X,0.2),t,n;args...)
        $fun!(X::VectorField,t::AbstractCurve,n::Int=7;args...) = $fun!(Flow(X,0.2),t,n;args...)
        function $fun!(ϕ::Flow,t::AbstractCurve,n::Int=7;args...)
            $fun!(t;args...)
            ϕt = t
            for i ∈ 1:n
                ϕt = ϕ(ϕt)
                $fun!(fiber(ϕt);args...)
            end
        end

        function $fun(X,t::Components,n::Int=7;args...)
            out = $fun(X,t[1],n;args...)
            display(out)
            for i ∈ 2:length(t)
                $fun!(X,t[i],n;args...)
            end
            return out
        end
        function $fun!(X,t::Components,n::Int=7;args...)
            $fun!(X,t[1],n;args...)
            for i ∈ 2:length(t)
                $fun!(X,t[i],n;args...)
            end
        end
    end
end

end # module
