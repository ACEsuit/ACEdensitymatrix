module DensityTransformations

include(joinpath(@__DIR__, "EXT_utils.jl"))

export AbstractTransform, FullTangent, HorLiftTangent, set_tangent!, backward_transform, forward_transform

abstract type AbstractTransform end

mutable struct HorLiftTangent <: AbstractTransform
    nocc::Integer
    c0::Matrix
    function HorLiftTangent(nocc::Integer)
        new(nocc, Matrix{Float64}(undef, 0, 0))
    end
end

mutable struct FullTangent <: AbstractTransform
    nocc::Integer
    c0::Matrix
    function FullTangent(nocc::Integer)
        new(nocc, Matrix{Float64}(undef, 0, 0))
    end
end

function set_tangent!(transform::HorLiftTangent, c0_::AbstractMatrix; s0::Union{AbstractMatrix,Nothing}=nothing)
    if (s0 == nothing)
        transform.c0 = c0_[:,1:transform.nocc]
    else
        transform.c0 = s0^0.5 * c0_[:,1:transform.nocc]
    end
end

function forward_transform(transform::HorLiftTangent, c_::AbstractMatrix; s::Union{AbstractMatrix,Nothing}=nothing)
    if (s == nothing)
        c = c_[:,1:transform.nocc]
    else
        c = s^0.5 * c_[:,1:transform.nocc]
    end
    gamma_hor = grassmann_log(c, transform.c0)
    return gamma_hor
end

function backward_transform(transform::HorLiftTangent, gamma_hor::AbstractMatrix; s::Union{AbstractMatrix,Nothing}=nothing)
    c = grassmann_exp(gamma_hor, transform.c0)
    if (s == nothing)
        c_ = c
    else
        c_ = s^-0.5 * c
    end
    return c_
end

function set_tangent!(transform::FullTangent, c0_::AbstractMatrix; s0::Union{AbstractMatrix,Nothing}=nothing)
    if (s0 == nothing)
        transform.c0 = c0_
    else
        transform.c0 = s0^0.5 * c0_
    end
end

function forward_transform(transform::FullTangent, c_::AbstractMatrix; s::Union{AbstractMatrix,Nothing}=nothing)
    if (s == nothing)
        c = c_[:,1:transform.nocc]
    else
        c = s^0.5 * c_[:,1:transform.nocc]
    end
    gamma_hor = grassmann_log(c, transform.c0[:,1:transform.nocc])
    cvir = transform.c0[:,transform.nocc+1:end]
    cocc = transform.c0[:,1:transform.nocc]
    gamma = cvir*cvir'*gamma_hor*cocc' + cocc*gamma_hor'*cvir*cvir'

    #d0 = transform.c0[:,1:transform.nocc] * transform.c0[:,1:transform.nocc]'
    #@show norm(gamma), norm(d0*gamma + gamma*d0 - gamma)
    return gamma
end

function backward_transform(transform::FullTangent, gamma::AbstractMatrix; s::Union{AbstractMatrix,Nothing}=nothing)
    cocc = transform.c0[:,1:transform.nocc]
    gamma_hor = gamma*cocc
    c = grassmann_exp(gamma_hor, transform.c0[:,1:transform.nocc])
    if (s == nothing)
        c_ = c
    else
        c_ = s^-0.5 * c
    end
    return c_
end

end
