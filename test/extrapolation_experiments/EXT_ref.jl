
using ACEdensitymatrix
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "EXT_utils.jl"))

mutable struct GExtModel
    𝔹::Matrix # design matrix - evaluation of the ACE basis
    gammas::Array
    gamma_ref::Matrix
    descr_ref::Vector

    function GExtModel()
        new(Matrix{Float64}(undef, 0, 0),
            Array{Float64}(undef, 0, 0, 0),
            Matrix{Float64}(undef, 0, 0),
            Vector{Float64}(undef, 0))
    end
end

"""Return the centered HCore design matrix stored by `train!`"""
design_mat(model::GExtModel) = model.𝔹

"""Return the regularized Gram matrix of the stored HCore design"""
gram_mat(model::GExtModel; λ=0) = regularized_gram(design_mat(model); λ=λ)

function train!(model::GExtModel, traj, training_set, ref_index, c0)
    gammas = []

    frame_ref = read_frame(traj, ref_index)
    model.descr_ref = vec(frame_ref["Core Hamiltonian"])
    s = frame_ref["Overlap"]
    c_ = frame_ref["Coefficients"]'
    sqrt_s = s^0.5
    c = sqrt_s * c_
    model.gamma_ref = grassmann_log(c, c0)

    training_frames = [read_frame(traj, i) for i in training_set]
    training_descrs = [
        vec(frame["Core Hamiltonian"]) for frame in training_frames
    ]
    model.𝔹 = center_design(hcat(training_descrs...), model.descr_ref)

    for ii in eachindex(training_frames)
        frame_i = training_frames[ii]
        s = frame_i["Overlap"]

        c_ = frame_i["Coefficients"]'
        c = s^0.5 * c_
        gamma = grassmann_log(c, c0)
        push!(gammas, gamma)

    end
    model.gammas = gammas
    return training_descrs
end

function test(model::GExtModel, traj, training_descrs, test_set, c0; λ=(1.1e-5)^2)
    errors = []
    q = length(training_descrs)
    for i in test_set
        frame_i = read_frame(traj, i)
        descr_i = vec(frame_i["Core Hamiltonian"])
        target = descr_i - model.descr_ref
        a = fit_coeffs(design_mat(model), target; λ=λ)

        guess_gamma = (1.0 - sum(a))*copy(model.gamma_ref)
        for j in 1:q
            guess_gamma += a[j]*model.gammas[j]
        end

        s = frame_i["Overlap"]
        guess_c = grassmann_exp(guess_gamma, c0)
        guess_c_ = s^-0.5 * guess_c
        guess_d = guess_c_ * guess_c_'

        c = frame_i["Coefficients"]'
        d = c * c'

        error = norm(guess_d - d)/norm(d)
        push!(errors, error)
    end
    return errors
end

function main()
qmax = 20
frame_max = 100

traj = TrajectoryHDF5("data/new_datasets/oxirane.h5")
frame_ref = read_frame(traj, 0)
s0 = frame_ref["Overlap"]
c0_ = frame_ref["Coefficients"]'
c0 = s0^0.5 * c0_

qset = 20:5:qmax
errors =zeros(length(qset))
frame_errors = [Float64[] for _ in qset]

for (k,q) in enumerate(qset)
    for frame_no in qmax:frame_max-1
        # start = 0
        training_set = frame_no-q:frame_no-2
        ref_index = frame_no-1
        test_set = frame_no:frame_no
        # tangent point for the Grassmann mappings
        
        model = GExtModel()

        training_descrs = train!(model, traj, training_set, ref_index, c0)
        error = test(model, traj, training_descrs, test_set, c0)[1]
        @show error
        push!(frame_errors[k], error)
    end
    errors[k] = mean(frame_errors[k])
end

@eval using Plots
Plots.plot(qset, errors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
