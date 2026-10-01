
include("../../src/utils/hdf5.jl")
using LinearAlgebra

"""Grassmann logarithm."""
function grassmann_log(c, c0)
    psi, s, r = svd(c' * c0, full=false)
    cstar = c * psi * r'
    L = (I(size(c, 1)) - c0 * c0') * cstar
    u, s, v = svd(L, full=false)
    arcsin_s = diagm(asin.(s))
    return u * arcsin_s * v'
end

"""Grassmann exponential."""
function grassmann_exp(gamma, c0)
    q, s, v = svd(gamma, full=false)
    sin_s = diagm(sin.(s))
    cos_s = diagm(cos.(s))
    return c0 * v * cos_s * v' + q * sin_s * v'
end

mutable struct GExtModel
    A::Matrix
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

mutable struct NewModel
    A::Matrix
    Ds::Array
    D_ref::Matrix
    B_ref::Vector
    𝔹::Matrix

    function NewModel()
        new(Matrix{Float64}(undef, 0, 0),
            Array{Float64}(undef, 0, 0, 0),
            Matrix{Float64}(undef, 0, 0),
            Vector{Float64}(undef, 0),
            Matrix{Float64}{undef, 0, 0} )
    end
end

function train!(model::GExtModel, traj, training_set, ref_index, c0; eps=1e-4)
    q = length(training_set)
    gammas = []

    frame_ref = read_frame(traj, ref_index)
    model.descr_ref = vec(frame_ref["Core Hamiltonian"])
    s = frame_ref["Overlap"]
    c_ = frame_ref["Coefficients"]'
    sqrt_s = s^0.5
    c = sqrt_s * c_
    model.gamma_ref = grassmann_log(c, c0)

    model.A = similar(model.descr_ref, (q, q))

    for (ii, i) in enumerate(training_set)
        frame_i = read_frame(traj, i)
        s = frame_i["Overlap"]
        descr_i = vec(frame_i["Core Hamiltonian"])

        for (jj, j) in enumerate(training_set)
            frame_j = read_frame(traj, j)
            descr_j = vec(frame_j["Core Hamiltonian"])
            model.A[ii, jj] = dot(descr_i - model.descr_ref, descr_j - model.descr_ref)
        end
        model.A[ii, ii] += eps^2

        c_ = frame_i["Coefficients"]'
        c = s^0.5 * c_
        gamma = grassmann_log(c, c0)
        push!(gammas, gamma)

    end
    model.gammas = gammas
end

function train_new!(model::GExtModel, traj, training_set, ref_index, c0; eps=1e-4)
    q = length(training_set)
    gammas = []

    frame_ref = read_frame(traj, ref_index)
    model.descr_ref = vec(frame_ref["Core Hamiltonian"])
    s = frame_ref["Overlap"]
    c_ = frame_ref["Coefficients"]'
    sqrt_s = s^0.5
    c = sqrt_s * c_
    model.gamma_ref = grassmann_log(c, c0)

    model.A = similar(model.descr_ref, (q, q))

    for (ii, i) in enumerate(training_set)
        frame_i = read_frame(traj, i)
        s = frame_i["Overlap"]
        descr_i = vec(frame_i["Core Hamiltonian"]) # This is to be changed: replaced with the evaluation of the ace basis

        for (jj, j) in enumerate(training_set)
            frame_j = read_frame(traj, j)
            descr_j = vec(frame_j["Core Hamiltonian"]) # Same as above
            model.A[ii, jj] = dot(descr_i - model.descr_ref, descr_j - model.descr_ref)
        end
        model.A[ii, ii] += eps^2

        c_ = frame_i["Coefficients"]'
        c = s^0.5 * c_
        gamma = grassmann_log(c, c0)
        push!(gammas, gamma)

    end
    model.gammas = gammas
end

function test(model::GExtModel, traj, training_set, ref_index, test_set, c0)
    errors = []
    q = length(training_set)
    for i in test_set
        frame_i = read_frame(traj, i)
        descr_i = vec(frame_i["Core Hamiltonian"])
        b = similar(descr_i, q)

        for (jj, j) in enumerate(training_set)
            frame_j = read_frame(traj, j)
            descr_j = vec(frame_j["Core Hamiltonian"])
            b[jj] = dot(descr_i - model.descr_ref, descr_j - model.descr_ref)
        end

        a = model.A \ b

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
info = read_info(traj)
frame_ref = read_frame(traj, 0)
s0 = frame_ref["Overlap"]
c0_ = frame_ref["Coefficients"]'
c0 = s0^0.5 * c0_

qset = 20:5:qmax
errors =zeros(length(qset))

for (k,q) in enumerate(qset)
    for frame_no in qmax+1:frame_max
        # start = 0
        training_set = frame_no-q-1:frame_no-2
        ref_index = frame_no-1
        test_set = frame_no:frame_no
        # tangent point for the Grassmann mappings
        
        model = GExtModel()

        train!(model, traj, training_set, ref_index, c0)
        error = test(model, traj, training_set, ref_index, test_set, c0)[1]
        @show error
        errors[k] += error
    end
end

errors = errors ./ (frame_max - 1 - qmax)

@eval using Plots
Plots.plot(qset, errors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
