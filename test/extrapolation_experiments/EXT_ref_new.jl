module RefExtrapolation

using ACEdensitymatrix
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "EXT_utils.jl"))
using .EXTUtils

include(joinpath(@__DIR__, "transformations.jl"))
using .DensityTransformations

export extrapolate!

function extrapolate!(train_frames, ref_frame, test_frame; λ=1e-14)
    nocc = Int(sum(test_frame["atomic_numbers"]) / 2)
    transform = HorLiftTangent(nocc)
    set_tangent!(transform, ref_frame["C"])


    descr_ref = vec(ref_frame["H"])
    c = ref_frame["C"]
    gamma_ref = forward_transform(transform, c)

    training_descrs = [
        vec(frame["H"]) for frame in train_frames
    ]
    B = center_design(hcat(training_descrs...), descr_ref)

    gammas = []
    for frame in train_frames
        gamma = forward_transform(transform, frame["C"])
        push!(gammas, gamma)
    end

    descr_target = vec(test_frame["H"]) - descr_ref
    a = fit_coeffs(B, descr_target; λ=λ)
    guess_gamma = (1.0 - sum(a))*copy(gamma_ref)
    for (i, frame) in enumerate(train_frames)
        gamma = forward_transform(transform, frame["C"])
        guess_gamma += a[i]*gamma
    end

    c = backward_transform(transform, guess_gamma)
    d_pred = c * c'

    sinv_sqrt = test_frame["S"]^(-0.5)
    d_ref = test_frame["D"]

    d_ref_ao = sinv_sqrt * d_ref * sinv_sqrt
    diff_ao = sinv_sqrt * (d_pred - d_ref) * sinv_sqrt
    re = norm(diff_ao) / norm(d_ref_ao)
    return (re=re,)
end

end

if abspath(PROGRAM_FILE) == @__FILE__

    using ACEdensitymatrix

    using .RefExtrapolation
    using Statistics

    include(joinpath(@__DIR__, "EXT_utils.jl"))
    using .EXTUtils

    q = 20
    nwindows = 80
    λ = (1.1e-5)^2
    rcut = 6.5
    data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

    last_idx = q + nwindows - 1
    trajectory = TrajectoryHDF5(data_file)
    info = read_info(trajectory)
    nocc = div(info["nelectrons"], 2)
    frames = [convert_frame(read_frame(trajectory, i)) for i in 0:last_idx]
    close(trajectory.file)

    errors = Float64[]
    for test_idx in q:last_idx
        train_frames = frames[(test_idx - q + 1):(test_idx - 1)]
        result = extrapolate!(train_frames, frames[test_idx], frames[test_idx + 1]; λ=λ)
        println(result)
        push!(errors, result.re)
        println("frame=$test_idx RE=$(result.re)")
    end

    println(mean(errors))
end
