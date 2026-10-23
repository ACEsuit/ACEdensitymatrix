## Global direct density extrapolation with equivariant ACE descriptors

# Stack all distinct atom-pair and angular descriptors, with no block weights
# One shared coefficient vector solves
#     sum_i alpha_i (B_i - B_ref) ≈ B_test - B_ref
# and predicts D_raw = D_ref + sum_i alpha_i (D_i - D_ref)
# Retract D_raw to rank nocc and report RE after the target S^(-1/2) transform
# q includes q-1 interpolation frames and the final reference frame

if !isdefined(@__MODULE__, :ACEDExtrapolation)
    Base.include(@__MODULE__, joinpath(@__DIR__, "EXT_ACE_D.jl"))
end

module ACEDExtrapolation_Global

using ACEdensitymatrix, LinearAlgebra

import ..ACEDExtrapolation
using ..ACEDExtrapolation: ACEDExtModel, frame2meta

export ACEDExtModel_Global, extrapolate!, design_mat, gram_mat, training_size, frame2meta

mutable struct ACEDExtModel_Global{M}
    local_model::M
    coeffs::Vector{Float64}
    trained::Bool
end

ACEDExtModel_Global(spec; coupling_backend=:new, radial_mode=:sample) =
    ACEDExtModel_Global(ACEDExtModel(spec; coupling_backend=coupling_backend,
        radial_mode=radial_mode), Float64[], false)

training_size(model::ACEDExtModel_Global) = model.trained ? length(model.coeffs) : "model has not been trained"

function basis_values(model::ACEDExtModel_Global, frames; parallel=false)
    blocks = ACEDExtrapolation.basis_values(model.local_model, frames; parallel=parallel)
    return reduce(vcat, [block for atom in blocks for block in atom])
end

function design_mat(model::ACEDExtModel_Global, train_frames, ref_frame)
    basis = basis_values(model, [train_frames; [ref_frame]])
    return ACEDExtrapolation.center_design(view(basis, :, 1:length(train_frames)), view(basis, :, size(basis, 2)))
end

gram_mat(model::ACEDExtModel_Global, train_frames, ref_frame; λ=0) =
    ACEDExtrapolation.regularized_gram(design_mat(model, train_frames, ref_frame); λ=λ)

function extrapolate!(model::ACEDExtModel_Global, train_frames, ref_frame, test_frame;
                      nocc=round(Int, tr(ref_frame["D"])), λ=1e-14, parallel=false)
    isempty(train_frames) && throw(ArgumentError("at least one interpolation frame is required"))
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame, test_frame]]; parallel=parallel)
    coeffs = ACEDExtrapolation.fit_coeffs(view(basis, :, 1:ntrain),
        view(basis, :, ntrain + 1), view(basis, :, ntrain + 2); λ=λ)
    d_raw = copy(ref_frame["D"])
    for (coeff, frame) in zip(coeffs, train_frames)
        d_raw .+= coeff .* (frame["D"] .- ref_frame["D"])
    end
    # Shared scalar coefficients preserve Hermiticity before the density retraction
    d_pred = ACEDExtrapolation.eigen_retraction(d_raw, nocc)
    sinv = test_frame["S"]^(-0.5)
    re = norm(sinv * (d_pred - test_frame["D"]) * sinv) / norm(sinv * test_frame["D"] * sinv)
    model.coeffs = coeffs
    model.trained = true
    return (re=re, d_pred=d_pred)
end

end

if abspath(PROGRAM_FILE) == @__FILE__

    using .ACEDExtrapolation_Global
    using ACEdensitymatrix, Statistics

    nu = 4
    degree = 3
    q = 20
    nwindows = 80
    λ = 6e-8
    rcut = 6.5
    zcut = 10.0
    parallel = false
    data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

    trajectory = TrajectoryHDF5(data_file)
    nocc = div(read_info(trajectory)["nelectrons"], 2)
    frames = [convert_frame(read_frame(trajectory, i)) for i in 0:(q + nwindows - 1)]
    close(trajectory.file)
    meta = frame2meta(first(frames))
    model_spec = Dict{String,Any}(
        "species" => Dict(z => Dict{String,Any}("n_orbs" => n, "ord" => nu,
            "maxdeg" => degree, "rcut" => rcut, "zcut" => zcut) for (z, n) in zip(meta.species, meta.n_orbs)),
        "atomic_numbers" => meta.atomic_numbers, "ao_labels" => meta.ao_labels)
    model = ACEDExtModel_Global(model_spec)

    errors = Float64[]
    for test_col in (q + 1):length(frames)
        train_frames = frames[(test_col - q):(test_col - 2)]
        result = extrapolate!(model, train_frames, frames[test_col - 1], frames[test_col];
                              nocc=nocc, λ=λ, parallel=parallel)
        push!(errors, result.re)
        println("frame=$(test_col - 1) RE=$(result.re)")
    end
    println("Global D: nu=$nu degree=$degree q=$q lambda=$λ mean_RE=$(mean(errors))")

end
