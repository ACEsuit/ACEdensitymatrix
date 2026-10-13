## Global horizontal-lift extrapolation with onsite ACE descriptors

# Stack the onsite (l,0) descriptors for all atoms, without block weights
# A single coefficient vector solves
#     sum_i alpha_i (B_i - B_ref) ≈ B_test - B_ref
# and predicts the entire horizontal lift as
#     gamma_pred = sum_i alpha_i grassmann_log(C_i, C_ref)
# Shared scalar coefficients preserve C_ref' * gamma_pred = 0
# Apply grassmann_exp directly, without projecting gamma_pred
# q counts q-1 interpolation frames plus the last frame as reference

module ACEGammaExtrapolation_Global

using ACEdensitymatrix
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "EXT_utils.jl"))

export ACEGammaExtModel_Global, extrapolate!, design_mat, gram_mat, training_size, frame2meta

struct AtomBlock
    atom_idx::Int
    z::Int
    basis_idxs::Vector{Int}
end

mutable struct ACEGammaExtModel_Global
    onsite_models::Dict{Int,AbstractModel}
    atom_blocks::Vector{AtomBlock}
    coeffs::Vector{Float64}
    trained::Bool
end

training_size(model::ACEGammaExtModel_Global) = model.trained ? length(model.coeffs) : "model has not been trained"

function ACEGammaExtModel_Global(model_spec; coupling_backend=:new)
    species_dict = model_spec["species"]
    species = sort(collect(keys(species_dict)))
    onsite_models = Dict{Int,AbstractModel}()
    for z in species
        spec = species_dict[z]
        rcut = get(spec, "rcut_on", spec["rcut"])
        onsite_models[z] = On_Model(spec["maxdeg"], spec["ord"], rcut, z, species,
            length(spec["n_orbs"]) - 1, spec["n_orbs"]; coupling_backend=coupling_backend, fixed_particle_number=true)
    end

    layout = ao_block_labels(model_spec["ao_labels"])
    atom_blocks = AtomBlock[]
    for (atom_idx, z) in enumerate(model_spec["atomic_numbers"])
        l1max, l2max = get_L(onsite_models[z])
        ls = sort(unique(layout.angular_momenta[layout.atom_ids .== atom_idx]))
        # Same (l,0) channels as the blockwise horizontal-lift experiment
        basis_idxs = [l * (l2max + 1) + 1 for l in ls]
        push!(atom_blocks, AtomBlock(atom_idx, z, basis_idxs))
    end
    return ACEGammaExtModel_Global(onsite_models, atom_blocks, Float64[], false)
end

"""Stack flattened onsite descriptors with one column per configuration"""
function basis_values(model::ACEGammaExtModel_Global, frames; parallel=false)
    atom_vals = parallel_map(model.atom_blocks; parallel=parallel) do atom
        onsite = model.onsite_models[atom.z]
        evaluator = Chain([onsite.model.layers[layer] for layer in 1:5]...)
        ps, st = Lux.setup(MersenneTwister(1234), evaluator)
        atom_filter = filter_on(get_cutoff(onsite))
        vals = map(frames) do frame
            env = get_state(frame["R"], atom.atom_idx, atom.atom_idx; atom_filter=atom_filter)
            evaluator(env, ps, st)[1]
        end
        packed = pack_basis_evaluations(vals)
        reduce(vcat, packed[atom.basis_idxs])
    end
    return reduce(vcat, atom_vals)
end

function design_mat(model::ACEGammaExtModel_Global, train_frames, ref_frame)
    basis = basis_values(model, [train_frames; [ref_frame]])
    return center_design(view(basis, :, 1:length(train_frames)), view(basis, :, size(basis, 2)))
end

gram_mat(model::ACEGammaExtModel_Global, train_frames, ref_frame; λ=0) =
    regularized_gram(design_mat(model, train_frames, ref_frame); λ=λ)

function extrapolate!(model::ACEGammaExtModel_Global, train_frames, ref_frame, test_frame;
                      nocc, λ=1e-14, parallel=false)
    isempty(train_frames) && throw(ArgumentError("at least one interpolation frame is required"))
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame, test_frame]]; parallel=parallel)
    design = center_design(view(basis, :, 1:ntrain), view(basis, :, ntrain + 1))
    target = basis[:, ntrain + 2] - basis[:, ntrain + 1]
    coeffs = fit_coeffs(design, target; λ=λ)

    origin = view(ref_frame["C"], :, 1:nocc)
    gamma_pred = zeros(Float64, size(origin))
    for (coeff, frame) in zip(coeffs, train_frames)
        gamma_pred .+= coeff .* grassmann_log(view(frame["C"], :, 1:nocc), origin)
    end
    # Measure horizontality before the exponential, without a corrective projection
    tan_res = norm(origin' * gamma_pred) / max(norm(gamma_pred), eps(Float64))
    c_pred = grassmann_exp(gamma_pred, origin)
    d_pred = c_pred * c_pred'
    sinv_sqrt = test_frame["S"]^(-0.5)
    d_ref = test_frame["D"]
    re = norm(sinv_sqrt * (d_pred - d_ref) * sinv_sqrt) / norm(sinv_sqrt * d_ref * sinv_sqrt)
    model.coeffs = coeffs
    model.trained = true
    return (re=re, d_pred=d_pred, tan_res=tan_res, design_norm=norm(design),
            descriptor_res=norm(design * coeffs - target) / max(norm(target), eps(Float64)))
end

end

using .ACEGammaExtrapolation_Global
using ACEdensitymatrix
using Statistics

nu = 2
degree = 3
q = 20
nwindows = 80
λ = 1e-14
rcut = 6.5
parallel = false
data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

last_idx = q + nwindows - 1
trajectory = TrajectoryHDF5(data_file)
info = read_info(trajectory)
nocc = div(info["nelectrons"], 2)
frames = [convert_frame(read_frame(trajectory, i); nocc=nocc) for i in 0:last_idx]
close(trajectory.file)

meta = frame2meta(first(frames))
model_spec = Dict{String,Any}(
    "species" => Dict(z => Dict{String,Any}("n_orbs" => n_orb, "maxdeg" => degree,
        "ord" => nu, "rcut" => rcut) for (z, n_orb) in zip(meta.species, meta.n_orbs)),
    "atomic_numbers" => meta.atomic_numbers, "ao_labels" => meta.ao_labels)
model = ACEGammaExtModel_Global(model_spec)

errors = Float64[]
tan_residuals = Float64[]
design_norms = Float64[]
descriptor_residuals = Float64[]
for test_idx in q:last_idx
    train_frames = frames[(test_idx - q + 1):(test_idx - 1)]
    result = extrapolate!(model, train_frames, frames[test_idx], frames[test_idx + 1];
                          nocc=nocc, λ=λ, parallel=parallel)
    push!(errors, result.re)
    push!(tan_residuals, result.tan_res)
    push!(design_norms, result.design_norm)
    push!(descriptor_residuals, result.descriptor_res)
    println("frame=$test_idx RE=$(result.re) tangent_residual=$(result.tan_res)")
end
println("Global horizontal lift: nu=$nu degree=$degree q=$q mean_RE=$(mean(errors))")
println("max_tangent_residual=$(maximum(tan_residuals)) design_norm_range=$(extrema(design_norms))")
println("mean_descriptor_residual=$(mean(descriptor_residuals)) ncoeffs=$(training_size(model))")

# Save individual windows so the comparison can be revisited without rerunning
result_file = joinpath(@__DIR__, "gamma_horlift_global_nu$(nu)_degree$(degree)_q$(q)_results.csv")
open(result_file, "w") do io
    println(io, "frame,re,tangent_residual,design_norm,descriptor_residual")
    for (i, test_idx) in enumerate(q:last_idx)
        println(io, join((test_idx, errors[i], tan_residuals[i], design_norms[i], descriptor_residuals[i]), ','))
    end
end
