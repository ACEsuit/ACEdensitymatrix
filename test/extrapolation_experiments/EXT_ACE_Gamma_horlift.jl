## Grassmann-tangent extrapolation with equivariant ACE descriptors

# This script predicts the density matrix of a target configuration from
# the `q` immediately preceding configurations using the flattened
# (`nu`, `degree`) equivariant ACE basis
# The last preceding frame is always the reference, so `q` consists of `q-1`
# interpolation frames plus one final reference frame

# Gamma has one AO-like row index and one occupied-orbital column index
# Consequently, the row block of angular momentum `l` transforms as `(l,0)`
# Only onsite environments are needed and every exact atom and `l` row block
# is fitted independently as
#     sum_i alpha_i (B_i - B_ref) ≈ B_test - B_ref
# then
#     Gamma_pred = sum_i alpha_i Gamma_i

# The predicted Gamma is projected back into the reference tangent space,
# mapped to the Grassmann manifold with the exponential map, and converted to
#     D_pred = C_pred * C_pred'

module ACEGammaExtrapolation

using ACEdensitymatrix
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "EXT_utils.jl"))

export ACEGammaExtModel, design_mats, gram_mats, extrapolate!, training_size, frame2meta

mutable struct AOBlock{T}
    l::Int
    rows::Vector{Int}
    basis_idx::Int
    coeffs::Vector{T}
end

struct AtomBlock{B}
    atom_idx::Int
    ao_blocks::Vector{B}
end

mutable struct ACEGammaExtModel{T,B}
    onsite_models::Dict{T,AbstractModel}
    atom_blocks::Vector{B}
    trained::Bool
end

training_size(model::ACEGammaExtModel) = model.trained ?
    length(model.atom_blocks[1].ao_blocks[1].coeffs) : "model has not been trained"

function ACEGammaExtModel(model_spec; coupling_backend=:new)
    species_dict = model_spec["species"]
    species = sort(collect(keys(species_dict)))
    t = eltype(species)
    onsite_models = Dict{t,AbstractModel}()

    for z in species
        spec = species_dict[z]
        cutoff = haskey(spec, "rcut_on") ? spec["rcut_on"] : spec["rcut"]
        onsite_models[z] = On_Model(spec["maxdeg"], spec["ord"], cutoff, z, species,
                                    length(spec["n_orbs"]) - 1, spec["n_orbs"];
                                    coupling_backend=coupling_backend)
    end

    z = model_spec["atomic_numbers"]
    labels = model_spec["ao_labels"]
    layout = ao_block_labels(labels)
    atom_ids = layout.atom_ids
    ls = layout.angular_momenta
    atom_blocks = AtomBlock{AOBlock{Float64}}[]

    for atom_idx in eachindex(z)
        onsite = onsite_models[z[atom_idx]]
        l1max, l2max = get_L(onsite)
        l_blocks = [(l1, l2) for l1 in 0:l1max for l2 in 0:l2max]
        atom_ls = sort(unique(ls[atom_ids .== atom_idx]))
        ao_blocks = AOBlock{Float64}[]

        for l in atom_ls
            basis_idx = findfirst(==((l, 0)), l_blocks)
            isnothing(basis_idx) && error("onsite model for atom $atom_idx has no (l,0) block")
            rows = findall((atom_ids .== atom_idx) .& (ls .== l))
            push!(ao_blocks, AOBlock(l, rows, basis_idx, Float64[]))
        end
        push!(atom_blocks, AtomBlock(atom_idx, ao_blocks))
    end
    return ACEGammaExtModel(onsite_models, atom_blocks, false)
end

function basis_values(model::ACEGammaExtModel, frames)
    z = first(frames)["atomic_numbers"]

    return map(model.atom_blocks) do atom
        onsite = model.onsite_models[z[atom.atom_idx]]
        evaluator = Chain([onsite.model.layers[layer] for layer in 1:5]...)
        ps, st = Lux.setup(MersenneTwister(1234), evaluator)
        atom_filter = filter_on(get_cutoff(onsite))
        vals = map(frames) do frame
            env = get_state(frame["R"], atom.atom_idx, atom.atom_idx;
                            atom_filter=atom_filter)
            evaluator(env, ps, st)[1]
        end
        basis = pack_basis_evaluations(vals)
        maximum(ao.basis_idx for ao in atom.ao_blocks) <= length(basis) ||
            error("basis and AO-block counts disagree for atom $(atom.atom_idx)")
        basis
    end
end

function design_mats(model::ACEGammaExtModel, train_frames, ref_frame)
    isempty(train_frames) && throw(ArgumentError("at least one interpolation frame is required"))
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame]])
    return [center_design(view(blocks[ao.basis_idx], :, 1:ntrain),
                          view(blocks[ao.basis_idx], :, ntrain + 1))
            for (atom, blocks) in zip(model.atom_blocks, basis) for ao in atom.ao_blocks]
end

gram_mats(model::ACEGammaExtModel, train_frames, ref_frame; λ=0) =
    [regularized_gram(design; λ=λ) for design in design_mats(model, train_frames, ref_frame)]

function extrapolate!(model::ACEGammaExtModel, train_frames, ref_frame, test_frame; λ=1e-14)
    isempty(train_frames) && throw(ArgumentError("at least one interpolation frame is required"))
    origin = ref_frame["C"]
    gammas = [grassmann_log(frame["C"], origin) for frame in train_frames]
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame, test_frame]])
    coeff_sets = Vector{Vector{Vector{Float64}}}(undef, length(model.atom_blocks))
    gamma_pred = zeros(Float64, size(origin))

    for (atom_idx, (atom, blocks)) in enumerate(zip(model.atom_blocks, basis))
        coeff_sets[atom_idx] = Vector{Float64}[]
        for ao in atom.ao_blocks
            block = blocks[ao.basis_idx]
            coeffs = fit_coeffs(view(block, :, 1:ntrain), view(block, :, ntrain + 1),
                                view(block, :, ntrain + 2); λ=λ)
            push!(coeff_sets[atom_idx], coeffs)

            dest = view(gamma_pred, ao.rows, :)
            for (coeff, gamma) in zip(coeffs, gammas)
                dest .+= coeff .* view(gamma, ao.rows, :)
            end
        end
    end

    for (atom, coeffs) in zip(model.atom_blocks, coeff_sets)
        for (ao, block_coeffs) in zip(atom.ao_blocks, coeffs)
            ao.coeffs = block_coeffs
        end
    end
    model.trained = true

    gamma_tan = gamma_pred - origin * (origin' * gamma_pred)
    c_pred = grassmann_exp(gamma_tan, origin)
    d_pred = c_pred * c_pred'
    sinv_sqrt = test_frame["S"]^(-0.5)
    d_ref = test_frame["D"]
    d_ref_ao = sinv_sqrt * d_ref * sinv_sqrt
    diff_ao = sinv_sqrt * (d_pred - d_ref) * sinv_sqrt
    re = norm(diff_ao) / norm(d_ref_ao)
    return (re=re, d_pred=d_pred)
end

end

using .ACEGammaExtrapolation
using ACEdensitymatrix
using Statistics

data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

nu = 2
degree = 3
q = 20
nwindows = 80
λ = 1e-14
rcut = 6.5

last_idx = q + nwindows - 1
trajectory = TrajectoryHDF5(data_file)
frames = [convert_frame(read_frame(trajectory, i)) for i in 0:last_idx]
close(trajectory.file)

meta = frame2meta(first(frames))
model_spec = Dict{String,Any}(
    "species" => Dict(
        z => Dict{String,Any}(
            "n_orbs" => n_orb,
            "maxdeg" => degree,
            "ord" => nu,
            "rcut" => rcut
        ) for (z, n_orb) in zip(meta.species, meta.n_orbs)
    ),
    "atomic_numbers" => meta.atomic_numbers,
    "ao_labels" => meta.ao_labels
)
model = ACEGammaExtModel(model_spec; coupling_backend=:new)

errors = Float64[]
for test_idx in q:last_idx
    train_idxs = (test_idx - q):(test_idx - 2)
    ref_idx = test_idx - 1
    train_frames = [frames[i + 1] for i in train_idxs]
    result = extrapolate!(model, train_frames, frames[ref_idx + 1],
                          frames[test_idx + 1]; λ=λ)
    push!(errors, result.re)
    println("frame=$test_idx RE=$(result.re)")
end

mean_re = mean(errors)
println("EXT_ACE_Gamma, nu=$nu, degree=$degree q=$q mean_RE=$mean_re")
