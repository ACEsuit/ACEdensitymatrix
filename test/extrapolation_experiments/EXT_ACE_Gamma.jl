## Direct density extrapolation with equivariant ACE descriptors

# This script predicts the density matrix of a target configuration from
# the `q` immediately preceding configurations using the flattened
# (`nu`, `degree`) equivariant ACE basis
# The last preceding frame is always the reference, so `q` consists of `q-1`
# interpolation frames plus one final reference frame

# Every exact atom pair and every `(l1,l2)` angular block is fitted
# independently as
#     sum_i alpha_i (B_i - B_ref) ≈ B_test - B_ref
# then
#     D_pred = D_ref + sum_i alpha_i (D_i - D_ref)

# The predicted blocks are assembled into the full density matrix, completed
# by Hermiticity, and retracted to an idempotent density with the correct trace
# The reported RE is evaluated in the orthonormal AO basis

module ACEDExtrapolation

using ACEdensitymatrix
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "EXT_utils.jl"))
include(joinpath(@__DIR__, "transformations.jl"))
using .DensityTransformations

export ACEDExtModel, design_mats, gram_mats, extrapolate!, training_size, frame2meta

mutable struct AOBlock{T}
    l1::Int
    l2::Int
    basis_idx::Int
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    coeffs::Vector{T}
end

struct AtomBlock{B}
    atom_i::Int
    atom_j::Int
    pos_i::Vector{Int}
    pos_j::Vector{Int}
    ao_blocks::Vector{B}
end

mutable struct ACEDExtModel{T,B}
    ace_model::Density_Model{T}
    atom_blocks::Vector{B}
    trained::Bool
end

training_size(model::ACEDExtModel) = model.trained ?
    length(model.atom_blocks[1].ao_blocks[1].coeffs) : "model has not been trained"

function atom_pairs(z)
    pairs = Tuple{Int,Int}[]
    for i in eachindex(z), j in eachindex(z)
        zi, zj = z[i], z[j]
        if i == j || zi < zj || (zi == zj && i < j)
            push!(pairs, (i, j))
        end
    end
    return pairs
end

function ACEDExtModel(model_spec; coupling_backend=:new)
    ace_model = Density_Model(model_spec["species"];
        coupling_backend=coupling_backend, fixed_particle_number=true)
    z = model_spec["atomic_numbers"]
    labels = model_spec["ao_labels"]
    atom_ids = ao_block_labels(labels).atom_ids
    atom_blocks = AtomBlock{AOBlock{Float64}}[]

    for (atom_i, atom_j) in atom_pairs(z)
        zi, zj = z[atom_i], z[atom_j]
        ace = atom_i == atom_j ? ace_model.Models[zi] : ace_model.Models[(zi, zj)]
        l1max, l2max = get_L(ace)
        norb1, norb2 = get_norbs(ace)
        ao_blocks = AOBlock{Float64}[]

        for (basis_idx, (l1, l2)) in enumerate((l1, l2) for l1 in 0:l1max for l2 in 0:l2max)
            rows = angular_orbital_range(norb1, l1)
            cols = angular_orbital_range(norb2, l2)
            push!(ao_blocks, AOBlock(l1, l2, basis_idx, rows, cols, Float64[]))
        end

        pos_i = findall(==(atom_i), atom_ids)
        pos_j = findall(==(atom_j), atom_ids)
        push!(atom_blocks, AtomBlock(atom_i, atom_j, pos_i, pos_j, ao_blocks))
    end
    return ACEDExtModel(ace_model, atom_blocks, false)
end

function basis_values(model::ACEDExtModel, frames)
    z = first(frames)["atomic_numbers"]
    rcut_on, rcut_off, zcut = get_cutoff(model.ace_model)

    return map(model.atom_blocks) do atom
        atom_i, atom_j = atom.atom_i, atom.atom_j
        zi, zj = z[atom_i], z[atom_j]
        if atom_i == atom_j
            ace = model.ace_model.Models[zi]
            atom_filter = filter_on(rcut_on)
        else
            ace = model.ace_model.Models[(zi, zj)]
            atom_filter = filter_off(rcut_off, zcut)
        end

        evaluator = Chain([ace.model.layers[layer] for layer in 1:5]...)
        ps, st = Lux.setup(MersenneTwister(1234), evaluator)
        vals = map(frames) do frame
            env = get_state(frame["R"], atom_i, atom_j; atom_filter=atom_filter)
            evaluator(env, ps, st)[1]
        end
        basis = pack_basis_evaluations(vals)
        length(basis) == length(atom.ao_blocks) ||
            error("basis and AO-block counts disagree for atoms $atom_i and $atom_j")
        basis
    end
end

function design_mats(model::ACEDExtModel, train_frames, ref_frame)
    isempty(train_frames) && throw(ArgumentError("at least one interpolation frame is required"))
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame]])
    return [center_design(view(blocks[ao.basis_idx], :, 1:ntrain),
                          view(blocks[ao.basis_idx], :, ntrain + 1))
            for (atom, blocks) in zip(model.atom_blocks, basis) for ao in atom.ao_blocks]
end

gram_mats(model::ACEDExtModel, train_frames, ref_frame; λ=0) =
    [regularized_gram(design; λ=λ) for design in design_mats(model, train_frames, ref_frame)]

function complete_hermitian!(d, atom_blocks, natoms)
    for atom in 1:natoms
        pos = only(block.pos_i for block in atom_blocks
                   if block.atom_i == atom && block.atom_j == atom)
        onsite = view(d, pos, pos)
        onsite .= (onsite .+ onsite') ./ 2
    end

    for atom_i in 1:(natoms - 1), atom_j in (atom_i + 1):natoms
        fwd = findfirst(block -> block.atom_i == atom_i && block.atom_j == atom_j, atom_blocks)
        rev = findfirst(block -> block.atom_i == atom_j && block.atom_j == atom_i, atom_blocks)
        xor(isnothing(fwd), isnothing(rev)) ||
            error("expected exactly one fitted orientation for atoms $atom_i and $atom_j")
        block = atom_blocks[something(fwd, rev)]
        d[block.pos_j, block.pos_i] .= d[block.pos_i, block.pos_j]'
    end
    return d
end

function extrapolate!(model::ACEDExtModel, train_frames, ref_frame, test_frame, transform; λ=1e-14)
    isempty(train_frames) && throw(ArgumentError("at least one training frame is required"))
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame, test_frame]])
    coeff_sets = Vector{Vector{Vector{Float64}}}(undef, length(model.atom_blocks))
    gamma_raw = zeros(Float64, size(test_frame["D"]))

    for (atom_idx, (atom, blocks)) in enumerate(zip(model.atom_blocks, basis))
        gamma_train = []
        for frame in train_frames
            # TODO: swap the loops for efficiency ?
            gamma = forward_transform(transform, frame["C"])
            gamma_block = get_block(gamma, atom.atom_i, atom.atom_j, frame["ao_labels"])

            push!(gamma_train, gamma_block)
        end

        gamma = forward_transform(transform, ref_frame["C"])
        gamma_ref = get_block(gamma, atom.atom_i, atom.atom_j, ref_frame["ao_labels"])

        gamma_block = zeros(Float64, length(atom.pos_i), length(atom.pos_j))
        coeff_sets[atom_idx] = Vector{Float64}[]

        for ao in atom.ao_blocks
            block = blocks[ao.basis_idx]
            coeffs = fit_coeffs(view(block, :, 1:ntrain), view(block, :, ntrain + 1),
                                view(block, :, ntrain + 2); λ=λ)
            push!(coeff_sets[atom_idx], coeffs)

            ref_target = view(gamma_ref, ao.rows, ao.cols)
            dest = view(gamma_block, ao.rows, ao.cols)
            dest .= ref_target
            for (coeff, gamma) in zip(coeffs, gamma_train)
                dest .+= coeff .* (view(gamma, ao.rows, ao.cols) .- ref_target)
            end
        end
        gamma_raw[atom.pos_i, atom.pos_j] .= gamma_block
    end

    complete_hermitian!(gamma_raw, model.atom_blocks, length(test_frame["atomic_numbers"]))
    for (atom, coeffs) in zip(model.atom_blocks, coeff_sets)
        for (ao, block_coeffs) in zip(atom.ao_blocks, coeffs)
            ao.coeffs = block_coeffs
        end
    end
    model.trained = true

    # DEBUG
    target_descriptor = vec(test_frame["H"])
    target_gamma = forward_transform(transform, test_frame["C"])
    gammas_train = []
    B = zeros(Float64, length(target_descriptor), length(train_frames))
    for (i, frame) in enumerate(train_frames)
        gamma = forward_transform(transform, frame["C"])
        push!(gammas_train, gamma)
        B[:,i] .= vec(frame["H"])
    end
    A = B' * B
    b = B' * target_descriptor
    coeffs = A \ b
    gamma_raw[:,:] .= 0.0
    for (i, coeff) in enumerate(coeffs)
        gamma_raw .+= coeff*gammas_train[i]
    end
    re_gamma = norm(gamma_raw - target_gamma)/norm(target_gamma)
    @show re_gamma
    # END DEBUG

    c = backward_transform(transform, gamma_raw; s=test_frame["S"])
    d_pred = c * c'

    sinv_sqrt = test_frame["S"]^(-0.5)
    c = sinv_sqrt * test_frame["C"][:,1:transform.nocc]
    d_ref = c * c'
    #d_ref = test_frame["D"]

    re = norm(d_pred - d_ref)/norm(d_ref)
    @show re

    #d_ref_ao = sinv_sqrt * d_ref * sinv_sqrt

    #diff_ao = d_pred - d_ref_ao #sinv_sqrt * (d_pred - d_ref) * sinv_sqrt
    #re = norm(diff_ao) / norm(d_ref_ao)
    #@show re
    return (re=re, d_pred=d_pred)
end

end

using .ACEDExtrapolation
using ACEdensitymatrix
using Statistics

data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

nu = 2
degree = 3
q = 20
nwindows = 80
λ = 1e-14
rcut = 6.5
zcut = 10.0

last_idx = q + nwindows - 1
trajectory = TrajectoryHDF5(data_file)
frames = [convert_frame(read_frame(trajectory, i)) for i in 0:last_idx]
info = read_info(trajectory)
close(trajectory.file)

nocc = div(info["nelectrons"], 2)
transform = ACEDExtrapolation.FullTangent(nocc)
ACEDExtrapolation.set_tangent!(transform, frames[1]["C"])

meta = frame2meta(first(frames))
model_spec = Dict{String,Any}(
    "species" => Dict(
        z => Dict{String,Any}(
            "n_orbs" => n_orb,
            "maxdeg" => degree,
            "ord" => nu,
            "rcut" => rcut,
            "zcut" => zcut
        ) for (z, n_orb) in zip(meta.species, meta.n_orbs)
    ),
    "atomic_numbers" => meta.atomic_numbers,
    "ao_labels" => meta.ao_labels
)
model = ACEDExtModel(model_spec; coupling_backend=:new)

errors = Float64[]
for test_idx in q:last_idx
    train_idxs = (test_idx - q):(test_idx - 2)
    ref_idx = test_idx - 1
    train_frames = [frames[i + 1] for i in train_idxs]
    result = extrapolate!(model, train_frames, frames[ref_idx + 1],
                          frames[test_idx + 1], transform; λ=λ)
    push!(errors, result.re)
    println("frame=$test_idx RE=$(result.re)")
end

mean_re = mean(errors)
println("EXT_ACE_D, nu=$nu, degree=$degree q=$q mean_RE=$mean_re")
