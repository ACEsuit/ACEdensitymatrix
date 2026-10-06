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
    ace_model = Density_Model(model_spec["species"]; coupling_backend=coupling_backend)
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

function extrapolate!(model::ACEDExtModel, train_frames, ref_frame, test_frame; λ=1e-14)
    isempty(train_frames) && throw(ArgumentError("at least one training frame is required"))
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame, test_frame]])
    coeff_sets = Vector{Vector{Vector{Float64}}}(undef, length(model.atom_blocks))
    d_raw = zeros(Float64, size(test_frame["D"]))

    for (atom_idx, (atom, blocks)) in enumerate(zip(model.atom_blocks, basis))
        d_train = [get_block(frame["D"], atom.atom_i, atom.atom_j, frame["ao_labels"])
                   for frame in train_frames]
        d_ref = get_block(ref_frame["D"], atom.atom_i, atom.atom_j, ref_frame["ao_labels"])
        d_block = zeros(Float64, length(atom.pos_i), length(atom.pos_j))
        coeff_sets[atom_idx] = Vector{Float64}[]

        for ao in atom.ao_blocks
            block = blocks[ao.basis_idx]
            coeffs = fit_coeffs(view(block, :, 1:ntrain), view(block, :, ntrain + 1),
                                view(block, :, ntrain + 2); λ=λ)
            push!(coeff_sets[atom_idx], coeffs)

            ref_target = view(d_ref, ao.rows, ao.cols)
            dest = view(d_block, ao.rows, ao.cols)
            dest .= ref_target
            for (coeff, d) in zip(coeffs, d_train)
                dest .+= coeff .* (view(d, ao.rows, ao.cols) .- ref_target)
            end
        end
        d_raw[atom.pos_i, atom.pos_j] .= d_block
    end

    complete_hermitian!(d_raw, model.atom_blocks, length(test_frame["atomic_numbers"]))
    for (atom, coeffs) in zip(model.atom_blocks, coeff_sets)
        for (ao, block_coeffs) in zip(atom.ao_blocks, coeffs)
            ao.coeffs = block_coeffs
        end
    end
    model.trained = true

    nocc = Int(sum(test_frame["atomic_numbers"]) / 2)
    d_pred = eigen_retraction(d_raw, nocc)
    sinv_sqrt = test_frame["S"]^(-0.5)
    d_ref = test_frame["D"]
    d_ref_ao = sinv_sqrt * d_ref * sinv_sqrt
    diff_ao = sinv_sqrt * (d_pred - d_ref) * sinv_sqrt
    re = norm(diff_ao) / norm(d_ref_ao)
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
close(trajectory.file)

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
                          frames[test_idx + 1]; λ=λ)
    push!(errors, result.re)
    println("frame=$test_idx RE=$(result.re)")
end

mean_re = mean(errors)
println("EXT_ACE_D, nu=$nu, degree=$degree q=$q mean_RE=$mean_re")
