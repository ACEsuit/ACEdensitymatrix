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

using ACEdensitymatrix
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "EXT_utils.jl"))
data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

"""Training data for one exact atom-pair `(l1,l2)` density block"""
struct DensityAngularFit
    l1::Int
    l2::Int
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    train_basis::Matrix{Float64}
    ref_basis::Vector{Float64}
    train_targets::Vector{Matrix{Float64}}
    ref_target::Matrix{Float64}
end

"""ACE evaluator and angular-block fits for one oriented atom pair"""
struct DensityPairFit
    atom_i::Int
    atom_j::Int
    pos_i::Vector{Int}
    pos_j::Vector{Int}
    evaluator
    ps
    st
    atom_filter
    angular_fits::Vector{DensityAngularFit}
end

"""
State of one direct-density rolling-window problem

`pair_fits` contains a distinct fit for every selected atom pair, even when
two pairs have the same chemical species
"""
mutable struct ACEDExtModel{m}
    ace_model::m
    pair_fits::Vector{DensityPairFit}
    natoms::Int
    d_size::Tuple{Int,Int}
    nocc::Int
end

ACEDExtModel(ace_model) = ACEDExtModel(ace_model, DensityPairFit[], 0, (0, 0), 0)

"""Return centered designs in pair-fit and angular-fit order"""
function design_mats(model::ACEDExtModel)
    return [center_design(fit.train_basis, fit.ref_basis)
            for pair in model.pair_fits for fit in pair.angular_fits]
end

"""Return regularized Gram matrices in the same order as `design_mats`"""
function gram_mats(model::ACEDExtModel; λ=0)
    return [regularized_gram(design; λ=λ) for design in design_mats(model)]
end

"""Select one orientation of every exact atom pair"""
function atom_pairs(z)
    pairs = Tuple{Int,Int}[]
    for i in eachindex(z), j in eachindex(z)
        zi, zj = z[i], z[j]

        # Fit one orientation of each offsite pair
        # Species order matches the keys in Density_Model.Models
        if i == j || zi < zj || (zi == zj && i < j)
            push!(pairs, (i, j))
        end
    end
    return pairs
end

"""
Train every exact atom-pair and `(l1,l2)` block independently

`ref_frame` supplies both the descriptor and density reference
All frames must already have been processed by `convert_frame`
"""
function train!(model::ACEDExtModel, train_frames, ref_frame)
    isempty(train_frames) && throw(ArgumentError("at least one interpolation frame is required"))
    z = ref_frame["atomic_numbers"]
    labels = ref_frame["ao_labels"]
    for frame in train_frames
        frame["atomic_numbers"] == z || error("atomic numbers change between training frames")
        vec(frame["ao_labels"]) == vec(labels) || error("AO labels change between training frames")
    end

    # Global AO positions place each atom-pair block in the full density matrix
    atom_ids = ao_block_labels(labels).atom_ids
    empty!(model.pair_fits)
    model.natoms = length(z)
    model.d_size = size(ref_frame["D"])
    model.nocc = Int(sum(z) / 2)

    for (atom_i, atom_j) in atom_pairs(z)
        zi, zj = z[atom_i], z[atom_j]
        if atom_i == atom_j
            # Onsite environments are centered on one atom
            ace = model.ace_model.Models[zi]
            atom_filter = filter_on(rcut)
        else
            # Offsite environments are centered on a bond and use zcut
            ace = model.ace_model.Models[(zi, zj)]
            atom_filter = filter_off(rcut, zcut)
        end

        # The first five Lux layers output the equivariant ACE basis
        # No fitted linear readout is used as a descriptor
        evaluator = Chain([ace.model.layers[layer] for layer in 1:5]...)
        ps, st = Lux.setup(MersenneTwister(1234), evaluator)

        # Evaluate every required environment once
        # The final packed column is B_ref and earlier columns align with D_i
        desc_frames = [train_frames; [ref_frame]]
        vals = map(desc_frames) do frame
            env = get_state(frame["R"], atom_i, atom_j; atom_filter=atom_filter)
            evaluator(env, ps, st)[1]
        end
        basis = pack_basis_evaluations(vals)
        d_blocks = [get_block(frame["D"], atom_i, atom_j, frame["ao_labels"])
                    for frame in train_frames]
        d_ref_block = get_block(ref_frame["D"], atom_i, atom_j, ref_frame["ao_labels"])

        # ACE returns one matrix-valued basis vector for every `(l1,l2)` block
        # Blocks follow this nested-loop order
        l1max, l2max = get_L(ace)
        norb1, norb2 = get_norbs(ace)
        l_blocks = [(l1, l2) for l1 in 0:l1max for l2 in 0:l2max]
        length(basis) == length(l_blocks) ||
            error("basis and angular-block counts disagree for atoms $atom_i and $atom_j")

        angular_fits = DensityAngularFit[]
        ntrain = length(train_frames)
        for (block_idx, (l1, l2)) in enumerate(l_blocks)
            # Each exact atom-pair and `(l1,l2)` block gets its own coefficients
            rows = angular_orbital_range(norb1, l1)
            cols = angular_orbital_range(norb2, l2)
            block = basis[block_idx]
            train_basis = Matrix{Float64}(view(block, :, 1:ntrain))
            ref_basis = Vector{Float64}(view(block, :, ntrain + 1))
            train_targets = [Matrix{Float64}(view(d, rows, cols)) for d in d_blocks]
            ref_target = Matrix{Float64}(view(d_ref_block, rows, cols))
            push!(angular_fits, DensityAngularFit(l1, l2, rows, cols, train_basis, ref_basis,
                                                  train_targets, ref_target))
        end

        pos_i = findall(==(atom_i), atom_ids)
        pos_j = findall(==(atom_j), atom_ids)
        push!(model.pair_fits, DensityPairFit(atom_i, atom_j, pos_i, pos_j, evaluator,
                                              ps, st, atom_filter, angular_fits))
    end
    return model
end

"""Symmetrize onsite blocks and fill omitted reverse offsite blocks"""
function complete_hermitian!(d, pair_fits, natoms)
    for atom in 1:natoms
        pos = only(pair.pos_i for pair in pair_fits if pair.atom_i == atom && pair.atom_j == atom)
        block = view(d, pos, pos)
        block_copy = copy(block)

        # Independent `(l1,l2)` and `(l2,l1)` fits need not be transposes
        block .= (block_copy .+ block_copy') ./ 2
    end

    for atom_i in 1:(natoms - 1), atom_j in (atom_i + 1):natoms
        fwd = findfirst(pair -> pair.atom_i == atom_i && pair.atom_j == atom_j, pair_fits)
        rev = findfirst(pair -> pair.atom_i == atom_j && pair.atom_j == atom_i, pair_fits)
        xor(isnothing(fwd), isnothing(rev)) ||
            error("expected exactly one fitted orientation for atoms $atom_i and $atom_j")
        pair = pair_fits[something(fwd, rev)]
        d[pair.pos_j, pair.pos_i] .= d[pair.pos_i, pair.pos_j]'
    end
    return d
end

"""Evaluate one target frame and return its retracted density and errors"""
function test(model::ACEDExtModel, frame; λ=1e-14)
    # Fill the global density one atom-pair block at a time
    # Reverse offsite blocks remain zero until Hermitian completion
    d_pred = zeros(Float64, model.d_size)

    for pair in model.pair_fits
        # Reuse the same partial Lux chain and parameters used during training
        env = get_state(frame["R"], pair.atom_i, pair.atom_j; atom_filter=pair.atom_filter)
        val = pair.evaluator(env, pair.ps, pair.st)[1]
        test_basis = pack_basis_evaluations([val])
        d_block = zeros(Float64, length(pair.pos_i), length(pair.pos_j))

        for (block_idx, fit) in enumerate(pair.angular_fits)
            # Coefficients are local to one exact atom pair and one `(l1,l2)` block
            coeffs = fit_coeffs(fit.train_basis, fit.ref_basis,
                                view(test_basis[block_idx], :, 1); λ=λ)
            dest = view(d_block, fit.rows, fit.cols)
            dest .= fit.ref_target
            for (coeff, target) in zip(coeffs, fit.train_targets)
                dest .+= coeff .* (target .- fit.ref_target)
            end
        end
        d_pred[pair.pos_i, pair.pos_j] .= d_block
    end

    complete_hermitian!(d_pred, model.pair_fits, model.natoms)
    d_ref = frame["D"]
    raw_re = norm(d_pred - d_ref) / norm(d_ref)

    # Replace the spectrum by the correct occupied and unoccupied eigenvalues
    # while retaining the predicted eigenspace
    d_ret = eigen_retraction(d_pred, model.nocc)

    # Map the orthogonal-basis density difference back to the AO basis
    sinv_sqrt = frame["S"]^(-0.5)
    d_ref_ao = sinv_sqrt * d_ref * sinv_sqrt
    diff_ao = sinv_sqrt * (d_ret - d_ref) * sinv_sqrt
    re = norm(diff_ao) / norm(d_ref_ao)

    return (re=re, raw_re=raw_re, d_pred=d_pred, d_ret=d_ret)
end

"""
Run `nwindows` rolling predictions

Trajectory indices are zero-based in the printed output
For `q=20`, target 20 uses frames 0:18 for interpolation and frame 19 as the
reference, while the default run predicts targets 20:99
"""
nu = 2
degree = 3
q = 20
nwindows = 80
λ = 1e-14
rcut = 6.5
zcut = 10.0

# q > 1 || error("q must exceed one")
# nwindows > 0 || error("nwindows must be positive")
# λ >= 0 || error("λ must be nonnegative")

# Load each required frame once
# Vector position `i+1` corresponds to zero-based trajectory frame i
last_idx = q + nwindows - 1
trajectory = TrajectoryHDF5(data_file)
frames = [convert_frame(read_frame(trajectory, i)) for i in 0:last_idx]
close(trajectory.file)

# Construct the species-level ACE model once and reuse it across windows
multiplicities = infer_orbital_multiplicities(first(frames))
dictionary = Dict(
    z_atom => Dict{String,Any}(
        "n_orbs" => norbs,
        "maxdeg" => degree,
        "ord" => nu,
        "rcut" => rcut,
        "zcut" => zcut
    ) for (z_atom, norbs) in multiplicities
)
ace_model = Density_Model(dictionary; coupling_backend=:new)

errors = Float64[]
for test_idx in q:last_idx
    # q preceding frames = q-1 interpolation frames plus one reference frame
    model = ACEDExtModel(ace_model)
    train_idxs = (test_idx - q):(test_idx - 2)
    ref_idx = test_idx - 1

    # Train
    train_frames = [frames[i + 1] for i in train_idxs]
    train!(model, train_frames, frames[ref_idx + 1])

    # Test
    result = test(model, frames[test_idx + 1]; λ=λ)
    push!(errors, result.re)
    println("frame=$test_idx RE=$(result.re)")
end

mean_re = mean(errors)
println("EXT_ACE_D, nu=$nu, degree=$degree q=$q mean_RE=$mean_re")
