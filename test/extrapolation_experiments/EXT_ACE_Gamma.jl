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

using ACEdensitymatrix
using LinearAlgebra
using Lux
using Random
using Statistics

include(joinpath(@__DIR__, "EXT_utils.jl"))

data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

"""Training data for one exact atom's `(l,0)` Gamma row block"""
struct GammaAngularFit
    l::Int
    rows::Vector{Int}
    basis_idx::Int
    train_basis::Matrix{Float64}
    ref_basis::Vector{Float64}
    train_targets::Vector{Matrix{Float64}}
end

"""Onsite ACE evaluator and angular fits for one exact atom"""
struct GammaAtomFit
    atom_idx::Int
    evaluator
    ps
    st
    atom_filter
    angular_fits::Vector{GammaAngularFit}
end

"""
State of one Gamma rolling-window problem

`origin` is the orthonormal occupied coefficient matrix of the final preceding
frame and `atom_fits` is populated by `train!` without species pooling
"""
mutable struct ACEGammaExtModel{m}
    onsite_models::m
    atom_fits::Vector{GammaAtomFit}
    origin::Matrix{Float64}
    gamma_size::Tuple{Int,Int}
end

ACEGammaExtModel(onsite_models) =
    ACEGammaExtModel(onsite_models, GammaAtomFit[], Matrix{Float64}(undef, 0, 0), (0, 0))

"""Return centered designs in atom-fit and angular-fit order"""
function design_mats(model::ACEGammaExtModel)
    return [center_design(fit.train_basis, fit.ref_basis)
            for atom in model.atom_fits for fit in atom.angular_fits]
end

"""Return regularized Gram matrices in the same order as `design_mats`"""
function gram_mats(model::ACEGammaExtModel; λ=0)
    return [regularized_gram(design; λ=λ) for design in design_mats(model)]
end

"""
Train the onsite `(l,0)` ACE descriptor of every exact atom independently

`ref_frame` is the final one of the `q` preceding frames, serving as both the
Grassmann tangent origin and the reference used to center the ACE descriptor
Its tangent-space representative is therefore exactly zero
All frames must already have been processed by `convert_frame`
"""
function train!(model::ACEGammaExtModel, train_frames, ref_frame)
    isempty(train_frames) && throw(ArgumentError("at least one interpolation frame is required"))
    z = ref_frame["atomic_numbers"]
    labels = ref_frame["ao_labels"]
    # left for debugging, but not strictly necessary since the frames are already converted
    # for frame in train_frames
    #     frame["atomic_numbers"] == z || error("atomic numbers change between training frames")
    #     vec(frame["ao_labels"]) == vec(labels) || error("AO labels change between training frames")
    # end

    # Map every global AO row to its exact atom and l block
    layout = ao_block_labels(labels)
    atom_ids = layout.atom_ids
    ls = layout.angular_momenta

    # The final preceding frame is the Grassmann origin
    # Its logarithm is zero, so it is omitted from `gammas`
    model.origin = ref_frame["C"]
    model.gamma_size = size(model.origin)
    empty!(model.atom_fits)

    # Express all q-1 interpolation frames in one tangent space before
    # dividing Gamma into atom/l row blocks
    gammas = [grassmann_log(frame["C"], model.origin) for frame in train_frames]

    for atom_idx in eachindex(z)
        z_atom = z[atom_idx]
        onsite = model.onsite_models[z_atom]

        # The first five Lux layers output the equivariant ACE basis
        # No fitted linear readout is used as a descriptor
        evaluator = Chain([onsite.model.layers[layer] for layer in 1:5]...)
        ps, st = Lux.setup(MersenneTwister(1234), evaluator)
        atom_filter = filter_on(rcut)

        # Evaluate each onsite environment once
        # The final packed column is B_ref and earlier columns align with `gammas`
        desc_frames = [train_frames; [ref_frame]]
        vals = map(desc_frames) do frame
            env = get_state(frame["R"], atom_idx, atom_idx; atom_filter=atom_filter)
            evaluator(env, ps, st)[1]
        end
        basis = pack_basis_evaluations(vals)

        # The partial model returns blocks in this nested-loop order
        # Retain only `(l,0)`, the transformation type of Gamma's l row block
        l1max, l2max = get_L(onsite)
        l_blocks = [(l1, l2) for l1 in 0:l1max for l2 in 0:l2max]
        length(basis) == length(l_blocks) ||
            error("basis and angular-block counts disagree for atom $atom_idx")

        angular_fits = GammaAngularFit[]
        atom_ls = sort(unique(ls[atom_ids .== atom_idx]))
        ntrain = length(train_frames)
        for l in atom_ls
            basis_idx = findfirst(==((l, 0)), l_blocks)
            isnothing(basis_idx) && error("onsite model for atom $atom_idx has no (l,0) block")

            # Select all radial and magnetic AO rows for this exact atom and l
            # Equivalent atoms intentionally receive separate fits
            rows = findall((atom_ids .== atom_idx) .& (ls .== l))
            block = basis[basis_idx]
            train_basis = Matrix{Float64}(view(block, :, 1:ntrain))
            ref_basis = Vector{Float64}(view(block, :, ntrain + 1))
            train_targets = [Matrix{Float64}(view(gamma, rows, :)) for gamma in gammas]
            push!(angular_fits, GammaAngularFit(l, rows, basis_idx, train_basis, ref_basis, train_targets))
        end

        push!(model.atom_fits, GammaAtomFit(atom_idx, evaluator, ps, st, atom_filter, angular_fits))
    end
    return model
end

"""Evaluate one target frame, exponentiate Gamma, and return density errors"""
function test(model::ACEGammaExtModel, frame; λ=1e-14)
    # Atom/l row blocks partition Gamma, so each independent prediction can be
    # written directly into its global rows
    gamma_pred = zeros(Float64, model.gamma_size)

    for atom in model.atom_fits
        # Evaluate the target onsite environment once and reuse its `(l,0)` blocks
        env = get_state(frame["R"], atom.atom_idx, atom.atom_idx; atom_filter=atom.atom_filter)
        val = atom.evaluator(env, atom.ps, atom.st)[1]
        test_basis = pack_basis_evaluations([val])

        for fit in atom.angular_fits
            # Coefficients are local to one exact atom and one l value
            coeffs = fit_coeffs(fit.train_basis, fit.ref_basis,
                                view(test_basis[fit.basis_idx], :, 1); λ=λ)
            dest = view(gamma_pred, fit.rows, :)
            for (coeff, target) in zip(coeffs, fit.train_targets)
                # Gamma_ref=0, so the centered affine reconstruction is sum_i a_i Gamma_i
                dest .+= coeff .* target
            end
        end
    end

    # Independent fits can introduce a small component parallel to the
    # occupied reference space, so project it away before the exponential map
    gamma_tan = gamma_pred - model.origin * (model.origin' * gamma_pred)

    # The Grassmann exponential returns an orthonormal occupied coefficient
    # matrix, so C*C' is an idempotent density with the correct trace
    c_pred = grassmann_exp(gamma_tan, model.origin)
    d_pred = c_pred * c_pred'
    d_ref = frame["D"]
    raw_re = norm(d_pred - d_ref) / norm(d_ref)

    # Map the orthogonal-basis density difference back to the AO basis before
    # computing the same RE reported by the direct-D script
    sinv_sqrt = frame["S"]^(-0.5)
    d_ref_ao = sinv_sqrt * d_ref * sinv_sqrt
    diff_ao = sinv_sqrt * (d_pred - d_ref) * sinv_sqrt
    re = norm(diff_ao) / norm(d_ref_ao)
    tan_correction = norm(gamma_tan - gamma_pred) / max(norm(gamma_pred), eps(Float64))

    return (
        re=re,
        raw_re=raw_re,
        gamma_pred=gamma_pred,
        gamma_tan=gamma_tan,
        tan_correction=tan_correction,
        d_pred=d_pred
    )
end

"""
Run `nwindows` rolling predictions

Trajectory indices are zero-based in the printed output
For `q=20`, target 20 uses frames 0:18 for interpolation and frame 19 as both
the descriptor and Grassmann reference, while the default run predicts 20:99
"""
nu = 2
degree = 3
q = 20
nwindows = 80
λ = 1e-14
rcut = 6.5

# q > 1 || error("q must exceed one")
# nwindows > 0 || error("nwindows must be positive")
# λ >= 0 || error("λ must be nonnegative")

# Load each required frame once
# Vector position `i+1` corresponds to zero-based trajectory frame i
last_idx = q + nwindows - 1
trajectory = TrajectoryHDF5(data_file)
frames = [convert_frame(read_frame(trajectory, i)) for i in 0:last_idx]
close(trajectory.file)

# Reuse species-level ACE construction while each rolling window receives
# fresh descriptors, tangent vectors, and local least-squares problems
multiplicities = infer_orbital_multiplicities(first(frames))
species = sort(collect(keys(multiplicities)))
onsite_models = Dict{Int,Any}(
    z_atom => On_Model(degree, nu, rcut, z_atom, species,
                       length(multiplicities[z_atom]) - 1, multiplicities[z_atom]; coupling_backend=:new)
    for z_atom in species
)

errors = Float64[]
for test_idx in q:last_idx
    # q preceding frames = q-1 interpolation frames plus one reference frame
    # The reference is also the tangent origin
    ref_idx = test_idx - 1
    ref_frame = frames[ref_idx + 1]
    model = ACEGammaExtModel(onsite_models)

    # Train
    train_idxs = (test_idx - q):(test_idx - 2)
    train_frames = [frames[i + 1] for i in train_idxs]
    train!(model, train_frames, ref_frame)

    # Test
    result = test(model, frames[test_idx + 1]; λ=λ)
    push!(errors, result.re)
    println("frame=$test_idx RE=$(result.re)")
end

mean_re = mean(errors)
println("EXT_ACE_Gamma, nu=$nu, degree=$degree q=$q mean_RE=$mean_re")
