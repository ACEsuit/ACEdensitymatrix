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
using .EXTUtils

export ACEDExtModel, design_mats, gram_mats, extrapolate!, training_size

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

function ACEDExtModel(model_spec; coupling_backend=:new,
                      radial_mode::Symbol=:sample)
    ace_model = Density_Model(model_spec["species"];
        coupling_backend=coupling_backend, fixed_particle_number=true,
        radial_mode=radial_mode)
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

function basis_values(model::ACEDExtModel, frames; parallel=false)
    z = first(frames)["atomic_numbers"]
    rcut_on, rcut_off, zcut = get_cutoff(model.ace_model)

    return parallel_map(model.atom_blocks; parallel=parallel) do atom
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
        packed = pack_basis_evaluations(vals)
        length(packed) == length(atom.ao_blocks) ||
            error("basis and AO-block counts disagree for atoms $atom_i and $atom_j")
        packed
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

function extrapolate!(model::ACEDExtModel, train_frames, ref_frame, test_frame;
                      λ=1e-14, parallel=false)
    isempty(train_frames) && throw(ArgumentError("at least one training frame is required"))
    ntrain = length(train_frames)
    basis = basis_values(model, [train_frames; [ref_frame, test_frame]];
                         parallel=parallel)
    natom_blocks = length(model.atom_blocks)
    coeff_sets = [Vector{Vector{Float64}}(undef, length(atom.ao_blocks))
                  for atom in model.atom_blocks]
    # Use one atom-pair buffer in both paths so threads never write to d_raw concurrently
    d_blocks = [zeros(Float64, length(atom.pos_i), length(atom.pos_j))
                for atom in model.atom_blocks]
    d_raw = zeros(Float64, size(test_frame["D"]))
    # Extract density blocks once in both paths so threaded jobs only read shared inputs
    d_trains = [[get_block(frame["D"], atom.atom_i, atom.atom_j,
                           frame["ao_labels"]) for frame in train_frames]
                for atom in model.atom_blocks]
    d_refs = [get_block(ref_frame["D"], atom.atom_i, atom.atom_j,
                        ref_frame["ao_labels"]) for atom in model.atom_blocks]

    indices = [(atom_idx, ao_idx) for atom_idx in 1:natom_blocks
               for ao_idx in eachindex(model.atom_blocks[atom_idx].ao_blocks)]
    ao_fits = parallel_map(indices; parallel=parallel) do index
        atom_idx, ao_idx = index
        ao = model.atom_blocks[atom_idx].ao_blocks[ao_idx]
        block = basis[atom_idx][ao.basis_idx]
        λ_block = _block_λ(λ, atom_idx, ao_idx)
        ao_coeffs = fit_coeffs(view(block, :, 1:ntrain), view(block, :, ntrain + 1),
                               view(block, :, ntrain + 2); λ=λ_block)
        ref_target = view(d_refs[atom_idx], ao.rows, ao.cols)
        ao_pred = Matrix(ref_target)
        for (coeff, d) in zip(ao_coeffs, d_trains[atom_idx])
            ao_pred .+= coeff .* (view(d, ao.rows, ao.cols) .- ref_target)
        end
        return (ao_coeffs, ao_pred)
    end

    for ((atom_idx, ao_idx), (block_coeffs, d_block_pred)) in zip(indices, ao_fits)
        ao = model.atom_blocks[atom_idx].ao_blocks[ao_idx]
        coeff_sets[atom_idx][ao_idx] = block_coeffs
        d_blocks[atom_idx][ao.rows, ao.cols] .= d_block_pred
    end
    for (atom, d_block) in zip(model.atom_blocks, d_blocks)
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

_block_λ(λ::Real, atom_idx, ao_idx) = λ
_block_λ(λ::AbstractVector{<:AbstractVector}, atom_idx, ao_idx) = λ[atom_idx][ao_idx]

"""
Minimize f(λ) over λ_bounds: coarse scan in log10 λ, then golden-section
refinement around the best scan point. Returns the best λ over ALL evaluations.
"""
function minimize_log_lambda(f; λ_bounds=(1e-16, 1e2), n_scan=19, xtol=0.02)
    history = Tuple{Float64,Float64}[]
    function obj(x)
        v = f(10.0^x)
        v = isfinite(v) ? Float64(v) : Inf
        push!(history, (x, v))
        return v
    end

    xs  = collect(range(log10(λ_bounds[1]), log10(λ_bounds[2]); length=n_scan))
    res = [obj(x) for x in xs]
    k   = argmin(res)

    lo, hi = xs[max(k - 1, 1)], xs[min(k + 1, n_scan)]
    φ  = (sqrt(5) - 1) / 2
    x1 = hi - φ * (hi - lo);  f1 = obj(x1)
    x2 = lo + φ * (hi - lo);  f2 = obj(x2)
    while hi - lo > xtol
        if f1 < f2
            hi = x2; x2 = x1; f2 = f1
            x1 = hi - φ * (hi - lo); f1 = obj(x1)
        else
            lo = x1; x1 = x2; f1 = f2
            x2 = lo + φ * (hi - lo); f2 = obj(x2)
        end
    end

    xbest, vbest = history[argmin(last.(history))]
    isinf(vbest) && @warn "All evaluations failed; returned λ is meaningless."
    return (λ=10.0^xbest, value=vbest,
            scan=(λ=10.0 .^ xs, value=res),
            history=[(10.0^x, v) for (x, v) in history],
            at_boundary=(k == 1 || k == n_scan))
end

function optimal_lambdas(model::ACEDExtModel, train_frames, ref_frame, test_frame;
                         λ_bounds=(1e-16, 1e2), n_scan=19, xtol=0.02,
                         parallel=false, verbose=true)
    isempty(train_frames) && throw(ArgumentError("at least one training frame is required"))
    ntrain = length(train_frames)

    basis = basis_values(model, [train_frames; [ref_frame, test_frame]];
                         parallel=parallel)
    d_trains = [[get_block(frame["D"], atom.atom_i, atom.atom_j, frame["ao_labels"])
                 for frame in train_frames] for atom in model.atom_blocks]
    d_refs   = [get_block(ref_frame["D"], atom.atom_i, atom.atom_j, ref_frame["ao_labels"])
                for atom in model.atom_blocks]
    d_tests  = [get_block(test_frame["D"], atom.atom_i, atom.atom_j, test_frame["ao_labels"])
                for atom in model.atom_blocks]

    indices = [(atom_idx, ao_idx) for atom_idx in eachindex(model.atom_blocks)
               for ao_idx in eachindex(model.atom_blocks[atom_idx].ao_blocks)]

    results = parallel_map(indices; parallel=parallel) do index
        atom_idx, ao_idx = index
        ao    = model.atom_blocks[atom_idx].ao_blocks[ao_idx]
        block = basis[atom_idx][ao.basis_idx]
        Btrain = view(block, :, 1:ntrain)
        bref   = view(block, :, ntrain + 1)
        btest  = view(block, :, ntrain + 2)

        ref_t  = view(d_refs[atom_idx], ao.rows, ao.cols)
        true_t = view(d_tests[atom_idx], ao.rows, ao.cols)

        t     = vec(true_t .- ref_t)
        Δ     = hcat([vec(view(d, ao.rows, ao.cols) .- ref_t) for d in d_trains[atom_idx]]...)
        tnorm = max(norm(true_t), eps())

        f = λ -> try
            a = fit_coeffs(Btrain, bref, btest; λ=λ)
            norm(t - Δ * a) / tnorm
        catch e
            (e isa SingularException || e isa PosDefException || e isa LAPACKException) || rethrow()
            Inf
        end

        return minimize_log_lambda(f; λ_bounds=λ_bounds, n_scan=n_scan, xtol=xtol)
    end

    λs  = [zeros(Float64, length(atom.ao_blocks)) for atom in model.atom_blocks]
    res = [zeros(Float64, length(atom.ao_blocks)) for atom in model.atom_blocks]
    scans = Dict{Tuple{Int,Int},Any}()
    for ((atom_idx, ao_idx), r) in zip(indices, results)
        λs[atom_idx][ao_idx]  = r.λ
        res[atom_idx][ao_idx] = r.value
        scans[(atom_idx, ao_idx)] = (scan=r.scan, history=r.history)
    end

    if verbose
        flat = reduce(vcat, λs)
        nbound = count(r -> r.at_boundary, results)
        println("optimal λ over $(length(flat)) blocks: ",
                "min=$(minimum(flat)), median=$(sort(flat)[cld(length(flat), 2)]), max=$(maximum(flat))")
        nbound > 0 && @warn "$nbound blocks have their scan minimum on the boundary of λ_bounds"
    end

    return (λ=λs, re_blocks=res, indices=indices, scans=scans)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .ACEDExtrapolation
    using ACEdensitymatrix
    using Statistics

    include(joinpath(@__DIR__, "EXT_utils.jl"))
    using .EXTUtils

    data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

    nu = 2
    degree = 3
    q = 20
    nwindows = 80
    λ = 1e-14
    # Set true when Julia is started with multiple threads; single-threaded BLAS
    # avoids oversubscription of the many independent small QR solves
    parallel = false
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
                              frames[test_idx + 1]; λ=λ, parallel=parallel)
        push!(errors, result.re)
        println("frame=$test_idx RE=$(result.re)")
    end

    mean_re = mean(errors)
    println("EXT_ACE_D, nu=$nu, degree=$degree q=$q mean_RE=$mean_re")
end
