using ACEdensitymatrix
using LinearAlgebra

"""Grassmann logarithm in the tangent space at `origin`."""
function grassmann_log(coefficients, origin)
    psi, _, rotation = svd(coefficients' * origin; full=false)
    aligned = coefficients * psi * rotation'
    chord = (I(size(coefficients, 1)) - origin * origin') * aligned
    left, values, right = svd(chord; full=false)
    return left * Diagonal(asin.(clamp.(values, -1.0, 1.0))) * right'
end

"""Grassmann exponential from the tangent space at `origin`."""
function grassmann_exp(gamma, origin)
    left, values, right = svd(gamma; full=false)
    return origin * right * Diagonal(cos.(values)) * right' +
        left * Diagonal(sin.(values)) * right'
end

"""Center the columns of `train_vals` on `ref_vals`"""
function center_design(train_vals, ref_vals)
    nrows, ncoeffs = size(train_vals)
    length(ref_vals) == nrows ||
        throw(DimensionMismatch("training and reference descriptors have incompatible sizes"))

    # Center every training descriptor on the common reference
    t = promote_type(eltype(train_vals), eltype(ref_vals))
    design = Matrix{t}(undef, nrows, ncoeffs)
    for col in 1:ncoeffs
        view(design, :, col) .= view(train_vals, :, col) .- ref_vals
    end
    return design
end

"""Return `design' * design + λI`"""
function regularized_gram(design; λ=0)
    λ >= 0 || throw(ArgumentError("λ must be nonnegative"))
    t = promote_type(eltype(design), typeof(float(λ)))
    gram = Matrix{t}(design' * design)
    for i in axes(gram, 1)
        gram[i, i] += λ
    end
    return gram
end

"""Solve `design * coeffs ≈ target` by regularized QR"""
function fit_coeffs(design_vals::AbstractMatrix, target_vals::AbstractVector; λ=1e-14)
    nrows, ncoeffs = size(design_vals)
    length(target_vals) == nrows ||
        throw(DimensionMismatch("design and target have incompatible sizes"))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative"))

    # Promote the inputs once so the in-place QR uses a single scalar type
    t = promote_type(eltype(design_vals), eltype(target_vals), typeof(float(λ)))
    design = Matrix{t}(design_vals)
    target = Vector{t}(target_vals)

    # Solve the ordinary least-squares problem directly when λ is zero
    λ == 0 && return qr!(design) \ target

    # Append sqrt(λ)I so QR solves the ridge problem without normal equations
    aug = zeros(t, nrows + ncoeffs, ncoeffs)
    view(aug, 1:nrows, :) .= design
    sqrt_λ = sqrt(t(λ))
    for i in 1:ncoeffs
        aug[nrows + i, i] = sqrt_λ
    end
    aug_target = zeros(t, nrows + ncoeffs)
    view(aug_target, 1:nrows) .= target
    return qr!(aug) \ aug_target
end

"""Solve `sum_i coeffs[i] * (B_i - B_ref) ≈ B_test - B_ref` by regularized QR"""
function fit_coeffs(train_vals, ref_vals, test_vals; λ=1e-14)
    design = center_design(train_vals, ref_vals)
    length(test_vals) == size(design, 1) ||
        throw(DimensionMismatch("training and test descriptors have incompatible sizes"))
    target = test_vals .- ref_vals
    return fit_coeffs(design, target; λ=λ)
end

"""Return one-based atom indices and angular labels in canonical AO order"""
function ao_block_labels(labels)
    # Reorder labels and retain the metadata that defines each AO block
    _, atom_ids, ls, ms = apply_reorder(labels; full_info=true)
    return (
        atom_ids=atom_ids .+ 1,
        angular_momenta=ls,
        magnetic_indices=ms
    )
end

"""Return the radial multiplicity of every angular shell by species"""
function infer_orbital_multiplicities(frame)
    layout = ao_block_labels(frame["ao_labels"])
    mults = Dict{Int,Vector{Int}}()
    for (atom_idx, z) in enumerate(frame["atomic_numbers"])
        # Count complete magnetic shells for this exact atom
        atom_ls = layout.angular_momenta[layout.atom_ids .== atom_idx]
        norbs = map(0:maximum(atom_ls)) do l
            ncomp = count(==(l), atom_ls)
            nmag = 2l + 1
            ncomp % nmag == 0 || error("atom $atom_idx has an incomplete l=$l shell")
            div(ncomp, nmag)
        end
        if haskey(mults, z)
            # Equivalent species must use identical AO layouts
            mults[z] == norbs || error("atoms of species $z use inconsistent AO bases")
        else
            mults[z] = norbs
        end
    end
    return mults
end

"""Construct ACE basis and molecular metadata from one converted frame"""
function frame2dict(frame; nu, degree, rcut, zcut=10.0)
    mults = infer_orbital_multiplicities(frame)
    species = Dict(
        z => Dict{String,Any}(
            "n_orbs" => norbs,
            "maxdeg" => degree,
            "ord" => nu,
            "rcut" => rcut,
            "zcut" => zcut
        ) for (z, norbs) in mults
    )
    return Dict{String,Any}(
        "species" => species,
        "atomic_numbers" => copy(frame["atomic_numbers"]),
        "ao_labels" => copy(vec(frame["ao_labels"]))
    )
end

"""Return the AO range containing every radial copy of angular momentum `l`"""
function angular_orbital_range(norbs, l)
    # Skip every lower-l shell in canonical AO order
    first_idx = 1 + sum(norbs[k + 1] * (2k + 1) for k in 0:(l - 1); init=0)
    n = norbs[l + 1] * (2l + 1)
    return first_idx:(first_idx + n - 1)
end

"""Flatten matrix-valued ACE bases into one column per configuration"""
function pack_basis_evaluations(vals)
    isempty(vals) && throw(ArgumentError("at least one basis evaluation is required"))
    nblocks = length(first(vals))
    all(length(val) == nblocks for val in vals) ||
        throw(DimensionMismatch("basis block counts are inconsistent"))

    return map(1:nblocks) do block_idx
        # Fix the element types and packed column length from the first sample
        ref = vals[1][block_idx]
        matrix_t = eltype(ref)
        scalar_t = eltype(matrix_t)
        col_len = length(ref) * length(matrix_t)
        packed = Matrix{scalar_t}(undef, col_len, length(vals))

        for sample_idx in eachindex(vals)
            block = vals[sample_idx][block_idx]
            eltype(block) === matrix_t ||
                throw(DimensionMismatch("block $block_idx has inconsistent matrix types"))
            length(block) == length(ref) ||
                throw(DimensionMismatch("block $block_idx has inconsistent basis counts"))

            # Static matrices are contiguous, so reinterpret and copy one column
            offset = (sample_idx - 1) * col_len + 1
            copyto!(packed, offset, reinterpret(scalar_t, block), 1, col_len)
        end
        packed
    end
end
