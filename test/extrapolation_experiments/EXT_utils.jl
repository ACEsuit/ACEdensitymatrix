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
