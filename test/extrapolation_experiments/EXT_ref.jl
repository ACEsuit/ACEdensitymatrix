module RefExtrapolation

using ACEdensitymatrix
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "EXT_utils.jl"))
using .EXTUtils

include(joinpath(@__DIR__, "transformations.jl"))
using .DensityTransformations

export extrapolate!, optimal_lambda!

using LinearAlgebra

"""
    optimal_lambda!(train_frames, ref_frame, test_frame; tangent_frame=nothing, ...)

Same interface as `extrapolate!`, but instead of using a given λ it searches
for the λ that minimizes the relative error on `test_frame`.
Returns (λ, re, scan, history).
"""
function optimal_lambda!(train_frames, ref_frame, test_frame;
                         tangent_frame=nothing,
                         log10λ_bounds=(-16.0, 2.0),
                         n_scan=19,
                         xtol=0.02,
                         verbose=true)

    nocc = Int(sum(test_frame["atomic_numbers"]) / 2)
    q = length(train_frames)
    verbose && println("q = $q")

    transform = HorLiftTangent(nocc)
    set_tangent!(transform, isnothing(tangent_frame) ? ref_frame["C"] : tangent_frame["C"])

    descr_ref       = vec(ref_frame["Hcore"])
    training_descrs = [vec(f["Hcore"]) for f in train_frames]
    B               = center_design(hcat(training_descrs...), descr_ref)
    descr_target    = vec(test_frame["Hcore"]) - descr_ref

    gamma_ref = forward_transform(transform, ref_frame["C"])
    gammas    = [forward_transform(transform, f["C"]) for f in train_frames]

    sinv_sqrt = test_frame["S"]^(-0.5)
    d_ref     = test_frame["D"]
    d_ref_ao_norm = norm(sinv_sqrt * d_ref * sinv_sqrt)

    history = Tuple{Float64,Float64}[]

    function rel_err(x)
        re = try
            a = fit_coeffs(B, descr_target; λ=10.0^x)
            guess_gamma = (1.0 - sum(a)) * copy(gamma_ref)
            for (ai, g) in zip(a, gammas)
                guess_gamma += ai * g
            end
            c = backward_transform(transform, guess_gamma)
            d_pred = c * c'
            norm(sinv_sqrt * (d_pred - d_ref) * sinv_sqrt) / d_ref_ao_norm
        catch e
            (e isa SingularException || e isa PosDefException || e isa LAPACKException) || rethrow()
            Inf
        end
        re = isfinite(re) ? re : Inf
        push!(history, (x, re))
        return re
    end

    xs  = collect(range(log10λ_bounds[1], log10λ_bounds[2]; length=n_scan))
    res = [rel_err(x) for x in xs]
    k   = argmin(res)

    if verbose && (k == 1 || k == n_scan)
        @warn "Scan minimum is on the boundary of log10λ_bounds " *
              "(log10λ = $(xs[k])); widen the range, or regularization doesn't help here."
    end

    lo, hi = xs[max(k - 1, 1)], xs[min(k + 1, n_scan)]
    φ  = (sqrt(5) - 1) / 2
    x1 = hi - φ * (hi - lo);  f1 = rel_err(x1)
    x2 = lo + φ * (hi - lo);  f2 = rel_err(x2)
    while hi - lo > xtol
        if f1 < f2
            hi = x2; x2 = x1; f2 = f1
            x1 = hi - φ * (hi - lo); f1 = rel_err(x1)
        else
            lo = x1; x1 = x2; f1 = f2
            x2 = lo + φ * (hi - lo); f2 = rel_err(x2)
        end
    end

    xbest, rebest = history[argmin(last.(history))]
    verbose && println("optimal λ = $(10.0^xbest)  (re = $rebest, $(length(history)) evaluations)")

    λs_scan = 10.0 .^ xs
    history_λ = [(10.0^x, re) for (x, re) in history]

    return (λ=10.0^xbest, re=rebest,
            scan=(λ=λs_scan, re=res),
            history=history_λ)
end

function extrapolate!(train_frames, ref_frame, test_frame; λ=1e-14, tangent_frame=nothing)
    nocc = Int(sum(test_frame["atomic_numbers"]) / 2)
    transform = HorLiftTangent(nocc)
    set_tangent!(transform, ref_frame["C"])

    # just for consistency with the old reference which used the
    # first frame of the trajectory as the tangent point
    if isnothing(tangent_frame)
        transform = HorLiftTangent(nocc)
        set_tangent!(transform, ref_frame["C"])
    else
        transform = HorLiftTangent(nocc)
        set_tangent!(transform, tangent_frame["C"])
    end

    descr_ref = vec(ref_frame["Hcore"])
    c = ref_frame["C"]
    gamma_ref = forward_transform(transform, c)

    training_descrs = [
        vec(frame["Hcore"]) for frame in train_frames
    ]
    B = center_design(hcat(training_descrs...), descr_ref)

    gammas = []
    for frame in train_frames
        gamma = forward_transform(transform, frame["C"])
        push!(gammas, gamma)
    end

    descr_target = vec(test_frame["Hcore"]) - descr_ref
    a = fit_coeffs(B, descr_target; λ=λ)
    guess_gamma = (1.0 - sum(a))*copy(gamma_ref)
    for (i, frame) in enumerate(train_frames)
        gamma = forward_transform(transform, frame["C"])
        guess_gamma += a[i]*gamma
    end

    c = backward_transform(transform, guess_gamma)
    d_pred = c * c'

    sinv_sqrt = test_frame["S"]^(-0.5)
    d_ref = test_frame["D"]

    d_ref_ao = sinv_sqrt * d_ref * sinv_sqrt
    diff_ao = sinv_sqrt * (d_pred - d_ref) * sinv_sqrt
    re = norm(diff_ao) / norm(d_ref_ao)
    return (re=re,)
end

end

if abspath(PROGRAM_FILE) == @__FILE__

    using ACEdensitymatrix

    using .RefExtrapolation
    using Statistics

    include(joinpath(@__DIR__, "EXT_utils.jl"))
    using .EXTUtils

    q = 20
    nwindows = 80
    λ = (1.1e-5)^2
    rcut = 6.5
    data_file = joinpath(@__DIR__, "..", "..", "data", "new_datasets", "oxirane.h5")

    last_idx = q + nwindows - 1
    trajectory = TrajectoryHDF5(data_file)
    info = read_info(trajectory)
    nocc = div(info["nelectrons"], 2)
    frames = [convert_frame(read_frame(trajectory, i)) for i in 0:last_idx]
    close(trajectory.file)

    errors = Float64[]
    for test_idx in q:last_idx
        train_frames = frames[(test_idx - q + 1):(test_idx - 1)]
        result = extrapolate!(train_frames, frames[test_idx], frames[test_idx + 1]; λ=λ)
        println(result)
        push!(errors, result.re)
        println("frame=$test_idx RE=$(result.re)")
    end

    println(mean(errors))
end
