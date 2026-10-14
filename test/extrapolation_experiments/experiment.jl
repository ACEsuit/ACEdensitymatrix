using ACEdensitymatrix
using JSON
using Printf

include(joinpath(@__DIR__, "EXT_utils.jl"))
using .EXTUtils

include(joinpath(@__DIR__, "EXT_ACE_D.jl"))
import .ACEDExtrapolation

include(joinpath(@__DIR__, "EXT_ACE_Gamma.jl"))
import .ACEGammaExtrapolation

include(joinpath(@__DIR__, "EXT_ACE_Gamma_horlift_global.jl"))
import .ACEGammaExtrapolation_Global

include(joinpath(@__DIR__, "EXT_ref.jl"))
import .RefExtrapolation

MOLECULES = ["cytosine", "ethyl_acetate", "glycine", "oxirane", "proline"]
MOLECULES = ["proline"]
QS = 5:5:50
QS = [20]
NU_DEGREE = [(2, 3), (3, 4), (4, 4)]    # (nu, degree)
NU_DEGREE = [(2, 3)]
NFRAMES = 200

λ = 1e-8
# Set true when Julia is started with multiple threads; single-threaded BLAS
# avoids oversubscription of the many independent small QR solves
PARALLEL = false
RCUT = 6.5
ZCUT = 10.0

DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "new_datasets")
RESULTS_DIR = joinpath(@__DIR__, "results")

# for better comparison, we start the extrapolation at a common frame
# for every possible value of q
start_idx = maximum(QS)

const METHODS = [
    (name = "ACE_D",
     make = spec -> ACEDExtrapolation.ACEDExtModel(spec; coupling_backend=:new),
     extrapolate! = ACEDExtrapolation.extrapolate!),
    (name = "ACE_Gamma",
     make = spec -> ACEGammaExtrapolation.ACEGammaExtModel(spec; coupling_backend=:new),
     extrapolate! = ACEGammaExtrapolation.extrapolate!),
    (name = "ACE_Gamma_global",
     make = spec -> ACEGammaExtrapolation_Global.ACEGammaExtModel_Global(spec),
     extrapolate! = ACEGammaExtrapolation_Global.extrapolate!),
]

function main()
    for (im, molecule) in enumerate(MOLECULES)
        # ---------- load or resume results ----------
        path = joinpath(RESULTS_DIR, "$(molecule).json")
        if isfile(path)
            results = JSON.parsefile(path)
        else
            results = Dict{String,Any}(
                "molecule" => molecule,
                "settings" => Dict{String,Any}(
                    "lambda" => λ, "rcut" => RCUT, "zcut" => ZCUT,
                    "n_frames" => NFRAMES),
                "entries" => Any[],
            )
        end
        entries = results["entries"]
        done = Set((e["method"], e["q"], e["nu"], e["degree"]) for e in entries)

        all_keys = vcat(
            [("ref", q, nothing, nothing) for q in QS],
            [(m.name, q, nu, d) for m in METHODS for (nu, d) in NU_DEGREE for q in QS],
        )
        n_total = length(all_keys)
        n_done = count(in(done), all_keys)

        println("=== [$im/$(length(MOLECULES))] $molecule  ($(n_done)/$n_total already done) ===")
        n_done == n_total && continue

        # common load frames
        trajectory = TrajectoryHDF5(joinpath(DATA_DIR, "$(molecule).h5"))
        frames = [convert_frame(read_frame(trajectory, i)) for i in 0:(NFRAMES - 1)]
        close(trajectory.file)
        meta = frame2meta(first(frames))
        last_idx = length(frames) - 1

        # reference extrapolation
        for q in QS
            key = ("ref", q, nothing, nothing)
            key in done && continue

            elapsed = @elapsed begin
                errs = Float64[]
                for test_idx in start_idx:last_idx
                    train_idxs = (test_idx - q):(test_idx - 2)
                    train_frames = [frames[i + 1] for i in train_idxs]
                    result = RefExtrapolation.extrapolate!(
                        train_frames, frames[test_idx], frames[test_idx + 1]; λ=λ)
                    push!(errs, result.re)
                end
            end

            push!(entries, Dict{String,Any}(
                "molecule" => molecule, "method" => "ref",
                "q" => q, "nu" => nothing, "degree" => nothing,
                "elapsed_s" => elapsed,
                "errors" => [isfinite(x) ? x : nothing for x in errs],
            ))
            push!(done, key)
            mkpath(dirname(path))
            tmp = path * ".tmp"
            open(io -> JSON.print(io, results), tmp, "w")
            mv(tmp, path; force=true)
            n_done += 1
            @printf("  [%3d/%d] %-16s q=%-2d %-12s %6.1f s\n",
                    n_done, n_total, "ref", q, "", elapsed)
        end

        # three ACE-based extrapolations
        for (nu, degree) in NU_DEGREE
            keys_nd = [(m.name, q, nu, degree) for m in METHODS for q in QS]
            all(in(done), keys_nd) && continue

            spec = Dict{String,Any}(
                "species" => Dict(
                    z => Dict{String,Any}(
                        "n_orbs" => n_orb,
                        "maxdeg" => degree,
                        "ord" => nu,
                        "rcut" => RCUT,
                        "zcut" => ZCUT
                    ) for (z, n_orb) in zip(meta.species, meta.n_orbs)
                ),
                "atomic_numbers" => meta.atomic_numbers,
                "ao_labels" => meta.ao_labels
            )

            for m in METHODS
                model = m.make(spec)
                for q in QS
                    key = (m.name, q, nu, degree)
                    key in done && continue

                    elapsed = @elapsed begin
                        errs = Float64[]
                        for test_idx in start_idx:last_idx
                            if mod(test_idx, 50) == 0
                                print(" $test_idx")
                            end
                            train_idxs = (test_idx - q):(test_idx - 2)
                            train_frames = [frames[i + 1] for i in train_idxs]
                            result = m.extrapolate!(
                                model, train_frames, frames[test_idx], frames[test_idx + 1];
                                λ=λ, parallel=PARALLEL)
                            push!(errs, result.re)
                        end
                    end
                    println()

                    push!(entries, Dict{String,Any}(
                        "molecule" => molecule, "method" => m.name,
                        "q" => q, "nu" => nu, "degree" => degree,
                        "elapsed_s" => elapsed,
                        "errors" => [isfinite(x) ? x : nothing for x in errs],
                    ))
                    push!(done, key)
                    mkpath(dirname(path))
                    tmp = path * ".tmp"
                    open(io -> JSON.print(io, results), tmp, "w")
                    mv(tmp, path; force=true)
                    n_done += 1
                    @printf("  [%3d/%d] %-16s q=%-2d %-12s %6.1f s\n",
                            n_done, n_total, m.name, q, "nu=$nu deg=$degree", elapsed)
                end
            end
        end
    end
    println("All experiments finished.")
end

main()
