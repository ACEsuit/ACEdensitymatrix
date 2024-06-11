using Statistics, PythonCall, JLD2 # , Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")
include("../src/tuned_models.jl")

Ndata = 3000

# read data 
routine = "data/new_datasets"
filenames = ["$routine/propanol.h5"]# , "$routine/hexanol.h5", "$routine/acrolein.h5", "$routine/phenol.h5", "$routine/toluene.h5", "$routine/acetaldehyde.h5", "$routine/aniline.h5", "$routine/nmacetamide.h5"]
frames = []
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i =0:10:Ndata-1 ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

frames_test = []
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i = Ndata:10:9999 ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# Load / construct a model 
# parameters
rcut = 10.0
zcut = 10.0
for order = 1:2
    for degree = 2:10
        # Try load a model (and when necessary, do a refit)
        println("Loading the model ...")
        println()

        DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict

        println("Model loaded!")
        println()

        fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I), Mode = "H")

        println("Saving the model ...")
        println()

        save("test/CHO_Models/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2", write_dict(DM))

        println("Model saved!")
        println()
    end
end

degree = 10
order = 2
rcut = 10.0
zcut = 10.0
DM = load("test/CHO_Models/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict

frame = frames[1]
R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
# using BenchmarkTools
# @btime H_pred = eval_model(DM, R, ao_lab; retraction = identity)
H_pred = eval_model(DM, R, ao_lab; retraction = identity)
H
norm(H_pred - H) / norm(H)

s, q = eigen(Symmetric(S))
s_half = q * Diagonal(s.^(-1/2)) * q'
s_half * S * s_half ≈ I

(s_half * H * s_half * C ) ./ C

eigen(H).values
eigen(s_half * H * s_half).values

H_origin = apply_reorder(ao_lab, H; debug=false, bothsides = true, inverse = true)
S_origin = apply_reorder(ao_lab, S; debug=false, bothsides = true, inverse = true)
C_origin = apply_reorder(ao_lab, C; debug=false, inverse = true)
u, e, v = svd(S_origin)
sqrt_S = u * diagm(sqrt.(e)) * v'
C_origin = sqrt_S^(-1) * C_origin # Loewding transformation

a = H*C[:,1]
b = S*C[:,1]
a ./ b

(H * C) ./ C

H_trans * C ./ C

C * C' - D
C2 = eigen(Symmetric(H), Symmetric(S)).vectors[:,1:17] 
C2 * C2' - D
C = eigen(Symmetric(s_half * H * s_half))
H_trans = Symmetric(s_half * H * s_half)
H_trans * C.vectors[:,1] ./ C.vectors[:,1]
C = C.vectors[:,1:17]
C * C' - D
tr(D)


f = frames[1]
H = copy(f["Core Hamiltonian"])
S = copy(f["Overlap"])
C = copy(frame["Coefficients"]')

CC = eigen(H,S).vectors[:,1:10]
tr(CC * CC')

(H * C) ./ (S * C)

u, e, v = svd(S)
sqrt_S = u * diagm(sqrt.(e)) * v'
C = sqrt_S * C # Loewding transformation
H = sqrt_S^(-1) * H * sqrt_S^(-1)
H * C[:,1] ./ C[:,1]

HH = apply_reorder(f["Basis set labels"], H; debug=false, bothsides = true)
HHH = apply_reorder(f["Basis set labels"], HH; debug=false, bothsides = true, inverse = true)
HHH == H

# validate the model - Training
RE = 0
RMSE = 0
MV = 0
train_ref = Vector{Float64}()
train_pred = Vector{Float64}()
for frame in frames
    R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
    H_pred = eval_model(DM, R, ao_lab) # predicted density matrix w/o retraction
    push!(train_ref, vec(H)...)
    push!(train_pred, vec(H_pred)...)
    RMSE += norm(H_pred - H)^2/(size(H,1)*size(H,2))
    RE += norm(H_pred - H)/norm(H)
end
println("Training RMSE per matrix element = $(sqrt(RMSE/length(frames)))")
println("Average training relative error in H: ||H_pred - H_ref|| / ||H||= $(RE/length(frames))")
println()

# validate the model - Test
RE = 0
RMSE = 0
MV = 0
test_ref = Vector{Float64}()
test_pred = Vector{Float64}()
for frame in frames_test
    R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
    H_pred = eval_model(DM, R, ao_lab) # predicted density matrix w/o retraction
    push!(test_ref, vec(H)...)
    push!(test_pred, vec(H_pred)...)
    RMSE += norm(H_pred - H)^2/(size(H,1)*size(H,2))
    RE += norm(H_pred - H)/norm(H)
end
println("Test RMSE per matrix element = $(sqrt(RMSE/length(frames_test)))")
println("Average test relative error in D: ||H - H_ref|| / ||H||= $(RE/length(frames_test))")
println()

# # visualize the error
# E = abs.(D_pred - D)
# plt = contourf(E);
# savefig("Error_Degree$(parsed_args["degree"])_Ord$(parsed_args["order"])")

println("Done.")

# Backup: codes for checking the saved and loaded model are the same (at least do the same thing)
# dm = read_dict(load("model_deg$(maxdeg)_ord$(ord)_rcut$(rcut)_zcut$(zcut).jld"))
# @time eval_model(dm, R, translate_frame(frames[1])["ao_labels"]) # predicted density matrix
# @time eval_model(DM, R, translate_frame(frames[1])["ao_labels"]) # predicted density matrix
# @show eval_model(dm, R, translate_frame(frames[1])["ao_labels"]) == eval_model(DM, R, translate_frame(frames[1])["ao_labels"])

using Plots

train_pred = train_pred[findall(x -> x<0.9, train_ref)]
train_ref = train_ref[findall(x -> x<0.9, train_ref)]
test_pred = test_pred[findall(x -> x<0.9, test_ref)]
test_ref = test_ref[findall(x -> x<0.9, test_ref)]
pos = Int.(1:floor(length(train_ref)/8000):length(train_ref))

plot(train_ref[pos], train_ref[pos], label = "Reference")
scatter!(train_ref[pos], train_pred[pos], label = "Training")
scatter!(test_ref[pos], test_pred[pos], label = "Test")