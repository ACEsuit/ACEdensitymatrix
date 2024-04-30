using Statistics, PythonCall, JLD # , Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")

Ndata = 200
rcut = 10.0
zcut = 10.0
degree = 4
order = 3

# read data 
filenames = ["data/propanol.h5", "data/esanol.h5", "data/acrolein.h5", "data/phenol.h5", "data/toluene.h5", "data/acetaldehyde.h5", "data/aniline.h5", "data/nmacetamide.h5"]
frames = []
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i =1:floor(10000/Ndata):10000 ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

frames_test = []
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i =2:floor(10000/Ndata):10000 ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# construct a model 
ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut), 
                6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                7 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut) )

println("Constructing the model ...")
println()

DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

println("Model constructed!")
println()

# fit the model
fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))

# load model
DM = load("test/CHON_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld")|> read_dict

# validate the model - Training
RE = 0
RMSE = 0
MV = 0
train_ref = Vector{Float64}()
train_pred = Vector{Float64}()
for frame in frames
    R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
    D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
    push!(train_ref, vec(D)...)
    push!(train_pred, vec(D_pred)...)
    RMSE += norm(D_pred - D)^2/(size(D,1)*size(D,2))
    RE += norm(D_pred - D)/norm(D)
    MV += norm(D_pred * D_pred - D_pred)
end
println("Training RMSE per matrix element = $(sqrt(RMSE/length(frames)))")
println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE/length(frames))")
println("Average training manifold violation = $(MV/length(frames))")
println()

# validate the model - Test
RE = 0
RMSE = 0
MV = 0
test_ref = Vector{Float64}()
test_pred = Vector{Float64}()
for frame in frames_test
    R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
    D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
    push!(test_ref, vec(D)...)
    push!(test_pred, vec(D_pred)...)
    RMSE += norm(D_pred - D)^2/(size(D,1)*size(D,2))
    RE += norm(D_pred - D)/norm(D)
    MV += norm(D_pred * D_pred - D_pred)
end
println("Test RMSE per matrix element = $(sqrt(RMSE/length(frames_test)))")
println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE/length(frames_test))")
println("Average test manifold violation = $(MV/length(frames_test))")
println()

# # visualize the error
# E = abs.(D_pred - D)
# plt = contourf(E);
# savefig("Error_Degree$(parsed_args["degree"])_Ord$(parsed_args["order"])")

# save the model
println("Saving the model ...")
println()

save("test/CHON_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))

println("Model saved!")
println()

println("Done.")

# Backup: codes for checking the saved and loaded model are the same (at least do the same thing)
# dm = read_dict(load("model_deg$(maxdeg)_ord$(ord)_rcut$(rcut)_zcut$(zcut).jld"))
# @time eval_model(dm, R, translate_frame(frames[1])["ao_labels"]) # predicted density matrix
# @time eval_model(DM, R, translate_frame(frames[1])["ao_labels"]) # predicted density matrix
# @show eval_model(dm, R, translate_frame(frames[1])["ao_labels"]) == eval_model(DM, R, translate_frame(frames[1])["ao_labels"])

using Plots
train_pred = train_pred[findall(x -> x<0.9, train_pred)]
train_ref = train_ref[findall(x -> x<0.9, train_ref)]
test_pred = test_pred[findall(x -> x<0.9, test_pred)]
test_ref = test_ref[findall(x -> x<0.9, test_ref)]
pos = Int.(1:floor(length(train_ref)/3000):length(train_ref))

plot(train_ref[pos], train_ref[pos], label = "Reference")
scatter!(train_ref[pos], train_pred[pos], label = "Training")
scatter!(test_ref[pos], test_pred[pos], label = "Test")
