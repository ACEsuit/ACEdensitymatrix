using Statistics, PythonCall, JLD # , Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")

Ndata = 100
rcut = 10.0
zcut = 10.0
order = 1

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

# construct Models
degreeset = [i for i = 2:6]
RE_train = zeros(length(degreeset))
RMSE_train = zeros(length(degreeset))
MV_train = zeros(length(degreeset))
RE_test = zeros(length(degreeset))
RMSE_test = zeros(length(degreeset))
MV_test = zeros(length(degreeset))

for (i, degree) in enumerate(degreeset)

    # if model exists
    #     read model from file
    # else
        ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut), 
                        6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                        7 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                        8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut) )
    
        println("Constructing the order $order degree $degree model ...")
        println()

        DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

        println("Model constructed!")
        println()
    # end

    # fit the model
    fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))

    # validate the model - Training
    for frame in frames
        R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
        D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
        RMSE_train[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
        RE_train[i] += norm(D_pred - D)/norm(D)
        MV_train[i] += norm(D_pred * D_pred - D_pred)
    end
    RMSE_train[i] = sqrt(RMSE_train[i]/length(frames))
    RE_train[i] /= length(frames)
    MV_train[i] /= length(frames)

    println("Training RMSE per matrix element = $(RMSE_train[i])")
    println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[i])")
    println("Average training manifold violation = $(MV_train[i])")
    println()

    for frame in frames_test
        R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
        D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
        RMSE_test[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
        RE_test[i] += norm(D_pred - D)/norm(D)
        MV_test[i] += norm(D_pred * D_pred - D_pred)
    end

    RMSE_test[i] = sqrt(RMSE_test[i]/length(frames_test))
    RE_test[i] /= length(frames_test)
    MV_test[i] /= length(frames_test)
    
    println("Test RMSE per matrix element = $(RMSE_test[i])")
    println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[i])")
    println("Average test manifold violation = $(MV_test[i])")
    println()

    println("Saving the model ...")
    println()

    save("test/CHON_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))

    println("Model saved!")
    println()

    println("Done for order $order degree $degree model")

end

# visualize the error
plt = plot(degreeset, RMSE_train, label = "Training RMSE", xlabel = "Degree", ylabel = "RMSE")
plot!(degreeset, RMSE_test, label = "Test RMSE")
savefig("RMSE_Order$(order)_rcut$(rcut)_zcut$(zcut)")

plt = plot(degreeset, RE_train, label = "Training Relative Error", xlabel = "Degree", ylabel = "RE")
plot!(degreeset, RE_test, label = "Test Relative Error")
savefig("RE_Order$(order)_rcut$(rcut)_zcut$(zcut)")

plt = plot(degreeset, MV_train, label = "Training Manifold Violation", xlabel = "Degree", ylabel = "MV")
plot!(degreeset, MV_test, label = "Test Manifold Violation")
savefig("MV_Order$(order)_rcut$(rcut)_zcut$(zcut)")

MV_train

# converging meaning that we need to go to higher correlation order