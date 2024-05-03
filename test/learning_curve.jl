using Statistics, PythonCall, JLD # , Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")

Ndata = 2000
rcut = 10.0
zcut = 10.0


# read data
filenames = ["data/propanol.h5"]#, "data/esanol.h5", "data/acrolein.h5", "data/phenol.h5", "data/toluene.h5", "data/acetaldehyde.h5", "data/aniline.h5", "data/nmacetamide.h5"]
frames = []
train_set = Int.(0:floor(10000/Ndata):9999)
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i in train_set ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

frames_test = []
test_set = rand(setdiff(0:9999, train_set), Ndata)
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# construct Models

degreeset = 2:6
ordset = 1:2
RE_train = zeros(length(ordset), length(degreeset))
RMSE_train = zeros(length(ordset), length(degreeset))
MV_train = zeros(length(ordset), length(degreeset))
RE_test = zeros(length(ordset), length(degreeset))
RMSE_test = zeros(length(ordset), length(degreeset))
MV_test = zeros(length(ordset), length(degreeset))

# DM = load("test/CHON_Models/model_maxdeg9_ord$(order)_rcut$(rcut)_zcut$(zcut).jld")|> read_dict
for (j, order) in enumerate(ordset)
    for (i, degree) in enumerate(degreeset)

        println("Constructing/Loading the order $order degree $degree model ...")
        println()

        try 
            global DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld")|> read_dict
        
            println("Model loaded!")
            println()
        catch
            ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut), 
                            6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                            # 7 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                            8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut) )
        
            global DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

            println("Model constructed!")
            println()
        end

        # save the unfitted model first for use later
        if haskey(DM.Models, 7)
            save("test/CHON_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))
        else
            save("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))
        end

        # fit the model, if it is not fully fitted yet
        if !isfitted(DM)
            fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
            # fit!(DM, frames)
        end

        # validate the model - Training
        for frame in frames
            R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
            D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
            RMSE_train[j,i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RE_train[j,i] += norm(D_pred - D)/norm(D)
            MV_train[j,i] += norm(D_pred * D_pred - D_pred)
        end
        RMSE_train[j,i] = sqrt(RMSE_train[j,i]/length(frames))
        RE_train[j,i] /= length(frames)
        MV_train[j,i] /= length(frames)

        println("Training RMSE per matrix element = $(RMSE_train[j,i])")
        println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[j,i])")
        println("Average training manifold violation = $(MV_train[j,i])")
        println()

        for frame in frames_test
            R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
            D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
            RMSE_test[j,i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RE_test[j,i] += norm(D_pred - D)/norm(D)
            MV_test[j,i] += norm(D_pred * D_pred - D_pred)
        end

        RMSE_test[j,i] = sqrt(RMSE_test[j,i]/length(frames_test))
        RE_test[j,i] /= length(frames_test)
        MV_test[j,i] /= length(frames_test)
    
        println("Test RMSE per matrix element = $(RMSE_test[j,i])")
        println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[j,i])")
        println("Average test manifold violation = $(MV_test[j,i])")
        println()

        println("Saving the model ...")
        println()

        if haskey(DM.Models, 7)
            save("test/CHON_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))
        else
            save("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))
        end

        println("Model saved!")
        println()

        println("Done for order $order degree $degree model")
        println()

    end
end

# visualize the error
using Plots

Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
    
plt = plot(degreeset, RMSE_train[1,:], label = "Order $(ordset[1]): Training RMSE", xlabel = "Degree", ylabel = "RMSE")
plot!(degreeset, RMSE_test[1,:], label = "Order $(ordset[1]): Test RMSE", linestyle = :dash)
for i in 2:size(RMSE_train,1)
    plot!(degreeset, RMSE_train[2,:], label = "Order $(ordset[i]): Training RMSE")
    plot!(degreeset, RMSE_test[2,:], label = "Order $(ordset[i]): Test RMSE", linestyle = :dash)
end
title!("RMSE vs Degree")
savefig("test/$Folder/RMSE_Order3&4_rcut$(rcut)_zcut$(zcut).png")

plt = plot(degreeset, RE_train[1,:], label = "Order $(ordset[1]): Training Relative Error", xlabel = "Degree", ylabel = "RE")
plot!(degreeset, RE_test[1,:], label = "Order $(ordset[1]): Test Relative Error", linestyle = :dash)
for i in 2:size(RE_train,1)
    plot!(degreeset, RE_train[2,:], label = "Order $(ordset[i]): Training Relative Error")
    plot!(degreeset, RE_test[2,:], label = "Order $(ordset[i]): Test Relative Error", linestyle = :dash)
end
title!("RE vs Degree")
savefig("test/$Folder/RE_Order3&4_rcut$(rcut)_zcut$(zcut).png")

plt = plot(degreeset, MV_train[1,:], label = "Order $(ordset[1]): Training Manifold Violation", xlabel = "Degree", ylabel = "MV")
plot!(degreeset, MV_test[1,:], label = "Order $(ordset[1]): Test Manifold Violation", linestyle = :dash)
for i in 2:size(MV_train,1)
    plot!(degreeset, MV_train[2,:], label = "Order $(ordset[i]): Training Manifold Violation")
    plot!(degreeset, MV_test[2,:], label = "Order $(ordset[i]): Test Manifold Violation", linestyle = :dash)
end
title!("MV vs Degree")
savefig("test/$Folder/MV_Order3&4_rcut$(rcut)_zcut$(zcut).png")

# converging meaning that we need to go to higher correlation order

# This observation inspires a fast way to construct the model
# [DM.Models[6,6].model.layers.AA2BB.layers[i].op == DM.Models[6,8].model.layers.AA2BB.layers[i].op for i = 1:9] |> all