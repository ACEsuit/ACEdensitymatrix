using Statistics, JLD
# using  PythonCall # PythonCall is used only when we need to call SKLearn

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")
include("../src/tuned_models.jl")

Ndata = 2000
rcut_on = 10.0
rcut_off = 10.0
zcut = 10.0


# read data
routine = "data/new_datasets"
filenames = ["$routine/propanol.h5"]# , "$routine/hexanol.h5", "$routine/acrolein.h5", "$routine/phenol.h5", "$routine/toluene.h5", "$routine/acetaldehyde.h5", "$routine/aniline.h5", "$routine/nmacetamide.h5"]
frames = []
train_set = 0:2999 # we take the first 2999 frames, as suggested in our test on the effect of number of data
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i in train_set ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

frames_test = []
test_set = 3000:9999 # rand(setdiff(0:9999, train_set), Ndata)
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# construct Models

# Here we generate the models with different orders and degrees
# to generate one specific model, just fixed these values and get 
# rid of the below loop.

degreeset = 2:6
ordset = 1:2
# RE_train = zeros(length(ordset), length(degreeset))
RMSE_train = zeros(length(ordset), length(degreeset))
RMSE_train_retract = zeros(length(ordset), length(degreeset))
MV_train = zeros(length(ordset), length(degreeset))
MV_train_retract = zeros(length(ordset), length(degreeset))
# RE_test = zeros(length(ordset), length(degreeset))
RMSE_test = zeros(length(ordset), length(degreeset))
RMSE_test_retract = zeros(length(ordset), length(degreeset))
MV_test = zeros(length(ordset), length(degreeset))
MV_test_retract = zeros(length(ordset), length(degreeset))

for (j, order) in enumerate(ordset)
    for (i, degree) in enumerate(degreeset)

        println("Constructing/Loading the order $order degree $degree model ...")
        println()
        refit = false

        try 
            global DM = load("test/CHO_Models/tuned/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld")|> read_dict
        
            println("Model loaded!")
            println()
        catch
            println("Model doesn't exist / fails to be loaded ...")
            println("Start construction ...")
            println()

            ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order+1, "rcut_on" => rcut_on, "rcut_off" => rcut_off, "zcut" => zcut), 
                            6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut_on" => rcut_on, "rcut_off" => rcut_off, "zcut" => zcut),
                            # 7 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut_on" => rcut_on, "rcut_off" => rcut_off, "zcut" => zcut),
                            8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut_on" => rcut_on, "rcut_off" => rcut_off, "zcut" => zcut) )
        
            global DM = Density_Model_tuned(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

            println("Model constructed!")
            println()

            # fit the model, if it is not fully fitted yet
            if !isfitted(DM)
                fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
                # fit!(DM, frames)
            end
            refit = true
        end

        # validate the model - Training
        for frame in frames
            R, D, atomic_number, ao_labels = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"]
            D_pred = eval_model(DM, R, ao_labels) # predicted density matrix
            D_pred_retract = eval_model(DM, R, ao_labels, retraction = D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix
            RMSE_train[j,i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_train_retract[j,i] += norm(D_pred_retract - D)^2/(size(D,1)*size(D,2))
            # RE_train[j,i] += norm(D_pred - D)/norm(D)
            MV_train[j,i] += norm(D_pred * D_pred - D_pred)
            MV_train_retract[j,i] += norm(D_pred_retract * D_pred_retract - D_pred_retract)
        end
        RMSE_train[j,i] = sqrt(RMSE_train[j,i]/length(frames))
        RMSE_train_retract[j,i] = sqrt(RMSE_train_retract[j,i]/length(frames))
        # RE_train[j,i] /= length(frames)
        MV_train[j,i] /= length(frames)
        MV_train_retract[j,i] /= length(frames)

        println("Training RMSE per matrix element = $(RMSE_train[j,i])")
        println("Training RMSE per matrix element (retracted) = $(RMSE_train_retract[j,i])")
        # println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[j,i])")
        println("Average training manifold violation = $(MV_train[j,i])")
        println("Average training manifold violation (retracted) = $(MV_train_retract[j,i])")
        println()

        for frame in frames_test
            R, D, atomic_number, ao_labels = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"]
            D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
            D_pred_retract = eval_model(DM, R, translate_frame(frame)["ao_labels"], retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            RMSE_test[j,i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_test_retract[j,i] += norm(D_pred_retract - D)^2/(size(D,1)*size(D,2))
            # RE_test[j,i] += norm(D_pred - D)/norm(D)
            MV_test[j,i] += norm(D_pred * D_pred - D_pred)
            MV_test_retract[j,i] += norm(D_pred_retract * D_pred_retract - D_pred_retract)
        end

        RMSE_test[j,i] = sqrt(RMSE_test[j,i]/length(frames_test))
        RMSE_test_retract[j,i] = sqrt(RMSE_test_retract[j,i]/length(frames_test))
        # RE_test[j,i] /= length(frames_test)
        MV_test[j,i] /= length(frames_test)
        MV_test_retract[j,i] /= length(frames_test)
    
        println("Test RMSE per matrix element = $(RMSE_test[j,i])")
        println("Test RMSE per matrix element (retracted) = $(RMSE_test_retract[j,i])")
        # println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[j,i])")
        println("Average test manifold violation = $(MV_test[j,i])")
        println("Average test manifold violation (retracted) = $(MV_test_retract[j,i])")
        println()

        if refit
            println("Saving the model ...")
            println()

            if haskey(DM.Models, 7)
                save("test/CHON_Models/tuned/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))
            else
                save("test/CHO_Models/tuned/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld", write_dict(DM))
            end

            println("Model saved!")
            println()
        end

        println("Done for order $order degree $degree model")
        println()

    end
end

# visualize the error
using Plots

Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
    
plt = plot(degreeset, log10.(RMSE_train[1,:]), label = "Order $(ordset[1]): Training RMSE", xlabel = "Degree", ylabel = "RMSE (10^y)", legendfontsize=7, color = 1)
plot!(degreeset, log10.(RMSE_test[1,:]), label = "Order $(ordset[1]): Test RMSE", linestyle = :dash, color = 1)
plot!(degreeset, log10.(RMSE_train_retract[1,:]), label = "Order $(ordset[1]): Training RMSE (retracted)", color = 1, markers = :diamond)
plot!(degreeset, log10.(RMSE_test_retract[1,:]), label = "Order $(ordset[1]): Test RMSE (retracted)", linestyle = :dash, color = 1, markers = :diamond)
for i in 2:size(RMSE_train,1)
    plot!(degreeset, log10.(RMSE_train[i,:]), label = "Order $(ordset[i]): Training RMSE", color = i)
    plot!(degreeset, log10.(RMSE_test[i,:]), label = "Order $(ordset[i]): Test RMSE", linestyle = :dash, color = i)
    plot!(degreeset, log10.(RMSE_train_retract[i,:]), label = "Order $(ordset[i]): Training RMSE (retracted)", color = i, marker = :diamond)
    plot!(degreeset, log10.(RMSE_test_retract[i,:]), label = "Order $(ordset[i]): Test RMSE (retracted)", linestyle = :dash, color = i, marker = :diamond)
end
title!("RMSE vs Degree")
savefig("test/$Folder/tuned/RMSE_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

# plt = plot(degreeset, log10.(RE_train[1,:]), label = "Order $(ordset[1]): Training Relative Error", xlabel = "Degree", ylabel = "RE (10^y)")
# plot!(degreeset, log10.(RE_test[1,:]), label = "Order $(ordset[1]): Test Relative Error", linestyle = :dash)
# for i in 2:size(RE_train,1)
#     plot!(degreeset, log10.(RE_train[i,:]), label = "Order $(ordset[i]): Training Relative Error")
#     plot!(degreeset, log10.(RE_test[i,:]), label = "Order $(ordset[i]): Test Relative Error", linestyle = :dash)
# end
# title!("RE vs Degree")
# savefig("test/$Folder/RE_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

plt = plot(degreeset, MV_train[1,:], label = "Order $(ordset[1]): Training Manifold Violation", xlabel = "Degree", ylabel = "||D_{pred}^2-D_{pred}||", legendfontsize=7, color = 1, legend = :topright)
plot!(degreeset, MV_test[1,:], label = "Order $(ordset[1]): Test Manifold Violation", linestyle = :dash, color = 1)
plot!(degreeset, MV_train_retract[1,:], label = "Order $(ordset[1]): Training Manifold Violation (retracted)", color = 1, marker = :diamond)
plot!(degreeset, MV_test_retract[1,:], label = "Order $(ordset[1]): Test Manifold Violation (retracted)", linestyle = :dash, color = 1, marker = :diamond)
for i in 2:size(MV_train,1)
    plot!(degreeset, MV_train[i,:], label = "Order $(ordset[i]): Training Manifold Violation", color = i)
    plot!(degreeset, MV_test[i,:], label = "Order $(ordset[i]): Test Manifold Violation", linestyle = :dash, color = i)
    plot!(degreeset, MV_train_retract[i,:], label = "Order $(ordset[i]): Training Manifold Violation (retracted)", color = i, marker = :diamond)
    plot!(degreeset, MV_test_retract[i,:], label = "Order $(ordset[i]): Test Manifold Violation (retracted)", linestyle = :dash, color = i, marker = :diamond)
end
title!("Manifold Violation vs Degree")
savefig("test/$Folder/tuned/MV_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

# converging meaning that we need to go to higher correlation order