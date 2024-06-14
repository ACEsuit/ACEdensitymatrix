using Statistics, JLD2 
# using  PythonCall # PythonCall is used only when we need to call SKLearn

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")

Ndata = 3000
rcut = 10.0
zcut = 10.0


# read data
routine = "data/new_datasets"
filenames = ["$routine/propanol_KS.h5"]#, "$routine/hexanol.h5", "$routine/acrolein.h5", "$routine/phenol.h5", "$routine/toluene.h5", "$routine/acetaldehyde.h5", "$routine/aniline.h5", "$routine/nmacetamide.h5"]
frames = []
train_set = 0:1:Ndata-1 # Int.(0:floor(10000/Ndata):9999)
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i in train_set ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

frames_test = []
test_set = 5000:10:9999# rand(setdiff(0:9999, train_set), Ndata)
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# construct Models

degreeset = 11:12
ordset = 1:2
# RE_train = zeros(length(ordset), length(degreeset))
RMSE_D_train = zeros(length(ordset), length(degreeset))
RMSE_DfromH_train = zeros(length(ordset), length(degreeset))
RMSE_H_train = zeros(length(ordset), length(degreeset))
E_D_train = zeros(length(ordset), length(degreeset))
E_DfromH_train = zeros(length(ordset), length(degreeset))
E_H_train = zeros(length(ordset), length(degreeset))
# RE_test = zeros(length(ordset), length(degreeset))
RMSE_D_test = zeros(length(ordset), length(degreeset))
RMSE_DfromH_test = zeros(length(ordset), length(degreeset))
RMSE_H_test = zeros(length(ordset), length(degreeset))
E_D_test = zeros(length(ordset), length(degreeset))
E_DfromH_test = zeros(length(ordset), length(degreeset))
E_H_test = zeros(length(ordset), length(degreeset))

for (j, order) in enumerate(ordset)
    for (i, degree) in enumerate(degreeset)

        println("Constructing/Loading the order $order degree $degree models ...")
        println()

        # construct or load the Density Matrix model
        try 
            global DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict
        
            println("DM Model loaded!")
            println()
        catch
            println("DM Model doesn't exist / fails to be loaded ...")
            println("Start construction ...")
            println()

            ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut), 
                            6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                            # 7 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                            8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut) )
        
            global DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

            println("Model constructed!")
            println()

            # fit the model, if it is not fully fitted yet
            if !isfitted(DM)
                fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
                Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
                save("test/$Folder/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2", write_dict(DM))
            end
            refit_D = true
        end

        # construct or load the Hamiltonian model
        try 
            global HM = load("test/CHO_Models/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict
        
            println("HM Model loaded!")
            println()
        catch
            println("HM Model doesn't exist / fails to be loaded ...")
            println("Make use of the DM, but subject to refit...")
                
            global HM = copy(DM)
                
            println("Model copied!")
            println()
            
            # fit the model
            fit!(HM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I), Mode = "H")
            Folder = haskey(HM.Models, 7) ? "CHON_Models" : "CHO_Models"
            save("test/$Folder/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2", write_dict(HM))
        end

        # validate the model - Training
        for frame in frames
            R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
            D_pred = eval_model(DM, R, ao_labels, retraction = D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            H_pred = eval_model(HM, R, ao_labels) # predicted KS matrix
            RMSE_D_train[j,i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_H_train[j,i] += norm(H_pred - H)^2/(size(H,1)*size(H,2))
            # RE_train[j,i] += norm(D_pred - D)/norm(D)
            E_D_train[j,i] += norm(D_pred - D)
            E_H_train[j,i] += norm(H_pred - H)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred_H = C_pred * C_pred'
            RMSE_DfromH_train[j,i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
            E_DfromH_train[j,i] += norm(D_pred_H - D)
        end
        RMSE_D_train[j,i] = sqrt(RMSE_D_train[j,i]/length(frames))
        RMSE_DfromH_train[j,i] = sqrt(RMSE_DfromH_train[j,i]/length(frames))
        RMSE_H_train[j,i] = sqrt(RMSE_H_train[j,i]/length(frames))
        # RE_train[j,i] /= length(frames)
        E_D_train[j,i] /= length(frames)
        E_DfromH_train[j,i] /= length(frames)
        E_H_train[j,i] /= length(frames)

        println("Training RMSE in D per matrix element = $(RMSE_D_train[j,i])")
        println("Training RMSE in D (from H) per matrix element = $(RMSE_DfromH_train[j,i])")
        println("Training RMSE in H per matrix element = $(RMSE_H_train[j,i])")
        # println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[j,i])")
        println("Average training error in D: ||D - D_ref|| = $(E_D_train[j,i])")
        println("Average training error in D (from H): ||D - D_ref|| = $(E_DfromH_train[j,i])")
        println("Average training error in H: ||H - H_ref|| = $(E_H_train[j,i])")
        println()

        for frame in frames_test
            R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
            D_pred = eval_model(DM, R, ao_labels, retraction = D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            H_pred = eval_model(HM, R, ao_labels) # predicted KS matrix
            RMSE_D_test[j,i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_H_test[j,i] += norm(H_pred - H)^2/(size(H,1)*size(H,2))
            # RE_test[j,i] += norm(D_pred - D)/norm(D)
            E_D_test[j,i] += norm(D_pred - D)
            E_H_test[j,i] += norm(H_pred - H)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred_H = C_pred * C_pred'
            RMSE_DfromH_test[j,i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
            E_DfromH_test[j,i] += norm(D_pred_H - D)
        end

        RMSE_D_test[j,i] = sqrt(RMSE_D_test[j,i]/length(frames_test))
        RMSE_DfromH_test[j,i] = sqrt(RMSE_DfromH_test[j,i]/length(frames_test))
        RMSE_H_test[j,i] = sqrt(RMSE_H_test[j,i]/length(frames_test))
        # RE_test[j,i] /= length(frames_test)
        E_D_test[j,i] /= length(frames_test)
        E_DfromH_test[j,i] /= length(frames_test)
        E_H_test[j,i] /= length(frames_test)
    
        println("Test RMSE in D per matrix element = $(RMSE_D_test[j,i])")
        println("Test RMSE in D (from H) per matrix element = $(RMSE_DfromH_test[j,i])")
        println("Test RMSE in H per matrix element = $(RMSE_H_test[j,i])")
        # println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[j,i])")
        println("Average test error in D: ||D - D_ref|| = $(E_D_test[j,i])")
        println("Average test error in D (from H): ||D - D_ref|| = $(E_DfromH_test[j,i])")
        println("Average test error in H: ||H - H_ref|| = $(E_H_test[j,i])")
        println()

        println("Done for order $order degree $degree model")
        println()

    end
end

DM_tuned = load("test/CHO_Models/tuned/model_maxdeg8_ord2_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict

RMSE_tuned_train = 0.0
RMSE_tuned_train_retract = 0.0
MV_tuned_train = 0.0
MV_tuned_train_retract = 0.0
for frame in frames
    R, D, atomic_number, ao_labels = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"]
    D_pred = eval_model(DM_tuned, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
    D_pred_retract = eval_model(DM_tuned, R, translate_frame(frame)["ao_labels"], retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
    RMSE_tuned_train += norm(D_pred - D)^2/(size(D,1)*size(D,2))
    RMSE_tuned_train_retract += norm(D_pred_retract - D)^2/(size(D,1)*size(D,2))
    # RE_train[j,i] += norm(D_pred - D)/norm(D)
    MV_tuned_train += norm(D_pred * D_pred - D_pred)
    MV_tuned_train_retract += norm(D_pred_retract * D_pred_retract - D_pred_retract)
end

RMSE_tuned_train = sqrt(RMSE_tuned_train/length(frames))
RMSE_tuned_train_retract = sqrt(RMSE_tuned_train_retract/length(frames))
# RE_train[j,i] /= length(frames)
MV_tuned_train /= length(frames)
MV_tuned_train_retract /= length(frames)

println("Training RMSE per matrix element = $(RMSE_tuned_train)")
println("Training RMSE per matrix element (retracted) = $(RMSE_tuned_train_retract)")
# println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[j,i])")
println("Average training manifold violation = $(MV_tuned_train)")
println("Average training manifold violation (retracted) = $(MV_tuned_train_retract)")
println()

RMSE_tuned_test = 0.0
RMSE_tuned_test_retract = 0.0
MV_tuned_test = 0.0
MV_tuned_test_retract = 0.0
for frame in frames_test
    R, D, atomic_number, ao_labels = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"]
    D_pred = eval_model(DM_tuned, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
    D_pred_retract = eval_model(DM_tuned, R, translate_frame(frame)["ao_labels"], retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
    RMSE_tuned_test += norm(D_pred - D)^2/(size(D,1)*size(D,2))
    RMSE_tuned_test_retract += norm(D_pred_retract - D)^2/(size(D,1)*size(D,2))
    # RE_test[j,i] += norm(D_pred - D)/norm(D)
    MV_tuned_test += norm(D_pred * D_pred - D_pred)
    MV_tuned_test_retract += norm(D_pred_retract * D_pred_retract - D_pred_retract)
end

RMSE_tuned_test = sqrt(RMSE_tuned_test/length(frames_test))
RMSE_tuned_test_retract = sqrt(RMSE_tuned_test_retract/length(frames_test))
# RE_test[j,i] /= length(frames_test)
MV_tuned_test /= length(frames_test)
MV_tuned_test_retract /= length(frames_test)

println("Test RMSE per matrix element = $(RMSE_tuned_test)")
println("Test RMSE per matrix element (retracted) = $(RMSE_tuned_test_retract)")
# println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[j,i])")
println("Average test manifold violation = $(MV_tuned_test)")
println("Average test manifold violation (retracted) = $(MV_tuned_test_retract)")
println()

##
# patch the error
# RMSE_train[2,8] = 0.0008777143183205528
# RMSE_train_retract[2,8] = 0.0006354771330419884
# MV_train[2,8] = 0.042757420352285004
# MV_train_retract[2,8] = 2.691017144582815e-13

# RMSE_test[2,8] = 0.0008955264057627274
# RMSE_test_retract[2,8] = 0.0006488052536318546
# MV_test[2,8] = 0.04340727096890023
# MV_test_retract[2,8] = 2.66460083065516e-13

# RMSE_train[2,9] = 0.0008195221551401519
# RMSE_train_retract[2,9] = 0.0005955033322326156
# MV_train[2,9] = 0.039778458414063404
# MV_train_retract[2,9] = 2.5967542239624364e-13

# RMSE_test[2,9] = 0.0008488733604316063
# RMSE_test_retract[2,9] = 0.0006162102541558284
# MV_test[2,9] = 0.0406484285103057
# MV_test_retract[2,9] = 2.6407226216579404e-13
##

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
plot!(degreeset, log10(RMSE_tuned_train_retract).*ones(length(degreeset)), label = "Tuned Training RMSE", color = 7, marker = :diamond)
title!("RMSE vs Degree")
savefig("test/$Folder/RMSE_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

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
savefig("test/$Folder/MV_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

# converging meaning that we need to go to higher correlation order

