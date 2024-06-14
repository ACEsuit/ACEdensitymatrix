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
filenames = ["$routine/hexanol_KS.h5"]#, "$routine/acrolein.h5", "$routine/phenol.h5", "$routine/toluene.h5", "$routine/acetaldehyde.h5", "$routine/aniline.h5", "$routine/nmacetamide.h5"]
frames = []
train_set = 0:1:Ndata-1 # Int.(0:floor(10000/Ndata):9999)
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i in train_set ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

frames_test = []
test_set = 5000:50:9999# rand(setdiff(0:9999, train_set), Ndata)
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# construct Models

degreeset = 2:10
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
            global DM = load("test/CHO_Models/Hexanol/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict
        
            println("DM Model loaded!")
            println()
        catch
            DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict
            # println("DM Model doesn't exist / fails to be loaded ...")
            # println("Start construction ...")
            # println()

            # ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut), 
            #                 6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
            #                 # 7 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
            #                 8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut) )
        
            # global DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

            # println("Model constructed!")
            # println()

            # fit the model, if it is not fully fitted yet
            if !isfitted(DM)
                fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
                Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
                save("test/$Folder/Hexanol/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2", write_dict(DM))
            end
            # refit_D = true
        end

        # construct or load the Hamiltonian model
        try 
            global HM = load("test/CHO_Models/Hamiltonian/Hexanol/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict
        
            println("HM Model loaded!")
            println()
        catch
            println("HM Model doesn't exist / fails to be loaded ...")
            println("Make use of the DM, but subject to refit...")
                
            global HM = load("test/CHO_Models/Hexanol/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict
                
            println("Model copied!")
            println()
            
            # fit the model
            fit!(HM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I), Mode = "H")
            Folder = haskey(HM.Models, 7) ? "CHON_Models" : "CHO_Models"
            save("test/$Folder/Hamiltonian/Hexanol/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2", write_dict(HM))
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

RMSE_D_train[3,6:7] = [0 0]
E_D_train[3,6:7] = [0 0]
RMSE_D_test[3,6:7] = [0 0]
E_D_test[3,6:7] = [0 0]

DM_tuned = load("test/CHO_Models/tuned/model_maxdeg8_ord2_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict
HM_tuned = load("test/CHO_Models/Hamiltonian/tuned/model_maxdeg8_ord2_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict

RMSE_tuned_D_train = 0.0
RMSE_tuned_DfromH_train = 0.0
RMSE_tuned_H_train = 0.0
E_tuned_D_train = 0.0
E_tuned_DfromH_train = 0.0
E_tuned_H_train = 0.0
for frame in frames
    R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
    D_pred = eval_model(DM_tuned, R, translate_frame(frame)["ao_labels"], retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
    H_pred = eval_model(HM_tuned, R, translate_frame(frame)["ao_labels"]) # predicted KS matrix
    RMSE_tuned_D_train += norm(D_pred - D)^2/(size(D,1)*size(D,2))
    RMSE_tuned_H_train += norm(H_pred - H)^2/(size(H,1)*size(H,2))
    # RE_train[j,i] += norm(D_pred - D)/norm(D)
    E_tuned_D_train += norm(D_pred - D)
    E_tuned_H_train += norm(H_pred - H)
    s, q = eigen(Symmetric(S))
    s_half = q * Diagonal(s.^(-1/2)) * q'
    H_pred = Symmetric(s_half * H_pred * s_half)
    C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
    D_pred_H = C_pred * C_pred'
    RMSE_tuned_DfromH_train += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
    E_tuned_DfromH_train += norm(D_pred_H - D)
end

RMSE_tuned_D_train = sqrt(RMSE_tuned_D_train/length(frames))
RMSE_tuned_DfromH_train = sqrt(RMSE_tuned_DfromH_train/length(frames))
RMSE_tuned_H_train = sqrt(RMSE_tuned_H_train/length(frames))
# RE_train[j,i] /= length(frames)
E_tuned_D_train /= length(frames)
E_tuned_DfromH_train /= length(frames)
E_tuned_H_train /= length(frames)

println("Training RMSE in D per matrix element = $(RMSE_tuned_D_train)")
println("Training RMSE in D (from H) per matrix element = $(RMSE_tuned_DfromH_train)")
println("Training RMSE in H per matrix element = $(RMSE_tuned_H_train)")
# println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[j,i])")
println("Average training error in D: ||D - D_ref|| = $(E_tuned_D_train)")
println("Average training error in D (from H): ||D - D_ref|| = $(E_tuned_DfromH_train)")
println("Average training error in H: ||H - H_ref|| = $(E_tuned_H_train)")
println()

RMSE_tuned_D_test = 0.0
RMSE_tuned_DfromH_test = 0.0
RMSE_tuned_H_test = 0.0
E_tuned_D_test = 0.0
E_tuned_DfromH_test = 0.0
E_tuned_H_test = 0.0
for frame in frames_test
    R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
    D_pred = eval_model(DM_tuned, R, translate_frame(frame)["ao_labels"], retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
    H_pred = eval_model(HM_tuned, R, translate_frame(frame)["ao_labels"]) # predicted KS matrix
    RMSE_tuned_D_test += norm(D_pred - D)^2/(size(D,1)*size(D,2))
    RMSE_tuned_H_test += norm(H_pred - H)^2/(size(H,1)*size(H,2))
    # RE_train[j,i] += norm(D_pred - D)/norm(D)
    E_tuned_D_test += norm(D_pred - D)
    E_tuned_H_test += norm(H_pred - H)
    s, q = eigen(Symmetric(S))
    s_half = q * Diagonal(s.^(-1/2)) * q'
    H_pred = Symmetric(s_half * H_pred * s_half)
    C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
    D_pred_H = C_pred * C_pred'
    RMSE_tuned_DfromH_test += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
    E_tuned_DfromH_test += norm(D_pred_H - D)
end

RMSE_tuned_D_test = sqrt(RMSE_tuned_D_test/length(frames_test))
RMSE_tuned_DfromH_test = sqrt(RMSE_tuned_DfromH_test/length(frames_test))
RMSE_tuned_H_test = sqrt(RMSE_tuned_H_test/length(frames_test))
# RE_train[j,i] /= length(frames_test)
E_tuned_D_test /= length(frames_test)
E_tuned_DfromH_test /= length(frames_test)
E_tuned_H_test /= length(frames_test)

println("Test RMSE in D per matrix element = $(RMSE_tuned_D_test)")
println("Test RMSE in D (from H) per matrix element = $(RMSE_tuned_DfromH_test)")
println("Test RMSE in H per matrix element = $(RMSE_tuned_H_test)")
# println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[j,i])")
println("Average test error in D: ||D - D_ref|| = $(E_tuned_D_test)")
println("Average test error in D (from H): ||D - D_ref|| = $(E_tuned_DfromH_test)")
println("Average test error in H: ||H - H_ref|| = $(E_tuned_H_test)")

# visualize the error
using Plots

Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
    
plt = plot(degreeset, log10.(RMSE_D_train[1,:]), label = "Order $(ordset[1]): Training RMSE (D)", xlabel = "Degree", ylabel = "RMSE (10^y)", legendfontsize=7, color = 1, legend = :outerbottomright)
plot!(degreeset, log10.(RMSE_D_test[1,:]), label = "Order $(ordset[1]): Test RMSE (D)", linestyle = :dash, color = 1)
plot!(degreeset, log10.(RMSE_DfromH_train[1,:]), label = "Order $(ordset[1]): Training RMSE (D from H)", color = 2)
plot!(degreeset, log10.(RMSE_DfromH_test[1,:]), label = "Order $(ordset[1]): Test RMSE (D from H)", linestyle = :dash, color = 2)
plot!(degreeset, log10.(RMSE_H_train[1,:]), label = "Order $(ordset[1]): Training RMSE (H)", color = 2, marker = :diamond)
plot!(degreeset, log10.(RMSE_H_test[1,:]), label = "Order $(ordset[1]): Test RMSE (H)", linestyle = :dash, color = 2, marker = :diamond)

for i in 2:size(RMSE_D_train,1)
    plot!(degreeset, log10.(RMSE_D_train[i,:]), label = "Order $(ordset[i]): Training RMSE (D)", color = 2i-1)
    plot!(degreeset, log10.(RMSE_D_test[i,:]), label = "Order $(ordset[i]): Test RMSE (D)", linestyle = :dash, color = 2i-1)
    plot!(degreeset, log10.(RMSE_DfromH_train[i,:]), label = "Order $(ordset[i]): Training RMSE (D from H)", color = 2i)
    plot!(degreeset, log10.(RMSE_DfromH_test[i,:]), label = "Order $(ordset[i]): Test RMSE (D from H)", linestyle = :dash, color = 2i)
    plot!(degreeset, log10.(RMSE_H_train[i,:]), label = "Order $(ordset[i]): Training RMSE (H)", color = 2i, marker = :diamond)
    plot!(degreeset, log10.(RMSE_H_test[i,:]), label = "Order $(ordset[i]): Test RMSE (H)", linestyle = :dash, color = 2i, marker = :diamond)
end
plot!(degreeset, log10(RMSE_tuned_D_test).*ones(length(degreeset)), label = "(2,8) - Tuned Test RMSE (D)", color = 7, linestyle = :dash)
plot!(degreeset, log10(RMSE_tuned_DfromH_test).*ones(length(degreeset)), label = "(2,8) - Tuned Test RMSE (D from H)", linestyle = :dash, color = 8)
ylims!(-4.0, -1.4)
title!("RMSE in D and H vs Degree")
savefig("test/$Folder/DH_RMSE_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

## Exploring the error in H
plt = plot(degreeset, log10.(E_H_train[1,:]), label = "Order $(ordset[1]): Training Error in H", xlabel = "Degree", ylabel = "Error (10^y)", legendfontsize=7, color = 1)
plot!(degreeset, log10.(E_H_test[1,:]), label = "Order $(ordset[1]): Test Error in H", linestyle = :dash, color = 1)
for i in 2:size(E_H_train,1)
    plot!(degreeset, log10.(E_H_train[i,:]), label = "Order $(ordset[i]): Training Error in H", color = i)
    plot!(degreeset, log10.(E_H_test[i,:]), label = "Order $(ordset[i]): Test Error in H", linestyle = :dash, color = i)
end
title!("Error in H vs Degree")

log10.(E_H_train)
E_guess = [-0.0946, -1.6, -2.04993 - (1.6759 - 1.37707)]
E_guess = [10^E_guess[1], 10^E_guess[2], 10^E_guess[3]]
y = log.(E_guess)
x = [1, 2, 3]
x = [x'; ones(length(x))']'
k = x \ y

xx = 1:0.5:3
yy = k[1].*xx .+ k[2]
plot!(xx, yy)
b = k[1]
a = exp(k[2])

plot(xx, a.*exp.(b*xx))
plot!([1,2,3], E_guess, seriestype=:scatter)
# plt = plot(degreeset, log10.(RE_train[1,:]), label = "Order $(ordset[1]): Training Relative Error", xlabel = "Degree", ylabel = "RE (10^y)")
# plot!(degreeset, log10.(RE_test[1,:]), label = "Order $(ordset[1]): Test Relative Error", linestyle = :dash)
# for i in 2:size(RE_train,1)
#     plot!(degreeset, log10.(RE_train[i,:]), label = "Order $(ordset[i]): Training Relative Error")
#     plot!(degreeset, log10.(RE_test[i,:]), label = "Order $(ordset[i]): Test Relative Error", linestyle = :dash)
# end
# title!("RE vs Degree")
# savefig("test/$Folder/RE_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

# converging meaning that we need to go to higher correlation order

## Let's track down the relationship between error in H and the resulting error in D
E_DfromH_train_v = [ 0 < E_DfromH_train[i,j] < 1 ? E_DfromH_train[i,j] : 0 for i in 1:size(E_DfromH_train,1), j in 1:size(E_DfromH_train,2)]
E_DfromH_test_v = [ 0 < E_DfromH_test[i,j] < 1 ? E_DfromH_test[i,j] : 0 for i in 1:size(E_DfromH_test,1), j in 1:size(E_DfromH_test,2)]
E_H_train_v = [ 0 < E_H_train[i,j] < 1 ? E_H_train[i,j] : 0 for i in 1:size(E_H_train,1), j in 1:size(E_H_train,2)]
E_H_test_v = [ 0 < E_H_test[i,j] < 1 ? E_H_test[i,j] : 0 for i in 1:size(E_H_test,1), j in 1:size(E_H_test,2)]

E_DH = [zip(E_H_train[i,:], E_DfromH_train[i,:]) for i in 1:size(E_H_train,1)]


plt = plot(log10.(E_H_train_v[1,:]), log10.(E_DfromH_train_v[1,:]), label = "Order $(ordset[1]): Training Error Propagation from H to D", xlabel = "Error on H (10^x)", ylabel = "Error on D (10^y)", legendfontsize=7, color = 1, seriestype=:scatter)
plot!(log10.(E_H_test_v[1,:]), log10.(E_DfromH_test_v[1,:]), label = nothing, color = 1, seriestype=:scatter, marker = :diamond)
for i = 2:size(E_H_test_v,1)
    plot!(log10.(E_H_train_v[i,:]), log10.(E_DfromH_train_v[i,:]), label = "Order $(ordset[i]) Error Propagation from H to D", xlabel = "Error on H (10^x)", ylabel = "Error on D (10^y)", legendfontsize=7, color = i, seriestype=:scatter)
    plot!(log10.(E_H_test_v[i,:]), log10.(E_DfromH_test_v[i,:]), label =nothing, color = i, seriestype=:scatter, marker = :diamond)
end

X = [log10.(E_H_train_v[1,:])..., log10.(E_H_test_v[1,:])...]
Y = [log10.(E_DfromH_train_v[1,:])..., log10.(E_DfromH_test_v[1,:])...] .- (2/3) .* X
Y = Y[findall(x -> x != -Inf, X)]
X = X[findall(x -> x != -Inf, X)]
# X = [X'; ones(length(X))']'
XX = reshape(ones(length(X)), (length(X), 1))
# X = reshape(X, (length(X), 1))
# Y = Y[findall(x -> x != -Inf, Y)]

k1 = XX'XX \ XX'Y
plot!([-2, 0], 2/3*[-2, 0] .+ (k1[1]), label = "y = $(k1[1]) * x", color = 3, linestyle = :dash)
# plot!([-1.6, -0.5], k[1]*[-1.6, -0.5], label = "y = $(k[1]) * x", color = 3, linestyle = :dash)

X = [log10.(E_H_train_v[2,:])..., log10.(E_H_test_v[2,:])...]
Y = [log10.(E_DfromH_train_v[2,:])..., log10.(E_DfromH_test_v[2,:])...] .- (2/3) .* X
Y = Y[findall(x -> x != -Inf, X)]
X = X[findall(x -> x != -Inf, X)]
# X = [X'; ones(length(X))']'
XX = reshape(ones(length(X)), (length(X), 1))
# X = reshape(X, (length(X), 1))
# Y = Y[findall(x -> x != -Inf, Y)]
k2 = XX'XX \ XX'Y
plot!([-2.2, 0], 2/3*[-2.2, 0] .+ (k2[1]), label = "y = $(k2[1]) * x", color = 4, linestyle = :dash)

plt = plot(E_H_train[1,:], E_DfromH_train[1,:], label = "Order $(ordset[1]): Training Error Propagation from H to D", xlabel = "Error on H", ylabel = "Error on D", legendfontsize=7, color = 1, seriestype=:scatter)
plot!(E_H_test[1,:], E_DfromH_test[1,:], label = nothing, color = 1, seriestype=:scatter, marker = :diamond)
for i = 2:size(E_H_test_v,1)
    plot!(E_H_train[i,:], E_DfromH_train[i,:], label = "Order $(ordset[i]) Error Propagation from H to D", xlabel = "Error on H", ylabel = "Error on D", legendfontsize=7, color = i, seriestype=:scatter)
    plot!(E_H_test[i,:], E_DfromH_test[i,:], label =nothing, color = i, seriestype=:scatter, marker = :diamond)
end
xlims!(0, .4)
ylims!(0, .2)
xx = 0:0.01:1
yy = 10^k2[1] .* xx.^(2/3)
plot!(xx, yy, label = "y = $(k2[1]) * x^(2/3)", color = 3, linestyle = :dash)

title!("Error Propagation from H to D (Lienar relationship)")