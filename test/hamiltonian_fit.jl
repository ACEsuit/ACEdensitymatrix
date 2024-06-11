using Statistics, PythonCall, JLD2 # , Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")
include("../src/tuned_models.jl")

Ndata = 3000

# read data 
routine = "data/new_datasets"
filenames = ["$routine/propanol_KS.h5"]# , "$routine/hexanol.h5", "$routine/acrolein.h5", "$routine/phenol.h5", "$routine/toluene.h5", "$routine/acetaldehyde.h5", "$routine/aniline.h5", "$routine/nmacetamide.h5"]
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
for order = 2:2
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

# validate the model - Training
order = 2
rcut = 10.0
zcut = 10.0
degreeset = 2:10
RE_D = zeros(length(degreeset))
RE_H_D = zeros(length(degreeset))
RE_D_test = zeros(length(degreeset))
RE_H_D_test = zeros(length(degreeset))
for (i,degree) in enumerate(degreeset)
    DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict
    HM = load("test/CHO_Models/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict
    for frame in frames
        R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
        D_pred = eval_model(DM, R, ao_lab; retraction = D -> eigen_retraction(D, Int(sum(at_no)/2))) # predicted density matrix with retraction
        RE_D[i] += norm(D_pred - D)/norm(D)
        H_pred = eval_model(HM, R, ao_lab; retraction = identity) # predicted density matrix w/o retraction
        s, q = eigen(Symmetric(S))
        s_half = q * Diagonal(s.^(-1/2)) * q'
        H_pred = Symmetric(s_half * H_pred * s_half)
        C_pred = eigen(H_pred).vectors[:,1:Int(sum(at_no)/2)]
        D_pred_H = C_pred * C_pred'
        RE_H_D[i] += norm(D_pred_H - D)/norm(D)
        push!(train_ref, vec(D)...)
        push!(train_pred, vec(D_pred)...)
        push!(train_pred_H, vec(D_pred_H)...)
    end
    # println("Training RMSE per matrix element = $(sqrt(RMSE/length(frames)))")
    RE_D[i] /= length(frames)
    RE_H_D[i] /= length(frames)
    println("Average training relative error (directly trained on D) for degree $degree: ||D_pred - D_ref|| / ||D||= $(RE_D[i])")
    println("Average training relative error (trained on H and then D) for degree $degree: ||D_pred_H - D_ref|| / ||D||= $(RE_H_D[i])")
    println()

    # validate the model - Test
    for frame in frames_test
        R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
        D_pred = eval_model(DM, R, ao_lab; retraction = D -> eigen_retraction(D, Int(sum(at_no)/2))) # predicted density matrix with retraction
        RE_D_test[i] += norm(D_pred - D)/norm(D)
        H_pred = eval_model(HM, R, ao_lab; retraction = identity) # predicted density matrix w/o retraction
        s, q = eigen(Symmetric(S))
        s_half = q * Diagonal(s.^(-1/2)) * q'
        H_pred = Symmetric(s_half * H_pred * s_half)
        C_pred = eigen(H_pred).vectors[:,1:Int(sum(at_no)/2)]
        D_pred_H = C_pred * C_pred'
        RE_H_D_test[i] += norm(D_pred_H - D)/norm(D)
        push!(test_ref, vec(D)...)
        push!(test_pred, vec(D_pred)...)
        push!(test_pred_H, vec(D_pred_H)...)
    end
    # println("Test RMSE per matrix element = $(sqrt(RMSE/length(frames_test)))")
    RE_D_test[i] /= length(frames_test)
    RE_H_D_test[i] /= length(frames_test)
    println("Average test relative error (directly trained on D) for degree $degree: ||D_pred - D_ref|| / ||D||= $(RE_D_test[i])")
    println("Average test relative error (trained on H and then D) for degree $degree: ||D_pred_H - D_ref|| / ||D||= $(RE_H_D_test[i])")
    println()
end

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

Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
    
plt = plot(degreeset, log10.(RE_D), label = "Order $order: Training Relative Error", xlabel = "Degree", ylabel = "RE (10^y)", legendfontsize=7, color = 1)
plot!(degreeset, log10.(RE_D_test), label = "Order $order: Test Relative Error", linestyle = :dash, color = 1)
plot!(degreeset, log10.(RE_H_D), label = "Order $order: Training Relative Error (H => D)", color = 2, marker = :diamond)
plot!(degreeset, log10.(RE_H_D_test), label = "Order $order: Test Relative Error (H => D)", linestyle = :dash, color = 2, marker = :diamond)
title!("RE vs Degree")
savefig("test/$Folder/Hamiltonian/RE_Order$(order)_rcut$(rcut)_zcut$(zcut).png")

# plt = plot(degreeset, log10.(RE_train[1,:]), label = "Order $(ordset[1]): Training Relative Error", xlabel = "Degree", ylabel = "RE (10^y)")
# plot!(degreeset, log10.(RE_test[1,:]), label = "Order $(ordset[1]): Test Relative Error", linestyle = :dash)
# for i in 2:size(RE_train,1)
#     plot!(degreeset, log10.(RE_train[i,:]), label = "Order $(ordset[i]): Training Relative Error")
#     plot!(degreeset, log10.(RE_test[i,:]), label = "Order $(ordset[i]): Test Relative Error", linestyle = :dash)
# end
# title!("RE vs Degree")
# savefig("test/$Folder/RE_Order$(minimum(ordset))-$(maximum(ordset))_rcut$(rcut)_zcut$(zcut).png")

degree = 10
order = 2
rcut = 10.0
zcut = 10.0
DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict
HM = load("test/CHO_Models/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict

train_ref = Vector{Float64}()
train_pred = Vector{Float64}()
train_pred_H = Vector{Float64}()
test_ref = Vector{Float64}()
test_pred = Vector{Float64}()
test_pred_H = Vector{Float64}()

for frame in frames
    R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
    D_pred = eval_model(DM, R, ao_lab; retraction = D -> eigen_retraction(D, Int(sum(at_no)/2))) # predicted density matrix with retraction
    # RE_D[i] += norm(D_pred - D)/norm(D)
    H_pred = eval_model(HM, R, ao_lab; retraction = identity) # predicted density matrix w/o retraction
    s, q = eigen(Symmetric(S))
    s_half = q * Diagonal(s.^(-1/2)) * q'
    H_pred = Symmetric(s_half * H_pred * s_half)
    C_pred = eigen(H_pred).vectors[:,1:Int(sum(at_no)/2)]
    D_pred_H = C_pred * C_pred'
    # RE_H_D[i] += norm(D_pred_H - D)/norm(D)
    push!(train_ref, vec(D)...)
    push!(train_pred, vec(D_pred)...)
    push!(train_pred_H, vec(D_pred_H)...)
end

for frame in frames_test
    R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
    D_pred = eval_model(DM, R, ao_lab; retraction = D -> eigen_retraction(D, Int(sum(at_no)/2))) # predicted density matrix with retraction
    # RE_D_test[i] += norm(D_pred - D)/norm(D)
    H_pred = eval_model(HM, R, ao_lab; retraction = identity) # predicted density matrix w/o retraction
    s, q = eigen(Symmetric(S))
    s_half = q * Diagonal(s.^(-1/2)) * q'
    H_pred = Symmetric(s_half * H_pred * s_half)
    C_pred = eigen(H_pred).vectors[:,1:Int(sum(at_no)/2)]
    D_pred_H = C_pred * C_pred'
    # RE_H_D_test[i] += norm(D_pred_H - D)/norm(D)
    push!(test_ref, vec(D)...)
    push!(test_pred, vec(D_pred)...)
    push!(test_pred_H, vec(D_pred_H)...)
end

train_ref = train_ref[findall(x -> x<0.9, train_ref)]
train_pred = train_pred[findall(x -> x<0.9, train_ref)]
train_pred_H = train_pred_H[findall(x -> x<0.9, train_ref)]
test_ref = test_ref[findall(x -> x<0.9, test_ref)]
test_pred = test_pred[findall(x -> x<0.9, test_ref)]
test_pred_H = test_pred_H[findall(x -> x<0.9, test_ref)]
pos = Int.(1:floor(length(train_ref)/10000):length(train_ref))
pos2 = rand(1:length(test_pred), 30000)
plot(abs.(test_ref[pos2]), abs.(test_ref[pos2]), label = "y = x", title = "Exact and predicted density matrix (element-wise)", xlabel = "Reference", ylabel = "Prediction")
plot([1e-8,1], [1e-8,1], label = "y = x", title = "Exact and predicted density matrix (element-wise)", xlabel = "Reference", ylabel = "Prediction")
scatter!(abs.(train_ref[pos]), abs.(train_pred[pos]), label = "Training", xaxis=:log, yaxis=:log, legend=:bottomright)
scatter!(abs.(test_ref[pos]), abs.(test_pred[pos]), label = "Test", xaxis=:log, yaxis=:log)
savefig("test/CHO_Models/Hamiltonian/Densitymatrix_scatter_propanol_$(degree)_$(order).png")