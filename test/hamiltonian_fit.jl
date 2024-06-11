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
    push!(frames_test,[ read_frame(molecule,Int(i)) for i = Ndata+4000:10:9999 ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# Load / construct a model 
# parameters
rcut = 10.0
zcut = 10.0
for order = 3:3
    for degree = 2:5
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

# validate the model - learning curve
order = 2
rcut = 10.0
zcut = 10.0
degreeset = 4:10
RE_D = zeros(length(degreeset))
RE_H_D = zeros(length(degreeset))
RE_D_test = zeros(length(degreeset))
RE_H_D_test = zeros(length(degreeset))
RMSE_D = zeros(length(degreeset))
RMSE_H_D = zeros(length(degreeset))
RMSE_D_test = zeros(length(degreeset))
RMSE_H_D_test = zeros(length(degreeset))
ME_D = zeros(length(degreeset))
ME_H_D = zeros(length(degreeset))
ME_D_test = zeros(length(degreeset))
ME_H_D_test = zeros(length(degreeset))
for (i,degree) in enumerate(degreeset)
    DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2")|> read_dict
    HM = load("test/CHO_Models/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2")|> read_dict
    for frame in frames
        R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
        D_pred = eval_model(DM, R, ao_lab; retraction = D -> eigen_retraction(D, Int(sum(at_no)/2))) # predicted density matrix with retraction
        RE_D[i] += norm(D_pred - D)/norm(D)
        RMSE_D[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
        ME_D[i] = maximum(maximum(abs.(D_pred - D))) > ME_D[i] ? maximum(maximum(abs.(D_pred - D))) : ME_D[i]
        H_pred = eval_model(HM, R, ao_lab; retraction = identity) # predicted density matrix w/o retraction
        s, q = eigen(Symmetric(S))
        s_half = q * Diagonal(s.^(-1/2)) * q'
        H_pred = Symmetric(s_half * H_pred * s_half)
        C_pred = eigen(H_pred).vectors[:,1:Int(sum(at_no)/2)]
        D_pred_H = C_pred * C_pred'
        RE_H_D[i] += norm(D_pred_H - D)/norm(D)
        RMSE_H_D[i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
        ME_H_D[i] = maximum(maximum(abs.(D_pred_H - D))) > ME_H_D[i] ? maximum(maximum(abs.(D_pred_H - D))) : ME_H_D[i]
    end
    # println("Training RMSE per matrix element = $(sqrt(RMSE/length(frames)))")
    RE_D[i] /= length(frames)
    RE_H_D[i] /= length(frames)
    RMSE_D[i] = sqrt(RMSE_D[i]/length(frames))
    RMSE_H_D[i] = sqrt(RMSE_H_D[i]/length(frames))
    println("Average training relative error (directly trained on D) for degree $degree: ||D_pred - D_ref|| / ||D||= $(RE_D[i])")
    println("Average training relative error (trained on H and then D) for degree $degree: ||D_pred_H - D_ref|| / ||D||= $(RE_H_D[i])")
    println("Average training RMSE per matrix element for degree $degree: $(RMSE_D[i])")
    println("Average training RMSE per matrix element (H => D) for degree $degree: $(RMSE_H_D[i])")
    println("Maximum training error per matrix element for degree $degree: $(ME_D[i])")
    println("Maximum training error per matrix element (H => D) for degree $degree: $(ME_H_D[i])")
    println()

    # validate the model - Test
    for frame in frames_test
        R, D, ao_lab, at_no, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
        D_pred = eval_model(DM, R, ao_lab; retraction = D -> eigen_retraction(D, Int(sum(at_no)/2))) # predicted density matrix with retraction
        RE_D_test[i] += norm(D_pred - D)/norm(D)
        RMSE_D_test[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
        ME_D_test[i] = maximum(maximum(abs.(D_pred - D))) > ME_D_test[i] ? maximum(maximum(abs.(D_pred - D))) : ME_D_test[i]
        H_pred = eval_model(HM, R, ao_lab; retraction = identity) # predicted density matrix w/o retraction
        s, q = eigen(Symmetric(S))
        s_half = q * Diagonal(s.^(-1/2)) * q'
        H_pred = Symmetric(s_half * H_pred * s_half)
        C_pred = eigen(H_pred).vectors[:,1:Int(sum(at_no)/2)]
        D_pred_H = C_pred * C_pred'
        RE_H_D_test[i] += norm(D_pred_H - D)/norm(D)
        RMSE_H_D_test[i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
        ME_H_D_test[i] = maximum(maximum(abs.(D_pred_H - D))) > ME_H_D_test[i] ? maximum(maximum(abs.(D_pred_H - D))) : ME_H_D_test[i]
    end
    # println("Test RMSE per matrix element = $(sqrt(RMSE/length(frames_test)))")
    RE_D_test[i] /= length(frames_test)
    RE_H_D_test[i] /= length(frames_test)
    RMSE_D_test[i] = sqrt(RMSE_D_test[i]/length(frames_test))
    RMSE_H_D_test[i] = sqrt(RMSE_H_D_test[i]/length(frames_test))
    println("Average test relative error (directly trained on D) for degree $degree: ||D_pred - D_ref|| / ||D||= $(RE_D_test[i])")
    println("Average test relative error (trained on H and then D) for degree $degree: ||D_pred_H - D_ref|| / ||D||= $(RE_H_D_test[i])")
    println("Average test RMSE per matrix element for degree $degree: $(RMSE_D_test[i])")
    println("Average test RMSE per matrix element (H => D) for degree $degree: $(RMSE_H_D_test[i])")
    println("Maximum test error per matrix element for degree $degree: $(ME_D_test[i])")
    println("Maximum test error per matrix element (H => D) for degree $degree: $(ME_H_D_test[i])")
    println()
end

using Plots

Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"


# visualize the error - line plot
plt = plot(degreeset, log10.(RE_D), label = "Order $order: Training Relative Error", xlabel = "Degree", ylabel = "RE (10^y)", legendfontsize=7, color = 1)
plot!(degreeset, log10.(RE_D_test), label = "Order $order: Test Relative Error", linestyle = :dash, color = 1)
plot!(degreeset, log10.(RE_H_D), label = "Order $order: Training Relative Error (H => D)", color = 2, marker = :diamond)
plot!(degreeset, log10.(RE_H_D_test), label = "Order $order: Test Relative Error (H => D)", linestyle = :dash, color = 2, marker = :diamond)
title!("RE vs Degree")
savefig("test/$Folder/Hamiltonian/RE_Order$(order)_rcut$(rcut)_zcut$(zcut).png")

plt = plot(degreeset, log10.(RMSE_D), label = "Order $order: Training RMSE", xlabel = "Degree", ylabel = "RMSE (10^y)", legendfontsize=7, color = 1)
plot!(degreeset, log10.(RMSE_D_test), label = "Order $order: Test RMSE", linestyle = :dash, color = 1)
plot!(degreeset, log10.(RMSE_H_D), label = "Order $order: Training RMSE (H => D)", color = 2, marker = :diamond)
plot!(degreeset, log10.(RMSE_H_D_test), label = "Order $order: Test RMSE (H => D)", linestyle = :dash, color = 2, marker = :diamond)
title!("RMSE vs Degree")
savefig("test/$Folder/Hamiltonian/RMSE_Order$(order)_rcut$(rcut)_zcut$(zcut).png")

plt = plot(degreeset, log10.(ME_D), label = "Order $order: Training Maximum Error", xlabel = "Degree", ylabel = "ME (10^y)", legendfontsize=7, color = 1)
plot!(degreeset, log10.(ME_D_test), label = "Order $order: Test Maximum Error", linestyle = :dash, color = 1)
plot!(degreeset, log10.(ME_H_D), label = "Order $order: Training Maximum Error (H => D)", color = 2, marker = :diamond)
plot!(degreeset, log10.(ME_H_D_test), label = "Order $order: Test Maximum Error (H => D)", linestyle = :dash, color = 2, marker = :diamond)
title!("ME vs Degree")
savefig("test/$Folder/Hamiltonian/ME_Order$(order)_rcut$(rcut)_zcut$(zcut).png")


# visualize the error - scatter plot
degree = 5
order = 3
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

pos = Int.(1:floor(length(train_ref)/10000):length(train_ref))
pos2 = rand(1:length(test_pred), 30000)
plot(abs.(test_ref[pos2]), abs.(test_ref[pos2]), label = "y = x", title = "Exact and predicted density matrix (element-wise)", xlabel = "Reference", ylabel = "Prediction")
plot([1e-8,1], [1e-8,1], label = "y = x", title = "Exact and predicted density matrix (element-wise)", xlabel = "Reference", ylabel = "Prediction")
scatter!(abs.(train_ref[pos]), abs.(train_pred[pos]), label = "Training", xaxis=:log, yaxis=:log, legend=:bottomright)
scatter!(abs.(test_ref[pos]), abs.(test_pred[pos]), label = "Test", xaxis=:log, yaxis=:log)
scatter!(abs.(train_ref[pos]), abs.(train_pred_H[pos]), label = "Training (H => D)", xaxis=:log, yaxis=:log, legend=:bottomright)
scatter!(abs.(test_ref[pos]), abs.(test_pred_H[pos]), label = "Test (H => D)", xaxis=:log, yaxis=:log)
savefig("test/CHO_Models/Hamiltonian/Densitymatrix_scatter_propanol_$(degree)_$(order).png")