using Statistics, JLD2 
# using PythonCall # PythonCall is used only when we need to call SKLearn

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")

Ndata = 3000
rcut = 10.0
zcut = 10.0


# read data
routine = "data/new_datasets"
# filenames = ["$routine/propanol_KS.h5"]
# filenames = ["$routine/propanol_KS.h5" , "$routine/hexanol_KS.h5", "$routine/acrolein_KS.h5"]#, "$routine/acetaldehyde_KS.h5"]
filenames = ["$routine/toluene_KS.h5"]
frames = []
train_set = 0:20:9999
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i in train_set ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

# filenames =  ["$routine/acetaldehyde_KS.h5"]
frames_test = []
test_set = rand(setdiff(0:9999, train_set), Int(Ndata/20)) # 5000:50:9999# 
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# Load a model

order = 2
degree = 4
DM = load("test/CH_Models/Toluene/DensityMatrix/model_maxdeg$(degree)_ord$(order)_rcut10.0_zcut10.0.jld2")|> read_dict

cutoffset = 4.0:0.5:7.0
# RE_train = zeros(length(ordset), length(degreeset))
RMSE_D_train = zeros(length(cutoffset))
RMSE_DfromH_train = zeros(length(cutoffset))
RMSE_H_train = zeros(length(cutoffset))
E_D_train = zeros(length(cutoffset))
E_DfromH_train = zeros(length(cutoffset))
E_H_train = zeros(length(cutoffset))

RMSE_D_test = zeros(length(cutoffset))
RMSE_DfromH_test = zeros(length(cutoffset))
RMSE_H_test = zeros(length(cutoffset))
E_D_test = zeros(length(cutoffset))
E_DfromH_test = zeros(length(cutoffset))
E_H_test = zeros(length(cutoffset))


# import ACEfit: solve

# function ACEfit.solve(solver::ACEfit.SKLEARN_ARD, A, y)
#     ARD = pyimport("sklearn.linear_model")."ARDRegression"
#     clf = ARD(max_iter = solver.n_iter, threshold_lambda = solver.threshold_lambda,
#               tol = solver.tol,
#               fit_intercept = false, compute_score = true)
#     clf.fit(A, y)
#     # if length(clf.scores_) < solver.n_iter
#     #     @info "ARD converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
#     # else
#     #     @warn "\n\nARD did not converge to tol=$(solver.tol) after n_iter=$(solver.n_iter) iterations.\n\n"
#     # end
#     c = clf.coef_
#     return Dict{String, Any}("C" => pyconvert(Array,c) )
# end

zcut = 8.0
for (i, cutoff) in enumerate(cutoffset)

        println("Constructing/Loading the order $order degree $degree model with cutoff $cutoff ...")
        println()

        # construct or load the Density Matrix model
        # DM = reset_cutoff(DM, cutoff, cutoff, cutoff, zcut)
        # if !isfitted(DM)
        #         println("Start fitting ...")
        #         # fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
        #         fit!(DM, frames; solver = ACEfit.QR())
        #         # Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
        #         save("test/CH_Models/Toluene/Cutoff_tests/DensityMatrix/model_maxdeg$(degree)_ord$(order)_rcut$(cutoff)_zcut$(zcut).jld2", write_dict(DM))
        #         println("DM Model fitted and saved!")
        # end
        DM = load("test/CH_Models/Toluene/Cutoff_tests/DensityMatrix/model_maxdeg$(degree)_ord$(order)_rcut$(cutoff)_zcut$(zcut).jld2")|> read_dict

        # construct or load the Hamiltonian model
        HM = reset_cutoff(DM, cutoff, cutoff, cutoff, zcut)
        if isfitted(HM)
                println("Start fitting ...")
                # fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
                fit!(HM, frames; solver = ACEfit.QR(), Mode = "H")
                # Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
                save("test/CH_Models/Toluene/Cutoff_tests/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(cutoff)_zcut$(zcut).jld2", write_dict(HM))
                println("HM Model fitted and saved!")
        end
            
        # validate the model - Training
        for frame in frames
            R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
            D_pred = eval_model(DM, R, ao_labels, retraction = D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            H_pred = eval_model(HM, R, ao_labels) # predicted KS matrix
            RMSE_D_train[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_H_train[i] += norm(H_pred - H)^2/(size(H,1)*size(H,2))
            # RE_train[j,i] += norm(D_pred - D)/norm(D)
            E_D_train[i] += norm(D_pred - D)
            E_H_train[i] += norm(H_pred - H)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred_H = C_pred * C_pred'
            RMSE_DfromH_train[i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
            E_DfromH_train[i] += norm(D_pred_H - D)
        end
        RMSE_D_train[i] = sqrt(RMSE_D_train[i]/length(frames))
        RMSE_DfromH_train[i] = sqrt(RMSE_DfromH_train[i]/length(frames))
        RMSE_H_train[i] = sqrt(RMSE_H_train[i]/length(frames))
        # RE_train[j,i] /= length(frames)
        E_D_train[i] /= length(frames)
        E_DfromH_train[i] /= length(frames)
        E_H_train[i] /= length(frames)

        println("Training RMSE in D per matrix element = $(RMSE_D_train[i])")
        println("Training RMSE in D (from H) per matrix element = $(RMSE_DfromH_train[i])")
        println("Training RMSE in H per matrix element = $(RMSE_H_train[i])")
        # println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[j,i])")
        println("Average training error in D: ||D - D_ref|| = $(E_D_train[i])")
        println("Average training error in D (from H): ||D - D_ref|| = $(E_DfromH_train[i])")
        println("Average training error in H: ||H - H_ref|| = $(E_H_train[i])")
        println()

        # validate the model - Test
        for frame in frames_test
            R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
            D_pred = eval_model(DM, R, ao_labels, retraction = D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            H_pred = eval_model(HM, R, ao_labels) # predicted KS matrix
            RMSE_D_test[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_H_test[i] += norm(H_pred - H)^2/(size(H,1)*size(H,2))
            # RE_test[j,i] += norm(D_pred - D)/norm(D)
            E_D_test[i] += norm(D_pred - D)
            E_H_test[i] += norm(H_pred - H)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred_H = C_pred * C_pred'
            RMSE_DfromH_test[i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
            E_DfromH_test[i] += norm(D_pred_H - D)
        end
        RMSE_D_test[i] = sqrt(RMSE_D_test[i]/length(frames_test))
        RMSE_DfromH_test[i] = sqrt(RMSE_DfromH_test[i]/length(frames_test))
        RMSE_H_test[i] = sqrt(RMSE_H_test[i]/length(frames_test))
        # RE_test[j,i] /= length(frames_test)
        E_D_test[i] /= length(frames_test)
        E_DfromH_test[i] /= length(frames_test)
        E_H_test[i] /= length(frames_test)

        println("Test RMSE in D per matrix element = $(RMSE_D_test[i])")
        println("Test RMSE in D (from H) per matrix element = $(RMSE_DfromH_test[i])")
        println("Test RMSE in H per matrix element = $(RMSE_H_test[i])")
        # println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[j,i])")
        println("Average test error in D: ||D - D_ref|| = $(E_D_test[i])")
        println("Average test error in D (from H): ||D - D_ref|| = $(E_DfromH_test[i])")
        println("Average test error in H: ||H - H_ref|| = $(E_H_test[i])")
        println()

        println("Done for order $order degree $degree model with cutoff $cutoff ...")
        println()

end

using Plots
plot(cutoffset, log10.(RMSE_D_train), label = "Training RMSE in D", xlabel = "Cutoff", ylabel = "RMSE")
plot!(cutoffset, log10.(RMSE_D_test), label = "Test RMSE in D", linestyle = :dash)

plot!(cutoffset, log10.(RMSE_DfromH_train), label = "Training RMSE in D (from H)", xlabel = "Cutoff", ylabel = "RMSE")
plot!(cutoffset, log10.(RMSE_DfromH_test), label = "Test RMSE in D (from H)", linestyle = :dash)

plot!(cutoffset, log10.(RMSE_H_train), label = "Training RMSE in H", xlabel = "Cutoff", ylabel = "RMSE")
plot!(cutoffset, log10.(RMSE_H_test), label = "Test RMSE in H", linestyle = :dash)


# fix rcut and optimize zcut
zcutset = 4.0:0.5:10.0
RMSE_D_train = zeros(length(zcutset))
RMSE_DfromH_train = zeros(length(zcutset))
RMSE_H_train = zeros(length(zcutset))
E_D_train = zeros(length(zcutset))
E_DfromH_train = zeros(length(zcutset))
E_H_train = zeros(length(zcutset))
RMSE_D_test = zeros(length(zcutset))
RMSE_DfromH_test = zeros(length(zcutset))
RMSE_H_test = zeros(length(zcutset))
E_D_test = zeros(length(zcutset))
E_DfromH_test = zeros(length(zcutset))
E_H_test = zeros(length(zcutset))

cutoff = 5.5
for (i, zcut) in enumerate(zcutset)

        println("Constructing/Loading the order $order degree $degree model with zcut $zcut ...")
        println()

        # construct or load the Density Matrix model
        DM = reset_cutoff(DM, cutoff, cutoff, cutoff, zcut)
        if !isfitted(DM)
                println("Start fitting ...")
                # fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
                fit!(DM, frames; solver = ACEfit.QR())
                # Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
                save("test/CH_Models/Toluene/Zcut_tests/DensityMatrix/model_maxdeg$(degree)_ord$(order)_rcut$(cutoff)_zcut$(zcut).jld2", write_dict(DM))
                println("DM Model fitted and saved!")
        end

        # construct or load the Hamiltonian model
        HM = reset_cutoff(DM, cutoff, cutoff, cutoff, zcut)
        if !isfitted(HM)
                println("Start fitting ...")
                # fit!(DM, frames; solver = ACEfit.QR(lambda = 1e-12, P = I))
                fit!(HM, frames; solver = ACEfit.QR())
                # Folder = haskey(DM.Models, 7) ? "CHON_Models" : "CHO_Models"
                save("test/CH_Models/Toluene/Zcut_tests/Hamiltonian/model_maxdeg$(degree)_ord$(order)_rcut$(cutoff)_zcut$(zcut).jld2", write_dict(HM))
                println("HM Model fitted and saved!")
        end
            
        # validate the model - Training
        for frame in frames
            R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
            D_pred = eval_model(DM, R, ao_labels, retraction = D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            H_pred = eval_model(HM, R, ao_labels) # predicted KS matrix
            RMSE_D_train[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_H_train[i] += norm(H_pred - H)^2/(size(H,1)*size(H,2))
            # RE_train[j,i] += norm(D_pred - D)/norm(D)
            E_D_train[i] += norm(D_pred - D)
            E_H_train[i] += norm(H_pred - H)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred_H = C_pred * C_pred'
            RMSE_DfromH_train[i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
            E_DfromH_train[i] += norm(D_pred_H - D)
        end
        RMSE_D_train[i] = sqrt(RMSE_D_train[i]/length(frames))
        RMSE_DfromH_train[i] = sqrt(RMSE_DfromH_train[i]/length(frames))
        RMSE_H_train[i] = sqrt(RMSE_H_train[i]/length(frames))
        # RE_train[j,i] /= length(frames)
        E_D_train[i] /= length(frames)
        E_DfromH_train[i] /= length(frames)
        E_H_train[i] /= length(frames)

        println("Training RMSE in D per matrix element = $(RMSE_D_train[i])")
        println("Training RMSE in D (from H) per matrix element = $(RMSE_DfromH_train[i])")
        println("Training RMSE in H per matrix element = $(RMSE_H_train[i])")
        # println("Average training relative error in D: ||D - D_ref|| / ||D||= $(RE_train[j,i])")
        println("Average training error in D: ||D - D_ref|| = $(E_D_train[i])")
        println("Average training error in D (from H): ||D - D_ref|| = $(E_DfromH_train[i])")
        println("Average training error in H: ||H - H_ref|| = $(E_H_train[i])")
        println()

        # validate the model - Test
        for frame in frames_test
            R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
            D_pred = eval_model(DM, R, ao_labels, retraction = D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            H_pred = eval_model(HM, R, ao_labels) # predicted KS matrix
            RMSE_D_test[i] += norm(D_pred - D)^2/(size(D,1)*size(D,2))
            RMSE_H_test[i] += norm(H_pred - H)^2/(size(H,1)*size(H,2))
            # RE_test[j,i] += norm(D_pred - D)/norm(D)
            E_D_test[i] += norm(D_pred - D)
            E_H_test[i] += norm(H_pred - H)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred_H = C_pred * C_pred'
            RMSE_DfromH_test[i] += norm(D_pred_H - D)^2/(size(D,1)*size(D,2))
            E_DfromH_test[i] += norm(D_pred_H - D)
        end
        RMSE_D_test[i] = sqrt(RMSE_D_test[i]/length(frames_test))
        RMSE_DfromH_test[i] = sqrt(RMSE_DfromH_test[i]/length(frames_test))
        RMSE_H_test[i] = sqrt(RMSE_H_test[i]/length(frames_test))
        # RE_test[j,i] /= length(frames_test)
        E_D_test[i] /= length(frames_test)
        E_DfromH_test[i] /= length(frames_test)
        E_H_test[i] /= length(frames_test)

        println("Test RMSE in D per matrix element = $(RMSE_D_test[i])")
        println("Test RMSE in D (from H) per matrix element = $(RMSE_DfromH_test[i])")
        println("Test RMSE in H per matrix element = $(RMSE_H_test[i])")
        # println("Average test relative error in D: ||D - D_ref|| / ||D||= $(RE_test[j,i])")
        println("Average test error in D: ||D - D_ref|| = $(E_D_test[i])")
        println("Average test error in D (from H): ||D - D_ref|| = $(E_DfromH_test[i])")
        println("Average test error in H: ||H - H_ref|| = $(E_H_test[i])")
        println()

        println("Done for order $order degree $degree model with zcut $zcut ...")
        println()

end

plot(zcutset, log.(RMSE_D_train), label = "Training RMSE in D", xlabel = "Zcut", ylabel = "RMSE")
plot!(zcutset, log.(RMSE_D_test), label = "Test RMSE in D", linestyle = :dash)