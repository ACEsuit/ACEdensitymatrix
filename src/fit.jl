module Fitting

using ACEdensitymatrix
using Setfield, LinearAlgebra, ACEfit, SparseArrays, DecoratedParticles, Lux, Random
export fit!


# ```
# fit function for onsite model:
#     model: a On_Model object, fitted or not
#     Rs: a vector of State objects <==> {R_II}_I
#     Ys: a vector of matrices <==> D_II
# ```
# TODO : think about what is the best way to incorporate the regularization term (in solver? in line? in variables?)
Dict_Int2Spec = Dict(1 => "H", 6 => "C", 7 => "N", 8 => "O")
Dict_Int2Orbs = Dict(0 => "S", 1 => "P", 2 => "D", 3 => "F")
Dict_Spec2Int = Dict("H" => 1, "C" => 6, "N" => 7, "O" => 8)
Dict_Orbs2Int = Dict("S" => 0, "P" => 1, "D" => 2, "F" => 3)

function fit!(model::AbstractModel, Rs::Union{Vector{PState{T}},Vector{Vector{PState{T}}}}, Ys::Vector{Matrix{TY}}; solver = ACEfit.SKLEARN_BRR(), λ = 1e-12, reg = :id) where {T, TY}
    TP = typeof(model)
    LLset = [(l1,l2) for l1 in 0:get_L(model)[1] for l2 in 0:get_L(model)[2]]
    n_orbs1, n_orbs2 = get_norbs(model)
    @assert(length(LLset) == length(model.ps.dot))

    layer_set = ["layer_$i" for i in 1:length(LLset)]
    layer_set = Symbol.(layer_set)

    # evaluate the EQM for all R in Rs, just once, and construct all the design matrices A -> A_all
    partial_md = Chain([model.model.layers[i] for i = 1:5]...)
    ps, st = Lux.setup(MersenneTwister(1234), partial_md)
    A_all = [ zeros((2*LLset[i][1]+1)*(2*LLset[i][2]+1)*length(Rs), size(model.ps.dot[i].W,2)) for i = 1:length(LLset) ]

    for (j,R) in enumerate(Rs)
        valset = partial_md(R,ps,st)[1]
        for (i, (l1,l2)) in enumerate(LLset)
            A_all[i][(2l1+1)*(2l2+1)*(j-1)+1:(2l1+1)*(2l2+1)*j,:] = flat(valset[i])
        end
    end

    if reg == :id
        Γ = [ I for i = 1:length(LLset) ]
    elseif reg == :smooth
        Γ = regularizer(model)
    end

    for (i, (l1,l2)) in enumerate(LLset)
        println("Fitting the $(Dict_Int2Orbs[l1])$(Dict_Int2Orbs[l2]) blocks ...")
        println()
        
        RMSE = 0

        # get the design matrix A for the (l1,l2) block, and delete the corresponding part in A_all
        A = popat!(A_all, 1)
        num = size(A)[2] # number of basis

        A = [A; λ*Γ[i]]
        
        for kk = 1 : size(model.ps.dot[i].W,1)
            ii, jj = k2ij(kk, n_orbs1[l1+1], n_orbs2[l2+1])
            # println("Fitting the ($ii,$jj)-th $(Dict_Int2Orbs[l1])$(Dict_Int2Orbs[l2]) block ...")
            
            Yij = [ get_Y(Ys[t], n_orbs1, n_orbs2, l1, l2, ii, jj) for t = 1:length(Ys) ]

            # construct Y 
            Y = zeros(Float64, length(Ys)*length(Yij[1]))
            for k in 1:length(Ys)
                Y[(k-1)*length(Yij[1])+1:k*length(Yij[1])] = Yij[k]
            end
            Y = [Y; zeros(num)]

            # solve for C[kk]
            C = ACEfit.solve(solver, A, Y)["C"];
            @set! model.ps.dot.$(layer_set[i]).W[kk,:] = C
            # list of potential solvers: ACEfit: QR, LSQR, RRQR, SKLEARN_BRR, SKLEARN_ARD, BLR, TruncatedSVD...
            
            RMSE += norm((A*C-Y)[1:end-num])^2/(length(Y)-num)
        end
        # println("RMSE = $(sqrt(RMSE/length(LLset)))")
        println("RMSE = $(sqrt(RMSE/size(model.ps.dot[i].W,1)))")
        println()

        GC.gc()
    end
    
    model = (TP <: On_Model) ? TP(model.model, model.ps, model.st, model.n_orbs, true) : TP(model.model, model.ps, model.st, model.n_orbs1, model.n_orbs2, true)
    # @show model.ps.dot[1].W
    # @show model.fitted
    # return model
end

function fit!(model::AbstractModel, Rs::Union{Vector{PState{T}},Vector{Vector{PState{T}}}}, Ys::Vector{Vector{TY}}; solver = ACEfit.SKLEARN_BRR(), λ = 1e-12, reg = :id, GC_switcher = false) where {T, TY}
    TP = typeof(model)
    LLset = [(l1,l2) for l1 in 0:get_L(model)[1] for l2 in 0:get_L(model)[2]]
    n_orbs1, n_orbs2 = get_norbs(model)
    @assert(length(LLset) == length(model.ps.dot))

    layer_set = ["layer_$i" for i in 1:length(LLset)]
    layer_set = Symbol.(layer_set)

    # evaluate the EQM for all R in Rs, just once, and construct all the design matrices A -> A_all
    partial_md = Chain([model.model.layers[i] for i = 1:5]...)
    ps, st = Lux.setup(MersenneTwister(1234), partial_md)
    
    println("Start constructing A")
    println()
    A_all = [ zeros((2*LLset[i][1]+1)*(2*LLset[i][2]+1)*length(Rs) + size(model.ps.dot[i].W,2), size(model.ps.dot[i].W,2)) for i = 1:length(LLset) ]
    
    if reg == :id
        Γ = [ I for i = 1:length(LLset) ]
    elseif reg == :smooth
        Γ = regularizer(model)
    end
    
    for i in 1:length(LLset)
        try 
            A_all[i][end-size(model.ps.dot[i].W,2)+1:end,:] = λ*Γ[i]
        catch 
            A_all[i][end-size(model.ps.dot[i].W,2)+1:end,:] = λ*Γ[i](size(model.ps.dot[i].W,2))
        end
    end

    # TODO: multi-threading ?
    for (j,R) in enumerate(Rs)
        valset = partial_md(R,ps,st)[1]
        for (i, (l1,l2)) in enumerate(LLset)
            A_all[i][(2l1+1)*(2l2+1)*(j-1)+1:(2l1+1)*(2l2+1)*j,:] = flat(valset[i])
        end
    end
    println("Finish constructing A")
    println()

    for (i, (l1,l2)) in enumerate(LLset)
        println("Fitting the $(Dict_Int2Orbs[l1])$(Dict_Int2Orbs[l2]) blocks ...")
        println()
        
        RMSE = 0

        # get the design matrix A for the (l1,l2) block, and delete the corresponding part in A_all
        # This following line avoid the fitting of subblocks being parallelizable, but it is totally fine because of potential memory issues of multi-threading
        A = popat!(A_all, 1)
        num = size(A)[2] # number of basis
        if length(A) > 1e6 && !GC_switcher
            GC_switcher = true
        end

        if GC_switcher; GC.gc(); end
        # A = [A; λ*Γ]
        
        for kk = 1 : size(model.ps.dot[i].W,1)
            # solve for C[kk]
            Y = [ popat!(Ys, 1); zeros(num) ]
            if GC_switcher; GC.gc(); end
            C = ACEfit.solve(solver, A, Y)["C"];
            @set! model.ps.dot.$(layer_set[i]).W[kk,:] = C
            # list of potential solvers: ACEfit: QR, LSQR, RRQR, SKLEARN_BRR, SKLEARN_ARD, BLR, TruncatedSVD...
            
            RMSE += norm((A*C-Y)[1:end-num])^2/(length(Y)-num)
            Y = nothing
            if GC_switcher; GC.gc(); end
        end
        # println("RMSE = $(sqrt(RMSE/length(LLset)))")
        println("RMSE = $(sqrt(RMSE/size(model.ps.dot[i].W,1)))")
        println()

        A = nothing
        if GC_switcher; GC.gc(); end
    end
    
    model = (TP <: On_Model) ? TP(model.model, model.ps, model.st, model.n_orbs, true) : TP(model.model, model.ps, model.st, model.n_orbs1, model.n_orbs2, true)
    # @show model.ps.dot[1].W
    # @show model.fitted
    # return model
end

function split_data(frames::Vector{Dict{String, Array}}, keys::Base.KeySet{Union{T,Tuple{T,T}}}; Mode = "D", rcut_on = 10.0, r_cut_off = 10.0, zcut = 10.0) where T
    Rs = Dict(key => [] for key in keys)
    Ys = Dict(key => [] for key in keys)
    
    for frame in frames
        f = convert_frame(frame)
        for key in keys
            if !(typeof(key) <: Tuple)
                for i in findall(x->x==key, f["atomic_numbers"])
                    push!(Rs[key], get_state(f["R"], i, i; atom_filter = filter_on(rcut_on)))
                    push!(Ys[key], get_block(f[Mode], i, i, f["ao_labels"]))
                end
            else
                i, j = key
                for ii in findall(x->x==i, f["atomic_numbers"])
                    # TODO: In fact, can we use only half of the jjs, larger than ii, and make use of symmetry
                    for jj in setdiff(findall(x->x==j, f["atomic_numbers"]),ii)
                        push!(Rs[key], get_state(f["R"], ii, jj; atom_filter = filter_off(r_cut_off, zcut)))
                        push!(Ys[key], get_block(f[Mode], ii, jj, f["ao_labels"]))
                    end
                end
            end
        end
    end

    return Rs, Ys
end

# Fit a whole Density_Model
# Here, frames can be non_franslated frame (directly read from data) which will be transfer to a readable format (i.e. convert_frame) in split_data function
# The function should return a fitted Density_Model

# TODO: check the types of Rs and Ys from the above function split_data
function fit!(model::Density_Model, Rs, Ys; solver = ACEfit.SKLEARN_BRR(), λ = 1e-12, reg = :id, Mode = "D", multi_thread = false, GC_switcher = false)
    
    if multi_thread
        Base.Threads.@threads for key in keys(model.Models)
            typeof(key) <: Tuple ? println("=== Fitting for $(Dict_Int2Spec[key[1]])-$(Dict_Int2Spec[key[2]]) offsite model ===") : println("==== Fitting for $(Dict_Int2Spec[key]) onsite model ====")
            println()
            if length(Rs[key]) == 0 || length(Ys[key]) == 0
                continue
            end
            model.Models[key] = fit!(model.Models[key], identity.(pop!(Rs,key)), assemble_Y( identity.(pop!(Ys,key)), get_norbs(model.Models[key])...); solver = solver, λ = λ, reg = reg, GC_switcher = GC_switcher)
        end
    else
        for key in keys(model.Models)
            typeof(key) <: Tuple ? println("=== Fitting for $(Dict_Int2Spec[key[1]])-$(Dict_Int2Spec[key[2]]) offsite model ===") : println("==== Fitting for $(Dict_Int2Spec[key]) onsite model ====")
            println()
            if length(Rs[key]) == 0 || length(Ys[key]) == 0
                continue
            end
            model.Models[key] = fit!(model.Models[key], identity.(pop!(Rs,key)), assemble_Y( identity.(pop!(Ys,key)), get_norbs(model.Models[key])...); solver = solver, λ = λ, reg = reg, GC_switcher = GC_switcher)
        end
    end
    if !isfitted(model)
        @warn("Some models are not fitted because there is a lack of corresponding data...")
    end
    # return model
end

"""
    fit!(model::Density_Model, frames::Vector{Dict{String, Array}; solver, λ, reg, Mode)

In line function fitting a Density_Model `model` (fitted or not) with the (new) training data `frame`

# Arguments
- `model`: Density_Model, A density matrix model
- `frame`: A set of training data that requires some specific form
- `solver`: Solver for solving a LS system that has to provide interface for the design matrix and RHS
- ` λ`: Float64, Regularization parameter, coefficient of the regularizer
- `reg`: Symbol, indicating type of regularizer; built in type is :reg and :smooth, identity and smooth priorer
- `Mode`: String, Mode in the frame that needs to be fitted; default is "D", 
          other possibility is "H" for Hamiltonian, "S" for the overlap, and other usage depending on the input frame

"""
function fit!(model::Density_Model,frames::Union{Dict{String, Array}, Vector{Dict{String, Array}}}; solver = ACEfit.SKLEARN_BRR(), λ = 1e-12, reg = :id, Mode = "D", multi_thread = false, GC_switcher = false)
    rcut_on, r_cut_off, zcut = get_cutoff(model)
    Rs, Ys = split_data(frames, keys(model.Models); Mode = Mode, rcut_on = rcut_on, r_cut_off = r_cut_off, zcut = zcut)
    fit!(model, Rs, Ys; solver = solver, λ = λ, reg = reg, Mode = Mode, multi_thread = multi_thread, GC_switcher = GC_switcher)
end

# function fit_with_tuned_sample(DM, filenames, Ndata = 10000, η = 1.2; train_set = nothing, Mode = "D", refit = true)
#     DH = Mode == "D" ? "DensityMatrix" : "Hamiltonian"
#     init = 0 # from which part of frames will we start testing the model
#     if train_set == nothing
#         train_set = [ Vector{Int64}(0:10:Ndata/10-1) for _ = 1:length(filenames) ]
#         init = 1 # if the train set is not given, it's initialized as an even set from the first 1/10 frames so the first 1000 frames are skipped from testing
#     end
#     test_set = Vector{Int64}(9/10*Ndata:10:Ndata-1)

#     # frames_train = []
#     # for (k,fname) in enumerate(filenames)
#     #     molecule = TrajectoryHDF5(fname)
#     #     push!(frames_train,[ read_frame(molecule,Int(i)) for i in train_set[k] ]...) # constructing a training data set with Ndata frames for a single .h5 file
#     # end
#     # frames_train = identity.(frames_train)

#     # frames_test = []
#     # for fname in filenames
#     #     molecule = TrajectoryHDF5(fname)
#     #     push!(frames_test,[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a test data set with Ndata frames for a single .h5 file
#     # end
#     # frames_test = identity.(frames_test)

#     frames_train = [ [] for _ = 1:length(filenames) ]
#     for (k,fname) in enumerate(filenames)
#         molecule = TrajectoryHDF5(fname)
#         push!(frames_train[k],[ read_frame(molecule,Int(i)) for i in train_set[k] ]...) # constructing a training data set with Ndata frames for a single .h5 file
#     end
#     frames_train = [ identity.(frames_train[i]) for i = 1:length(frames_train) ]
#     frames_train_all = union(frames_train...) |> unique

#     frames_test = [ [] for _ = 1:length(filenames) ]
#     for (k,fname) in enumerate(filenames)
#         molecule = TrajectoryHDF5(fname)
#         push!(frames_test[k],[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a training data set with Ndata frames for a single .h5 file
#     end
#     frames_test = [ identity.(frames_test[i]) for i = 1:length(frames_test) ]
#     frames_test_all = union(frames_test...) |> unique

#     if refit
#         fit!(DM, frames_train_all; solver = ACEfit.QR(), λ = 1e-4, Mode = Mode)
#     end

#     for i = init:8
#         # training rmse
#         rmse_train = [ validate_model(DM, frames_train[k]; Mode = Mode)[1] for k = 1:length(filenames) ]
#         # test rmse
#         rmse_test = [ validate_model(DM, frames_test[k]; Mode = Mode)[1] for k = 1:length(filenames) ]
#         if all(rmse_test .< η * rmse_train)
#             println("training set founded with $(sum(length(train_set[t]) for t in 1:length(train_set))) frames")
#             println("training RMSE: $(mean(rmse_train))")
#             println("test RMSE: $(mean(rmse_test))")
#             println()
#             break
#         else
#             println("test RMSE ($(mean(rmse_test))) is greater than $η times the training RMSE ($(mean(rmse_train)))")
#             println()
#         end
    
#         # Add more data
#         println("Entering $(i * Ndata/10) - $((i+1) * Ndata/10 - 1) frames")
#         println()
#         test_set_amended = Vector{Int64}(i * Ndata/10:(i+1) * Ndata/10 - 1)
#         for (k,fname) in enumerate(filenames)
#             frames_tmp = []
#             molecule = TrajectoryHDF5(fname)
#             push!(frames_tmp,[ read_frame(molecule,Int(i)) for i in test_set_amended ]...) # constructing a test data set with Ndata frames for a single .h5 file
#             for (kk,frame) in enumerate(frames_tmp)
#                 if !(test_set_amended[kk] in train_set[k])
#                     rmse_tmp = validate_model(DM, [frame])[1]
#                     if rmse_tmp > η * rmse_train[k]
#                         push!(frames_train[k], frame)
#                         push!(train_set[k], test_set_amended[kk])
#                     end
#                 end
#             end
#         end
#         frame_train = [ identity.(frames_train[i]) for i = 1:length(frames_train) ]
#         frames_train_all = union(frames_train...) |> unique

#         # Fit the model with the new training set
#         fit!(DM, frames_train_all; solver = ACEfit.QR(), λ = 1e-4, Mode = Mode)

#         save("test/$(Folder)/$(system)/tuned_sampling/$(DH)/QR1e-4/log_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut)_tol$(η)_Ndata$(Ndata)_strict.jld2", Dict("__id__" => "training_set", "train_set" => train_set, "rmse_train" => rmse_train, "rmse_test" => rmse_test, "Ndata" => length(train_set)))
#         save("test/$(Folder)/$(system)/tuned_sampling/$(DH)/QR1e-4/model_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut)_tol$(η)_Ndata$(Ndata)_strict.jld2", write_dict(DM))
#     end

#     rmse_train = validate_model(DM, frames_train_all)[1]
#     rmse_test = validate_model(DM, frames_test_all)[1]

#     println("training RMSE: $rmse_train")
#     println("test RMSE: $rmse_test")
#     if rmse_test < η * rmse_train
#         println("test RMSE ($rmse_test) is less than $η times the training RMSE ($rmse_train)")
#     else
#         @warn("test RMSE ($rmse_test) is greater than $η times the training RMSE ($rmse_train)")
#     end

#     save("test/$(Folder)/$(system)/tuned_sampling/$(DH)/QR1e-4/log_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut)_tol$(η)_Ndata$(Ndata)_strict.jld2", Dict("__id__" => "training_set", "train_set" => train_set, "rmse_train" => rmse_train, "rmse_test" => rmse_test, "Ndata" => length(train_set)))
#     save("test/$(Folder)/$(system)/tuned_sampling/$(DH)/QR1e-4/model_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut)_tol$(η)_Ndata$(Ndata)_strict.jld2", write_dict(DM))

#     return DM, train_set
# end

end # module