using ACEdensitymatrix, ACEfit, JLD2, LinearAlgebra

"""
This is a script that manually perform the commutator based active learning, just for illustration
"""

# Step 0: Data read
routine = "data/new_datasets"
system = "propanol"
filenames = ["$routine/$(system)_new.h5"]

# test set, always consists of the last 5000 frames
test_set = 5000:10:9999
frames_test = []
for (i,fname) in enumerate(filenames)
    molecule = TrajectoryHDF5(fname)
    # test_set = i <= 10 ? (75:99) : (7500:100:9999)
    push!(frames_test,[ read_frame(molecule,Int(i)) for i in test_set ]...) # constructing a test data set with Ndata frames for a single .h5 file
end
frames_test = identity.(frames_test)

# training set construction
train_set = Vector(0:499) # Initial training set
N_amend = 500 # frames added at each iteration

# add to the training set the worst 500 frames at each round
ITERATION = 2 # current iteration
for iter = 1:ITERATION
    comm_errs = readlines("data/commutators_$(iter)_5000.txt")
    comm_errs = [parse(Float64, strip(comm_err)) for comm_err in comm_errs if !isempty(strip(comm_err))]
    train_set_amendment = partialsortperm(comm_errs, 1:N_amend, rev=true) |> sort
    push!(train_set, train_set_amendment...)
    unique!(sort!(train_set))
end

frames = []
# train_set = 0:2:Ndata-1 # 0:10:9999
for (i,fname) in enumerate(filenames)
    molecule = TrajectoryHDF5(fname)
    # train_set = i <= 10 ? (0:74) : (0:100:7499)
    # train_set = 0:1:Ndata-1
    push!(frames,[ read_frame(molecule,Int(i)) for i in train_set ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

# step 1: read the base model
order = 3
degree = 8
rcut = 6.5
zcut = 10.0

DM = load("test/CHO_Models/$(system)/DensityMatrix/model_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut)_base_model_500_5000_iter_$(ITERATION-1).jld2")|> read_dict # the model after (ITERATION-1) iteration
println("DM Model loaded!")

# errors before iteration
RMSE_D_train, RMSE_D_train_MIN, RMSE_D_train_MAX, RE_D_train, ME_D_train = validate_model(DM, frames)
RMSE_D_test, RMSE_D_test_MIN, RMSE_D_test_MAX, RE_D_test, ME_D_test = validate_model(DM, frames_test)

println("original RMSE in D per matrix element in the amended training set = $(RMSE_D_train)")
println("original RMSE in D per matrix element in the test set = $(RMSE_D_test)")
println()

# step 2: refit
fit!(DM, frames; solver = ACEfit.QR(), λ = 1e-4)
save("test/CHO_Models/$(system)/DensityMatrix/model_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut)_base_model_500_5000_iter_$(ITERATION).jld2", write_dict(DM)) # save the new model
println("DM Model fitted and saved!")

# step 3: error with the updated model
RMSE_D_train, RMSE_D_train_MIN, RMSE_D_train_MAX, RE_D_train, ME_D_train = validate_model(DM, frames)
RMSE_D_test, RMSE_D_test_MIN, RMSE_D_test_MAX, RE_D_test, ME_D_test = validate_model(DM, frames_test)

println("new RMSE in D per matrix element in the amended training set = $(RMSE_D_train)")
println("new RMSE in D per matrix element in the test set = $(RMSE_D_test)")
println()
