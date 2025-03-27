using ACEdensitymatrix, ACEfit, JLD2

# Step 1: Specify a model, including hyper parameters that determine the size of the model and the orbital information
degree = 4;    # degree of the polynomial - resolution of the one particle basis
order = 2;     # correlation order = body order + const
rcut = 4.0;    # onsite and off site cutoff radius, which are chosen identically here but can be different
zcut = 10.0;   # bond cutoff radius, which is only used in the offsite environment
ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut), 
                6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut),
                8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut" => rcut, "zcut" => zcut) );
                # This is a CHO model - H: 2S, C: 3S2P1D, O: 3S2P1D

# Step 2: Construct a model of certain sizes and with randomize initialized parameters
Model = Density_Model(ao_dict::Dict);

# Step 3: Read the training and test data (frames of different molecules)

routine = "data/new_datasets"
training_molecule = ["propanol", "hexanol"]
test_molecule = ["ethanol"]
filenames = ["$routine/$(training_molecule[i])_new.h5" for i in 1:length(training_molecule)]
testfilenames = ["$routine/$(test_molecule[i])_new.h5" for i in 1:length(test_molecule)]

# Construct the training data set
frames = []
for (i,fname) in enumerate(filenames)
    train_set = 0:2999
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(j)) for j in train_set]...) # constructing a training data set with Ndata frames for a single .h5 file
end

frames = identity.(frames);
frames[1] # Detailed look at the structure of the data

# Test data
frames_test = []
for (i,fname) in enumerate(testfilenames)
    test_set = 0:99
    molecule = TrajectoryHDF5(fname)
    @show molecule
    push!(frames_test,[ read_frame(molecule,Int(j)) for j in test_set]...) # constructing a training data set with Ndata frames for a single .h5 file
end

frames_test = identity.(frames_test);

# Step 4: Fit the model
fit!(Model, frames; solver = ACEfit.QR(), λ = 1e-4, reg = :smooth)

# Step 5: Prediction
# Read a frame from ethanol - the one doesn't appear in the training set
fm = read_frame(TrajectoryHDF5("$routine/$(test_molecule[1])_new.h5"),1)

# ``translate" the frame
R, D, atomic_number, ao_labels, H, S, C = convert_frame(fm)["R"], convert_frame(fm)["D"], convert_frame(fm)["atomic_numbers"], convert_frame(fm)["ao_labels"], convert_frame(fm)["H"], convert_frame(fm)["S"], convert_frame(fm)["C"];
D_pred = eval_model(Model, R, ao_labels, retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2)))
D_pred - D # check the error

# An alternative, though not recommended...
D_pred_2 = eval_model(Model, fm)
D_pred_2 == D_pred # check that they are equivalent

# Step 6: Model validation
RMSE, RE, ME = validate_model(Model, frames_test)[[1,4,5]]

# Step 7: Save the model - for this we need JLD2
save("test_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2", write_dict(Model))

# Step 8: Load the model
Model = load("test_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2") |> read_dict

# Step 9: Model validation
RMSE, RE, ME = validate_model(Model, frames_test)[[1,4,5]]