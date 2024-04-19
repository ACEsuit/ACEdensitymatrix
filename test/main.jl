using Statistics, Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")

# read data 
molecule = TrajectoryHDF5("data/propanol.h5")
frames = [ read_frame(molecule,i) for i =1:200 ] # constructing a training data set with 200 frames on a single .h5 file

# construct a model 
ao_dict = Dict( 6 => Dict("n_orbs" => [3,2,1], "maxdeg" => 6, "ord" => 2, "rcut" => 6.0, "zcut" => 6.0), 
                1 => Dict("n_orbs" => [2], "maxdeg" => 6, "ord" => 2, "rcut" => 6.0, "zcut" => 6.0), 
                8 => Dict("n_orbs" => [3,2,1], "maxdeg" => 6, "ord" => 2, "rcut" => 6.0, "zcut" => 6.0) )

DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

# fit the model
fit!(DM, frames; solver = ACEfit.QR())

# evaluate the model
R, D = translate_frame(frames[1])["R"], translate_frame(frames[1])["D"] # an example configuration and its corresponding density matrix
D_pred = eval_model(DM, R, frame["Basis set labels"]) # predicted density matrix

# error
E = D_pred - D |> maximum

# visualize the error
E = abs.(D_pred - D)
contourf(E)