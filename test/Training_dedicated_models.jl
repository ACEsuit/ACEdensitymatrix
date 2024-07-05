using Statistics, PythonCall, JLD2 # , Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")

Ndata = 3000

# read data 
routine = "data/new_datasets"
system_name = "hexanol"
filenames = ["$routine/$(system_name)_KS.h5"]# , "$routine/acrolein.h5", "$routine/phenol.h5", "$routine/aniline.h5", "$routine/nmacetamide.h5"]
frames = []
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i = 0:1:Ndata-1 ]...) # constructing a training data set with Ndata frames for a single .h5 file
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
degree = 8
order = 3

# Try load a model (and when necessary, do a refit)
@show system_name, order, degree
println("Loading the DM model ...")
println()

DM = load("test/CHO_Models/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut)_new.jld2") |> read_dict
    
println("DM Model is loaded and is to be fitted...")
println()

# Fitting
fit!(DM, frames; solver = ACEfit.QR(), λ = 1e-7)
save("test/Dedicated_Models/$(system_name)/DensityMatrix/model_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut).jld2", write_dict(DM))
println("DM Model fitted and saved!")
println()

# For Hamiltonian
println("Fitting for HM Model ...")
println()

# Fitting
fit!(DM, frames; solver = ACEfit.QR(), λ = 1e-7, Mode = "H")
save("test/Dedicated_Models/$(system_name)/Hamiltonian/model_ord$(order)_maxdeg$(degree)_rcut$(rcut)_zcut$(zcut).jld2", write_dict(DM))
println("HM Model fitted and saved!")
println()

# Validate the fitted model ? 

# include("../src/utils/model_validation.jl")
# DM = load("test/CHO_Models/All_CHO/even_sample/model_maxdeg2_ord2_rcut10.0_zcut10.0.jld2") |> read_dict
# HM = load("test/CHO_Models/Hamiltonian/All_CHO/even_sample/model_maxdeg2_ord2_rcut10.0_zcut10.0.jld2") |> read_dict
# RMSE, RE, ME = validate_model(DM, frames)
# RMSE, RE, ME, RMSE_H, RE_H = validate_model(HM, frames, Mode = "H")