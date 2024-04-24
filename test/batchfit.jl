using Statistics, Plots, PythonCall
using ArgParse

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")


args = ArgParseSettings()
@add_arg_table args begin
    "--ndata", "-N"
        help = "Number of frames (from each molecule) to include in fit"
        arg_type = Int64
        default = 1
    "--rcut", "-r"
        help = "Cutoff radii (used for both onsite shpere and offsite cylinder)"
        arg_type = Float64
        default = 10.0
    "--zcut", "-z"
        help = "Bond Cutoff"
        arg_type = Float64
        default = 10.0
    "--degree", "-d"
        help = "Maximum polynomial degree for the ACE basis set."
        arg_type = Int64
        default = 2
    "--order", "-o"
        help = "Body order for the ACE basis set."
        arg_type = Int64
        default = 1
end

parsed_args = parse_args(ARGS, args)

# read data 
filenames = ["../data/propanol.h5", "../data/esanol.h5", "../data/acrolein.h5", "../data/phenol.h5", "../data/toluene.h5", "../data/acetaldehyde.h5"]
Ndata = parsed_args["ndata"] # number of training data we use for a single molecule
frames = []
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i =1:floor(10000/Ndata):10000 ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

# construct a model 
ao_dict = Dict( 6 => Dict("n_orbs" => [3,2,1], "maxdeg" => parsed_args["degree"], "ord" => parsed_args["order"], "rcut" => parsed_args["rcut"], "zcut" => parsed_args["zcut"]), 
                1 => Dict("n_orbs" => [2], "maxdeg" => parsed_args["degree"], "ord" => parsed_args["order"], "rcut" => parsed_args["rcut"], "zcut" => parsed_args["zcut"]), 
                8 => Dict("n_orbs" => [3,2,1], "maxdeg" => parsed_args["degree"], "ord" => parsed_args["order"], "rcut" => parsed_args["rcut"], "zcut" => parsed_args["zcut"]) )

println("Constructing the model ...")
println()

DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

println("Model constructed!")
println()

# fit the model
fit!(DM, frames; solver = ACEfit.SKLEARN_BRR())

# validate the model
fname = rand(filenames)
molecule = TrajectoryHDF5(fname)
frame = read_frame(molecule,rand(1:10000))
R, D = translate_frame(frame)["R"], translate_frame(frame)["D"] # read a random frame
@time D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix

# error in RMSE && relative error in D
E = norm(D_pred - D)/sqrt(size(D,1)*size(D,2))
println("Test RMSE = $E")
println()
E = norm(D_pred - D)/norm(D)
println("Relative error in D ||D - D_ref|| / ||D||= $E")
println()

# visualize the error
E = abs.(D_pred - D)
plt = contourf(E);
savefig("Error_Degree$(parsed_args["degree"])_Ord$(parsed_args["order"])")

# Manifold violation
# norm(D * D - D)
println("Manifold violation = $(norm(D_pred * D_pred - D_pred))")
# @assert norm(D_pred * D_pred - D_pred) ≤ (2*norm(D)+2)*norm(E) # an extremely rough error bound

println("Done.")