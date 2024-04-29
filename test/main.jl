using Statistics, Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")

# read data 
filenames = ["data/propanol.h5"]#, "data/esanol.h5", "data/acrolein.h5", "data/phenol.h5", "data/toluene.h5", "data/acetaldehyde.h5"]#, "data/aniline.h5", "data/nmacetamide.h5"]
Ndata = 120 # number of training data we use for a single molecule
frames = []
for fname in filenames
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(i)) for i =1:floor(10000/Ndata):10000 ]...) # constructing a training data set with Ndata frames for a single .h5 file
end
frames = identity.(frames)

# construct a model 
ao_dict = Dict( 6 => Dict("n_orbs" => [3,2,1], "maxdeg" => 6, "ord" => 1, "rcut" => 10.0, "zcut" => 10.0), 
                1 => Dict("n_orbs" => [2], "maxdeg" => 6, "ord" => 1, "rcut" => 10.0, "zcut" => 10.0),
                # 7 => Dict("n_orbs" => [3,2,1], "maxdeg" => 4, "ord" => 3, "rcut" => 10.0, "zcut" => 10.0), 
                8 => Dict("n_orbs" => [3,2,1], "maxdeg" => 6, "ord" => 1, "rcut" => 10.0, "zcut" => 10.0) )

DM = Density_Model(ao_dict::Dict) # a density matrix model corresponding to the atomic orbital dictionary

# fit the model
fit!(DM, frames)#; solver = ACEfit.QR(lambda = 1e-12, P = I))

# evaluate the model
# convert / something else <- translate
R, D = translate_frame(frames[1])["R"], translate_frame(frames[1])["D"]
D_pred = @time eval_model(DM, R, translate_frame(frames[1])["ao_labels"]) # predicted density matrix
R, D = translate_frame(frames[3])["R"], translate_frame(frames[3])["D"]
D_pred = @time eval_model(DM, R, translate_frame(frames[3])["ao_labels"]) # predicted density matrix

# maximum error
D_pred - D |> maximum

# visualize the error
E = abs.(D_pred - D)
norm(E) / norm(D) # relative error
contourf(E)
contourf(D)

RMSE = 0
for frame in frames
    R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
    D_pred = eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
    @time eval_model(DM, R, translate_frame(frame)["ao_labels"]) # predicted density matrix
    RMSE += norm(D_pred - D)^2 / size(D,1) / size(D,2) # RMSE error for a single frame - scaled
end
RMSE = RMSE / length(frames) |> sqrt # RMSE error for the training set

# Manifold violation
norm(D * D - D)
norm(D_pred * D_pred - D_pred)
@assert norm(D_pred * D_pred - D_pred) ≤ (2*norm(D)+2)*norm(E) # an extremely rough error bound