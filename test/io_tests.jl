using JLD2, FileIO

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")
include("../src/io.jl")

# model construction - constructing a whole Density_Model from a ao_dict that indicating the atomic orbital dictionary
# parameters 
ao_dict_1 = Dict( 6 => Dict("n_orbs" => [3,2,1], "maxdeg" => 2, "ord" => 2, "rcut" => 10.0, "zcut" => 10.0), 
                  1 => Dict("n_orbs" => [2], "maxdeg" => 2, "ord" => 2, "rcut" => 10.0, "zcut" => 10.0),
                  8 => Dict("n_orbs" => [3,2,1], "maxdeg" => 2, "ord" => 2, "rcut" => 10.0, "zcut" => 10.0) )
ao_dict_2 = Dict( 6 => Dict("n_orbs" => [3,2,1], "maxdeg" => 2, "ord" => 2, "rcut_on" => 10.0, "rcut_off" => 10.0, "zcut" => 10.0), 
                  1 => Dict("n_orbs" => [2], "maxdeg" => 2, "ord" => 2, "rcut_on" => 10.0, "rcut_off" => 10.0, "zcut" => 10.0),
                  8 => Dict("n_orbs" => [3,2,1], "maxdeg" => 2, "ord" => 2, "rcut_on" => 10.0, "rcut_off" => 10.0, "zcut" => 10.0) )

DM1 = Density_Model(ao_dict_1) # a density matrix model corresponding to the atomic orbital dictionary
DM2 = Density_Model(ao_dict_2) # a density matrix model corresponding to the atomic orbital dictionary

# randomly extract submodels from the whole model
Zi = rand(keys(ao_dict_1))::Int
Zj = rand(keys(ao_dict_1))::Int
Zi, Zj = min(Zi,Zj), max(Zi,Zj)
onsite_model1 = DM1.Models[Zi]
offsite_model1 = DM1.Models[Zi, Zj]
onsite_model2 = DM2.Models[Zi]
offsite_model2 = DM2.Models[Zi, Zj]

# construct random configurations

Ron = [PState(rr = SVector{3}(rand(3)), Zi = Zi, Zj = rand(keys(ao_dict_1))) for i = 1:10]
Roff = begin 
    rr0 = SVector{3}(rand(3))
    Roff = [PState(rr = SVector{3}(rand(3)), rr0 = rr0, Zi = Zi, Zj = Zj, Zk = rand(keys(ao_dict_1)), bond = false) for i = 1:10]
    push!(Roff, PState(rr = rr0, rr0 = rr0, Zi = Zi, Zj = Zj, Zk = Zj, bond = true))
end

# test for IO - onsite radial basis 
md = onsite_model1.model
l = md.layers.embed.layers.Rn
@show read_dict(write_dict(l)) == l

md = onsite_model2.model
l = md.layers.embed.layers.Rn
@show read_dict(write_dict(l)) == l

# test for IO - offsite radial basis
md = offsite_model1.model
l = md.layers.embed.layers.Rn
@show read_dict(write_dict(l)) == l

md = offsite_model2.model
l = md.layers.embed.layers.Rn
@show read_dict(write_dict(l)) == l

# test for IO - onsite model
d = write_dict(onsite_model1)
onmd = read_dict(d)
eval_model(onmd, Ron) == eval_model(onsite_model1, Ron)

d = write_dict(onsite_model2)
onmd = read_dict(d)
eval_model(onmd, Ron) == eval_model(onsite_model2, Ron)

# test for IO - offsite model
d = write_dict(offsite_model1)
offmd = read_dict(d)
eval_model(offmd, Roff) == eval_model(offsite_model1, Roff)

d = write_dict(offsite_model2)
offmd = read_dict(d)
eval_model(offmd, Roff) == eval_model(offsite_model2, Roff)

# test for IO - Density_Model
# Here we need a example data
molecule = TrajectoryHDF5("data/propanol.h5")
frame = read_frame(molecule,rand(1:10000))
R, D, ao_labels = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"]

d = write_dict(DM1)
dm = read_dict(d)
@time eval_model(dm, R, ao_labels);
@time eval_model(DM1, R, ao_labels);
@show eval_model(dm, R, ao_labels) == eval_model(DM1, R, ao_labels)

d2 = write_dict(DM2)
dm = read_dict(d2)
@time eval_model(dm, R, ao_labels);
@time eval_model(DM2, R, ao_labels);
@show eval_model(dm, R, ao_labels) == eval_model(DM2, R, ao_labels)

# test for IO - save and load to a local file
save("test/model.jld2", d)
dm = load("test/model.jld2") |> read_dict
@time eval_model(dm, R, ao_labels);
@time eval_model(DM1, R, ao_labels);
@show eval_model(dm, R, ao_labels) == eval_model(DM1, R, ao_labels)

save("test/model.jld2", d2)
dm = load("test/model.jld2") |> read_dict
@time eval_model(dm, R, ao_labels);
@time eval_model(DM2, R, ao_labels);
@show eval_model(dm, R, ao_labels) == eval_model(DM2, R, ao_labels)