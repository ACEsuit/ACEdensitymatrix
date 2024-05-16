using JLD

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
onsite_model = DM1.Models[Zi]
offsite_model = DM1.Models[Zi, Zj]

# construct random configurations

Ron = [State(rr = SVector{3}(rand(3)), Zi = Zi, Zj = rand(keys(ao_dict_1))) for i = 1:10]
Roff = begin 
    rr0 = SVector{3}(rand(3))
    Roff = [State(rr = SVector{3}(rand(3)), rr0 = rr0, Zi = Zi, Zj = Zj, Zk = rand(keys(ao_dict_1)), bond = false) for i = 1:10]
    push!(Roff, State(rr = rr0, rr0 = rr0, Zi = Zi, Zj = Zj, Zk = Zj, bond = true))
end

# test for IO - onsite radial basis 
md = onsite_model.model
l = md.layers.embed.layers.Rn
@show read_dict(write_dict(l)) == l

# test for IO - offsite radial basis
md = offsite_model.model
l = md.layers.embed.layers.Rn
@show read_dict(write_dict(l)) == l

# test for IO - onsite model
d = write_dict(onsite_model)
onmd = read_dict(d)
eval_model(onmd, Ron) == eval_model(onsite_model, Ron)

# test for IO - offsite model
d = write_dict(offsite_model)
offmd = read_dict(d)
eval_model(offmd, Roff) == eval_model(offsite_model, Roff)

# test for IO - Density_Model
# Here we need a example data
molecule = TrajectoryHDF5("data/propanol.h5")
frame = read_frame(molecule,rand(1:10000))
R, D, ao_labels = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["ao_labels"]

d = write_dict(DM)
dm = read_dict(d)
@time eval_model(dm, R, ao_labels);
@time eval_model(DM, R, ao_labels);
@show eval_model(dm, R, ao_labels) == eval_model(DM, R, ao_labels)

# test for IO - save and load to a local file
save("test/model.jld", d)
dm = load("test/model.jld") |> read_dict
@time eval_model(dm, R, ao_labels);
@time eval_model(DM, R, ao_labels);
@show eval_model(dm, R, ao_labels) == eval_model(DM, R, ao_labels)