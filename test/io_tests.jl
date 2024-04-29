include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")

# model construction
# parameters 
maxdeg = 4
ord = 1
rcut = 10.0
Zi = 6
Zs = [6,1,8]
Lmax = 2
n_orbs = [3,2,1]
# construct the basis
onsite_model = On_Model(maxdeg, ord, rcut, Zi, Zs, Lmax, n_orbs)
offsite_model = Off_Model(maxdeg, ord, rcut, rcut, Zi, Zi, Zs, Lmax, Lmax, n_orbs, n_orbs)

# read data
molecule = TrajectoryHDF5("data/propanol.h5")

frame = read_frame(molecule,2)
R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
R11 = get_state(R,1,1)
R15 = get_state(R,1,5)
@time eval_model(onsite_model, R11)
@time eval_model(offsite_model, R15)

md = offsite_model.model
l = md.layers.embed.layers.Rn
radial = offsite_radial_basis(l.maxdeg, l.rcut, l.zcut; l.pin, l.pout, l.r0, l.p, l.polynomial_type)
read_dict(write_dict(radial.Rnl)) == radial.Rnl

md = onsite_model.model
l = md.layers.embed.layers.Rn
radial = onsite_radial_basis(l.maxdeg, l.rcut; l.pin, l.pout, l.r0, l.p, l.polynomial_type)
read_dict(write_dict(radial.Rnl)) == radial.Rnl

d = write_dict(onsite_model)
onmd = read_dict(d)
eval_model(onmd, R11) == eval_model(onsite_model, R11)

d = write_dict(offsite_model)
offmd = read_dict(d)
eval_model(offmd, R15) == eval_model(offsite_model, R15)


# construct a whole model DM and use its submodel to test the above 
# use the below code to test the whole model

# d = write_dict(DM)
# dm = read_dict(d)
# @time eval_model(dm, R, translate_frame(frame)["ao_labels"]);
# @time eval_model(DM, R, translate_frame(frame)["ao_labels"]);

# eval_model(dm, R, translate_frame(frame)["ao_labels"]) == eval_model(DM, R, translate_frame(frame)["ao_labels"])