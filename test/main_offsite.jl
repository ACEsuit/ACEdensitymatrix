using Statistics, Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")

# model construction
# parameters 
maxdeg = 6
ord = 1
rcut = 10.0
zcut = 10.0
Zi = 6
Zj = 1
Zs = [6,1,8]
L1 = 2
L2 = 0
n_orbs1 = [3,2,1]
n_orbs2 = [2]
# construct the basis
offsite_model = Off_Model(maxdeg, ord, rcut, zcut, Zi, Zj, Zs, L1, L2, n_orbs1, n_orbs2)

# read data
molecule = TrajectoryHDF5("data/propanol.h5")

frame = read_frame(molecule,2)
R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
R12 = get_state(R,1,2)
R12[1]

offsite_model.model(R12, offsite_model.ps, offsite_model.st)

@time eval_model(offsite_model, R12)

# Constructed Rs and Ys and it is all set
Rs = []
Ys = []
N_data = 1000
i = 1

molecule = TrajectoryHDF5("data/propanol.h5")
while length(Rs) < N_data
    frame = read_frame(molecule,i)
    R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
    kk1 = rand([1,5,8])
    kk2 = rand([2,3,4,6,7,9,10,12])
    # kk2 = rand(setdiff([1,5,8],kk1))
    @assert frame["Atomic numbers"][kk1] == 6.0
    @assert frame["Atomic numbers"][kk2] == 1.0
    Rkk = get_state(R,kk1,kk2)
    Dkk = get_block(D,kk1,kk2,frame["Basis set labels"])
    push!(Rs, Rkk)
    push!(Ys, Dkk)
    i += 5
end
Rs = identity.(Rs)
Ys = identity.(Ys)

offsite_model = fit!(offsite_model, Rs, Ys; solver = ACEfit.LSQR())

offsite_model.fitted
offsite_model.ps.dot.layer_1.W
# eval_model(onsite_model, Rs[1])[4:9,10:14]
# Ys[1][4:9,10:14]


@time E = abs.(eval_model(offsite_model, Rs[1]) - Ys[1])

contourf(1:14,1:2,E)

# pos = [1,2,3,10,11,12,13,14]
# contourf(1:8,1:8,E[pos,pos])

Yss = vec(Ys[1])
for i = 2:length(Ys)
    push!(Yss, vec(Ys[i])...)
end 

Yrss = vec(eval_model(offsite_model, Rs[1]))
for i = 2:length(Rs)
    @show i
    push!(Yrss, vec(eval_model(offsite_model, Rs[i]))...)
end

Yss
Yrss

# plot(Yss[1:1000], Yss[1:1000])
# scatter!(Yss[1:5000], Yrss[1:5000])

# testing - interpolation

Rin = []
Yin = []
i = 1

while length(Rin) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
    kk1 = rand([1,5,8])
    kk2 = rand([2,3,4,6,7,9,10,12])
    # kk2 = rand(setdiff([1,5,8],kk1))
    Rkk = get_state(R,kk1,kk2)
    Dkk = get_block(D,kk1,kk2,frame["Basis set labels"])
    push!(Rin, Rkk)
    push!(Yin, Dkk)
    i += 1
end

Rin = identity.(Rin)
Yin = identity.(Yin)

@time eval_model(offsite_model, Rin[1])-Yin[1]

# plot and other comparisons
Yins = vec(Yin[1])
for i = 2:length(Yin)
    push!(Yins, vec(Yin[i])...)
end 

Yinss = vec(eval_model(offsite_model, Rin[1]))
for i = 2:length(Rin)
    @show i
    push!(Yinss, vec(eval_model(offsite_model, Rin[i]))...)
end

# testing - extropolation

Rext = []
Yext = []
i = 5001

while length(Rext) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
    kk1 = rand([1,5,8])
    # kk2 = rand(setdiff([1,5,8],kk1))
    kk2 = rand([2,3,4,6,7,9,10,12])
    Rkk = get_state(R,kk1,kk2)
    Dkk = get_block(D,kk1,kk2,frame["Basis set labels"])
    push!(Rext, Rkk)
    push!(Yext, Dkk)
    i += 1
end

Rext = identity.(Rext)
Yext = identity.(Yext)

@time eval_model(offsite_model, Rext[1])-Yext[1] |> norm

# plot and other comparisons
Yexts = vec(Yext[1])
for i = 2:length(Yext)
    push!(Yexts, vec(Yext[i])...)
end

Yextss = vec(eval_model(offsite_model, Rext[1]))
for i = 2:length(Rext)
    @show i
    push!(Yextss, vec(eval_model(offsite_model, Rext[i]))...)
end


# plot(Yt, Yt)

smallest = sortperm(abs.(Yss))
# smallest = 1:28000

posi = 1:8000
plot(Yss, Yss, label = "Ideal")
scatter!(Yss, Yrss, label = "Training")
scatter!(Yins, Yinss, label = "Interpolation Testing")
scatter!(Yexts, Yextss, label = "Extrapolation Testing")

# cross validation
# molecule = TrajectoryHDF5("data/propanol.h5")

# frame = read_frame(molecule,1)
# R, D = translate_frame(frame)["R"], translate_frame(frame)["D"]
# R11 = get_state(R,1,1)
# @time eval_model(onsite_model, R11) - D[1:14,1:14] |> norm