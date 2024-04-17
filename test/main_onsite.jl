using Statistics, Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
include("../src/fit.jl")

# model construction
# parameters 
maxdeg = 10
ord = 2
rcut = 10.0
Zi = 6
Zs = [6,1,8]
Lmax = 2
n_orbs = [3,2,1]
# construct the basis
onsite_model = On_Model(maxdeg, ord, rcut, Zi, Zs, Lmax, n_orbs)

# read data
molecule = TrajectoryHDF5("data/propanol.h5")

frame = read_frame(molecule,2)
R, D = translate_frame(frame)
R11 = get_state(R,1,1)
@time eval_model(onsite_model, R11)

# Constructed Rs and Ys and it is all set
Rs = []
Ys = []
N_data = 1000
i = 1

molecule = TrajectoryHDF5("data/propanol.h5")
while length(Rs) < N_data
    frame = read_frame(molecule,i)
    R, D = translate_frame(frame)
    kk = rand([1,5,8])
    @assert frame["Atomic numbers"][kk] == 6.0
    Rkk = get_state(R,kk,kk)
    Dkk = get_block(D,kk,kk,frame["Basis set labels"])
    push!(Rs, Rkk)
    push!(Ys, Dkk)
    i += 5
end
Rs = identity.(Rs)
Ys = identity.(Ys)

onsite_model = fit!(onsite_model, Rs, Ys)

onsite_model.fitted

# eval_model(onsite_model, Rs[1])[4:9,10:14]
# Ys[1][4:9,10:14]


E = abs.(eval_model(onsite_model, Rs[1]) - Ys[1])

contourf(1:14,1:14,E)

# pos = [1,2,3,10,11,12,13,14]
# contourf(1:8,1:8,E[pos,pos])

Yss = vec(Ys[1])
for i = 2:length(Ys)
    push!(Yss, vec(Ys[i])...)
end 

Yrss = vec(eval_model(onsite_model, Rs[1]))
for i = 2:length(Rs)
    @show i
    push!(Yrss, vec(eval_model(onsite_model, Rs[i]))...)
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
    R, D = translate_frame(frame)
    kk = rand([1,5,8])
    Rkk = get_state(R,kk,kk)
    Dkk = get_block(D,kk,kk,frame["Basis set labels"])
    push!(Rin, Rkk)
    push!(Yin, Dkk)
    i += 1
end

Rin = identity.(Rin)
Yin = identity.(Yin)

@time eval_model(onsite_model, Rin[1])-Yin[1]

# plot and other comparisons
Yins = vec(Yin[1])
for i = 2:length(Yin)
    push!(Yins, vec(Yin[i])...)
end 

Yinss = vec(eval_model(onsite_model, Rin[1]))
for i = 2:length(Rin)
    @show i
    push!(Yinss, vec(eval_model(onsite_model, Rin[i]))...)
end

# testing - extropolation

Rext = []
Yext = []
i = 5001

while length(Rext) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    kk = rand([1,5,8])
    Rkk = get_state(R,kk,kk)
    Dkk = get_block(D,kk,kk,frame["Basis set labels"])
    push!(Rext, Rkk)
    push!(Yext, Dkk)
    i += 1
end

Rext = identity.(Rext)
Yext = identity.(Yext)

@time eval_model(onsite_model, Rext[1])-Yext[1] |> norm

# plot and other comparisons
Yexts = vec(Yext[1])
for i = 2:length(Yext)
    push!(Yexts, vec(Yext[i])...)
end

Yextss = vec(eval_model(onsite_model, Rext[1]))
for i = 2:length(Rext)
    @show i
    push!(Yextss, vec(eval_model(onsite_model, Rext[i]))...)
end


# plot(Yt, Yt)

smallest = sortperm(abs.(Yss))
# smallest = 1:196000

posi = 1:10000
plot(Yss[smallest][posi], Yss[smallest][posi], label = "Ideal")
scatter(Yss[smallest][posi], Yrss[smallest][posi], label = "Training")
scatter!(Yins[smallest][posi], Yinss[smallest][posi], label = "Interpolation Testing")
scatter!(Yexts[smallest][posi], Yextss[smallest][posi], label = "Extrapolation Testing")