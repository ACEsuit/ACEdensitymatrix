using Statistics, Plots

include("../src/utils/transformations.jl")
include("../src/data_manupulation.jl")
include("../src/model_construction.jl")

# read data
molecule = TrajectoryHDF5("data/propanol.h5")

frame = read_frame(molecule,2)
R, D = translate_frame(frame)
R11 = get_state(R,1,1)
D11_sp = get_block(D,1,1,frame["Basis set labels"],0,1,1,2)

Rs = []
Ys = []
N_data = 2000
i = 1

while length(Rs) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_sp = get_block(D,1,1,frame["Basis set labels"],0,1,1,2)
    push!(Rs, R11)
    push!(Ys, D11_sp)
    i += 5
end

Rs
Ys
Y = ones(length(Ys)*length(Ys[1]))
for i in 1:length(Ys)
    Y[(i-1)*length(Ys[1])+1:i*length(Ys[1])] = Ys[i]
end

Y |> norm

# model construction
# parameters 
maxdeg = 4
ord = 3
rcut = 10.0
radial = onsite_radial(maxdeg, rcut)
Zi = 6
Zs = [6,1,8]
Lmax = 1
# construct the basis
basis = onsite_basis(maxdeg, ord, radial, Zi, Zs, Lmax; islong = false)

# model training

l_basis = size(basis.luxchain.layers.BB.op,1)
A = zeros(length(Rs)*length(Ys[1]), l_basis)

function flat(a)
    a = real(a)
    tmp = ones(length(a[1]),length(a))
    for i in 1:length(a)
        tmp[:,i] = a[i]
    end
    return tmp
end

for i = 1:length(Rs)
    A[(i-1)*length(Ys[1])+1:i*length(Ys[1]),:] = flat(Ref(ctran(Lmax)) .* basis(Rs[i]))    
end

C = (A'*A + 1e-11I) \ (A'*Y) # naive solver - just for illustration

A * C - Y  |> norm
norm(Y)
# testing

Rt = []
Yt = []
i = 2

while length(Rt) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_sp = get_block(D,1,1,frame["Basis set labels"],0,1,1,2)
    push!(Rt, R11)
    push!(Yt, D11_sp)
    i += 5
end

At = zeros(length(Rt)*length(Yt[1]), l_basis)

for i = 1:length(Rs)
    At[(i-1)*length(Yt[1])+1:i*length(Yt[1]),:] = flat(Ref(ctran(Lmax)) .* basis(Rt[i]))    
end

YY = ones(length(Yt)*length(Yt[1]))
for i in 1:length(Ys)
    YY[(i-1)*length(Yt[1])+1:i*length(Yt[1])] = Yt[i]
end
At * C - YY |> norm
YY |> norm

# plot and other comparisons
plot(YY, YY)
scatter!(Y, A * C, label = "Training")
scatter!(YY, At * C, label = "Testing")


e = A * C - Y |> maximum

## sd? 
D11_sd = get_block(D,1,1,frame["Basis set labels"],0,2,2,1)

Rs = []
Ys = []
N_data = 2000
i = 1

while length(Rs) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_sd = get_block(D,1,1,frame["Basis set labels"],0,2,3,1)
    push!(Rs, R11)
    push!(Ys, D11_sd)
    i += 5
end

Rs
Ys
Y = ones(length(Ys)*length(Ys[1]))
for i in 1:length(Ys)
    Y[(i-1)*length(Ys[1])+1:i*length(Ys[1])] = Ys[i]
end

Y |> norm

# model construction
# parameters 
maxdeg = 6
ord = 2
rcut = 10.0
radial = onsite_radial(maxdeg, rcut)
Zi = 6
Zs = [6,1,8]
Lmax = 2
# construct the basis
basis = onsite_basis(maxdeg, ord, radial, Zi, Zs, Lmax; islong = false)

# model training

l_basis = size(basis.luxchain.layers.BB.op,1)
A = zeros(length(Rs)*length(Ys[1]), l_basis)

for i = 1:length(Rs)
    A[(i-1)*length(Ys[1])+1:i*length(Ys[1]),:] = flat(Ref(ctran(Lmax)) .* basis(Rs[i]))    
end

C = (A'*A + 1e-11I) \ (A'*Y) # naive solver - just for illustration

A * C - Y  |> norm
norm(Y)
# testing

Rt = []
Yt = []
i = 2

while length(Rt) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_sp = get_block(D,1,1,frame["Basis set labels"],0,2,3,1)
    push!(Rt, R11)
    push!(Yt, D11_sp)
    i += 5
end

At = zeros(length(Rt)*length(Yt[1]), l_basis)

for i = 1:length(Rs)
    At[(i-1)*length(Yt[1])+1:i*length(Yt[1]),:] = flat(Ref(ctran(Lmax)) .* basis(Rt[i]))    
end

YY = ones(length(Yt)*length(Yt[1]))
for i in 1:length(Ys)
    YY[(i-1)*length(Yt[1])+1:i*length(Yt[1])] = Yt[i]
end
At * C - YY |> norm
YY |> norm

# plot and other comparisons
plot(YY, YY)
scatter!(Y, A * C, label = "Training")
scatter!(YY, At * C, label = "Testing")


e = A * C - Y |> maximum