using Statistics, Plots

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")

# read data
molecule = TrajectoryHDF5("data/propanol.h5")

frame = read_frame(molecule,2)
R, D = translate_frame(frame)
R11 = get_state(R,1,1)
D11_ss = get_block(D,1,1,frame["Basis set labels"],0,0,1,1)

Rs = []
Ys = []
N_data = 2000
i = 1

while length(Rs) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_ss = get_block(D,1,1,frame["Basis set labels"],0,0,1,2)[1]
    push!(Rs, R11)
    push!(Ys, D11_ss)
    i += 5
end

Rs
Y = Ys |> Vector{Float64}

Y .- mean(Y) |> norm

# model construction
# parameters 
maxdeg = 8
ord = 3
rcut = 10.0
radial = onsite_radial(maxdeg, rcut)
Zi = 6
Zs = [6,1,8]
Lmax = 0
# construct the basis
basis = onsite_basis(maxdeg, ord, radial, Zi, Zs, Lmax; islong = false)

# model training

l_basis = size(basis.luxchain.layers.BB.op,1)
A = zeros(length(Rs), l_basis)

for i = 1:length(Rs)
    A[i,:] = real(basis(Rs[i]))
end

Γ = I
λ = 1e-12
# C = (A'*A + λ*Γ) \ (A'*Y) # naive solver - just for illustration
C = qr([A; λ*Γ]) \ [Y; zeros(Float64,size(A,2))] # another naive solver

A * C - Y  |> norm

# testing

Rt = []
Yt = []
i = 2

while length(Rt) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_ss = get_block(D,1,1,frame["Basis set labels"],0,0,1,2)[1]
    push!(Rt, R11)
    push!(Yt, D11_ss)
    i += 5
end

At = zeros(length(Rt), l_basis)

for i = 1:length(Rt)
    At[i,:] = real(basis(Rt[i]))
end

At * C - Yt |> norm
Yt .- mean(Yt) |> norm

# plot and other comparisons
plot(Yt, Yt)
scatter!(Y, A * C, label = "Training")
scatter!(Yt, At * C, label = "Testing")


e = A * C - Y |> maximum

abs.((A * C - Y) ./ Y) |> maximum  # error ~ 0.2% - 1%
abs.((At * C - Yt) ./ Yt) |> maximum  # error ~ 0.2% - 1%

train_pred = A * C
test_pred = At * C
train_exact = Y
test_exact = Yt