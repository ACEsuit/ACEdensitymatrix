using Statistics

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
N_data = 10000

for i = 1:10:N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_ss = get_block(D,1,1,frame["Basis set labels"],0,0,2,3)[1]
    push!(Rs, R11)
    push!(Ys, D11_ss)
end

Rs
Y = Ys |> Vector{Float64}

Y .- mean(Y) |> norm

# model construction
maxdeg = 6
ord = 2
rcut = 6.0
radial = onsite_radial(maxdeg, rcut)
Zi = 6
Zs = [6,1,8]
Lmax = 0
basis = onsite_basis(maxdeg, ord, radial, Zi, Zs, Lmax; islong = false)

# model training

l_basis = size(basis.luxchain.layers.BB.op,1)
A = zeros(length(Rs), l_basis)

for i = 1:length(Rs)
    A[i,:] = real(basis(Rs[i]))
end

A

C = qr(A) \ Y
C = (A'*A + 1e-11I) \ (A'*Y)

A * C - Y  |> norm

# testing

Rt = []
Yt = []

for i = 2:10:N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    R11 = get_state(R,1,1)
    D11_ss = get_block(D,1,1,frame["Basis set labels"],0,0,2,3)[1]
    push!(Rt, R11)
    push!(Yt, D11_ss)
end

At = zeros(length(Rt), l_basis)

for i = 1:length(Rt)
    At[i,:] = real(basis(Rt[i]))
end

At * C - Yt |> norm
Yt .- mean(Yt) |> norm

molecule