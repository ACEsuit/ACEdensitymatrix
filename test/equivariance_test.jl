using LinearAlgebra

include("wigner.jl")
include("../src/utils/hdf5.jl")
include("../src/utils/reorder.jl")
include("../src/utils/transformations.jl")
include("../src/data_manupulation.jl")
include("../src/model_construction.jl")
molecule = TrajectoryHDF5("data/rotation_test.h5")

frame = read_frame(molecule,9999)
S = frame["Overlap"][:,:,1]
S2 = frame["Overlap"][:,:,2]
Q = frame["Rotation matrix"]

C_tilde = copy(frame["Coefficients"][:,:,1]')
C_tilde2 = copy(frame["Coefficients"][:,:,2]')

u, e, v = svd(S)
sqrt_overlap = u * diagm(sqrt.(e)) * v'
C = sqrt_overlap * C_tilde
C = apply_reorder(frame["Basis set labels"], C; debug=false)
D = C * C'

u, e, v = svd(S2)
sqrt_overlap2 = u * diagm(sqrt.(e)) * v'
C2 = sqrt_overlap2 * C_tilde2
C2 = apply_reorder(frame["Basis set labels"], C2; debug=false)
D2 = C2 * C2'

## Validate equivariance of the data
det(Q)  # rotation + reversion 
        # ==> -Q is a standard rotation that we should have sent to the wigner_D function. 
        # And after the reversion, there will be a sign change in the sp block 
        # (which we call a (0,1) block in some cases, and the sign change is (-1)^{L1+L2})

Sign = round(det(Q)) # 1 or -1 - indicating whether we have a pure rotation or a rotation + reversion

# Let's take the offsite Carbon-Carbon block (D[1:14,21:34]) to validate the equivariance

D_CC = D[1:14,21:34]
D_CC2 = D2[1:14,21:34]
Lmax = 2
n_orb = [3, 2, 1]
@assert length(n_orb) == Lmax + 1

function L2pos(L; n_orb = n_orb, Lmax = Lmax)
    @assert L <= Lmax
    init_pos = L == 0 ? 1 : sum([n_orb[l+1] * (2l+1) for l in 0:L-1]) + 1
    return init_pos
end

L2pos(L,i) = L2pos(L)+(i-1)*(2L+1):L2pos(L)+2L+(i-1)*(2L+1)

for L1 in 0:Lmax
    for L2 in 0:Lmax
        T = ctran(L1)
        T2 = ctran(L2)
        W = wigner_D(L1,Sign * Q) # Wigner_D matrix in the ACE (complex SHs) context
        W2 = wigner_D(L2,Sign * Q) 
        W_new = T * W * T' # Wigner_D matrix in the real SHs context
        W2_new = T2 * W2 * T2'
        for i = 1:n_orb[L1+1]
            for j = 1:n_orb[L2+1]
                @show isapprox(D_CC[L2pos(L1,i),L2pos(L2,j)],Sign^(L1+L2)*W_new'*D_CC2[L2pos(L1,i),L2pos(L2,j)]*W2_new,atol=1e-5)
            end
        end
    end
end

# the EQM basis should have the same equivariance as the data above
# onsite
maxdeg = 4
ord = 2
rcut = 10.0
Zi = 6
Zs = [6,1,8]
Lmax = 2
n_orb = [3, 2, 1]
# construct the basis
md = On_Model(maxdeg, ord, rcut, Zi, Zs, Lmax, n_orb)

# R, QR, D_CC = eval_model(onsite_model, R), D_CC2 = eval_model(onsite_model, QR), DCC and DCC2 should pass the above test too
frame = read_frame(molecule,9999)
Cord1 = frame["Coordinates"][:,:,1]
Cord2 = frame["Coordinates"][:,:,2]
Spec = Int.(frame["Atomic numbers"][:,1])
R1 = [State(rr = SVector{3}(Cord1[:,J] - Cord1[:,1]), Zi = Spec[1], Zj = Spec[J]) for J in setdiff(1:size(Cord1,2), [1])]
R2 = [State(rr = SVector{3}(Cord2[:,J] - Cord2[:,1]), Zi = Spec[1], Zj = Spec[J]) for J in setdiff(1:size(Cord1,2), [1])]

D_CC = eval_model(md, R1)
D_CC2 = eval_model(md, R2)

# Validate equivariance of the onsite basis

for L1 in 0:Lmax
    for L2 in 0:Lmax
        T = ctran(L1)
        T2 = ctran(L2)
        W = wigner_D(L1,Sign * Q) # Wigner_D matrix in the ACE (complex SHs) context
        W2 = wigner_D(L2,Sign * Q) 
        W_new = T * W * T' # Wigner_D matrix in the real SHs context
        W2_new = T2 * W2 * T2'
        for i = 1:n_orb[L1+1]
            for j = 1:n_orb[L2+1]
                @show isapprox(D_CC[L2pos(L1,i),L2pos(L2,j)],Sign^(L1+L2)*W_new'*D_CC2[L2pos(L1,i),L2pos(L2,j)]*W2_new,atol=1e-5)
            end
        end
    end
end

# offsite
maxdeg = 6
ord = 1
rcut = 10.0
zcut = 10.0
Zi = 6
Zj = 6
Zs = [6,1,8]
L1 = 2
L2 = 2
n_orbs1 = [3,2,1]
n_orbs2 = [3,2,1]
# construct the basis
md= Off_Model(maxdeg, ord, rcut, zcut, Zi, Zj, Zs, L1, L2, n_orbs1, n_orbs2)

# R, QR, D_CC = eval_model(onsite_model, R), D_CC2 = eval_model(onsite_model, QR), DCC and DCC2 should pass the above test too
frame = read_frame(molecule,9999)
Cord1 = frame["Coordinates"][:,:,1]
Cord2 = frame["Coordinates"][:,:,2]
Spec = Int.(frame["Atomic numbers"][:,1])
R1 = [State(rr = SVector{3}(Cord1[:,J] - Cord1[:,1]), rr0 = SVector{3}(Cord1[:,5] - Cord1[:,1]), Zi = Spec[1], Zj = Spec[5], Zk = Spec[J], bond = false) for J in setdiff(1:size(Cord1,2), [1, 5])]
push!(R1, State(rr = SVector{3}(Cord1[:,5] - Cord1[:,1]), rr0 = SVector{3}(Cord1[:,5] - Cord1[:,1]), Zi = Spec[1], Zj = Spec[1], Zk = Spec[5], bond = true))
R2 = [State(rr = SVector{3}(Cord2[:,J] - Cord2[:,1]), rr0 = SVector{3}(Cord2[:,5] - Cord2[:,1]), Zi = Spec[1], Zj = Spec[5], Zk = Spec[J], bond = false) for J in setdiff(1:size(Cord1,2), [1, 5])]
push!(R2, State(rr = SVector{3}(Cord2[:,5] - Cord2[:,1]), rr0 = SVector{3}(Cord2[:,5] - Cord2[:,1]), Zi = Spec[1], Zj = Spec[1], Zk = Spec[5], bond = true))

D_CC = eval_model(md, R1)
D_CC2 = eval_model(md, R2)

# Validate equivariance of the onsite basis

for L1 in 0:Lmax
    for L2 in 0:Lmax
        T = ctran(L1)
        T2 = ctran(L2)
        W = wigner_D(L1,Sign * Q) # Wigner_D matrix in the ACE (complex SHs) context
        W2 = wigner_D(L2,Sign * Q) 
        W_new = T * W * T' # Wigner_D matrix in the real SHs context
        W2_new = T2 * W2 * T2'
        for i = 1:n_orb[L1+1]
            for j = 1:n_orb[L2+1]
                @show isapprox(D_CC[L2pos(L1,i),L2pos(L2,j)],Sign^(L1+L2)*W_new'*D_CC2[L2pos(L1,i),L2pos(L2,j)]*W2_new,atol=1e-5)
            end
        end
    end
end