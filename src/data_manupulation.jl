using DecoratedParticles, StaticArrays, LinearAlgebra

include("../src/utils/hdf5.jl")
include("../src/utils/reorder.jl")

# translate a frame to a pair of data (R, D_R)
function translate_frame(frame::Dict{String,Array})
    Zs = Int64.(frame["Atomic numbers"]) 
    Rs = [ frame["Coordinates"][:,i] for i in 1:size(frame["Coordinates"],2) ]
    @assert length(Zs) == length(Rs)
    R = [State(rr = Rs[j], Z = Zs[j]) for j in 1:length(Zs)]

    C = copy(frame["Coefficients"]')
    S = frame["Overlap"]
    u, e, v = svd(S)
    sqrt_S = u * diagm(sqrt.(e)) * v'
    C = sqrt_S * C # Loewding transformation
    C = apply_reorder(frame["Basis set labels"], C; debug=false) # Reorder the basis set
    D = C * C' # Density matrix with correct ordering and on the manifold

    return (R, D)
end

function get_state(R,I,J)
    if I == J
        # Onsite local environment
        @assert I <= length(R)
        return [State(rr = SVector{3}(R[J].rr - R[I].rr), Zi = R[I].Z, Zj = R[J].Z) for J in setdiff(1:length(R), [I])]
    else
        # Offsite local environment - we have several options and here comes just one of them
        # TODO: Implement the other options
        @assert I <= length(R) && J <= length(R)
        RIJ = [State(rr = R[K].rr - R[I].rr, rr0 = R[J].rr - R[I].rr, ZK = R[K].Z, ZI = R[I].Z, ZJ = R[J].Z, bond = false) for K in setdiff(1:length(R), [I,J])]
        push!(RIJ, State(rr = R[J].rr - R[I].rr, rr0 = R[J].rr - R[I].rr, ZK = R[J].Z, ZI = R[I].Z, ZJ = R[J].Z, bond = true))
    end
end

function get_block(D::Matrix{Float64},I::Int64,J::Int64,ao_labels::Vector{String})
    # Get the block of the density matrix
    # I,J are the indices of the atoms
    # ao_labels are the labels of the basis set

    # first assure that the ao_labels are in the correct order
    ao_labels = apply_reorder(ao_labels)

    atom_ids, atom_symbols, shells, ls, ms = unpack(ao_labels)
    atom_ids .+= 1

    pos_I = findall(x->x==I, atom_ids)
    pos_J = findall(x->x==J, atom_ids)

    return D[pos_I, pos_J]
end

function get_block(D::Matrix{Float64},I::Int64,J::Int64,ao_labels::Vector{String},L1::Int64,L2::Int64)
    # Get the block of the density matrix
    # I,J are the indices of the atoms
    # L1,L2 are the angular momentum indices
    # ao_labels are the labels of the basis set
 
    # first assure that the ao_labels are in the correct order
    ao_labels = apply_reorder(ao_labels)

    atom_ids, atom_symbols, shells, ls, ms = unpack(ao_labels)
    atom_ids .+= 1

    pos_I = findall(x->x==I, atom_ids)
    pos_J = findall(x->x==J, atom_ids)

    pos_L1 = findall(x->x==L1, ls[pos_I])
    pos_L2 = findall(x->x==L2, ls[pos_J])

    return D[pos_I, pos_J][pos_L1, pos_L2]
end

function get_block(D::Matrix{Float64},I::Int64,J::Int64,ao_labels::Vector{String},L1::Int64,L2::Int64,μ1::Int64,μ2::Int64)
    # Get the block of the density matrix
    # I,J are the indices of the atoms
    # L1,L2 are the angular momentum indices
    # μ1,μ2 are the indices for the (L1,L2) blocks
    # ao_labels are the labels of the basis set
 
    # first assure that the ao_labels are in the correct order
    ao_labels = apply_reorder(ao_labels)

    atom_ids, atom_symbols, shells, ls, ms = unpack(ao_labels)
    atom_ids .+= 1

    pos_I = findall(x->x==I, atom_ids)
    pos_J = findall(x->x==J, atom_ids)

    pos_L1 = findall(x->x==L1, ls[pos_I])
    pos_L2 = findall(x->x==L2, ls[pos_J])

    pos_μ1 = 1+(μ1-1)*(2L1+1):2L1+1+(μ1-1)*(2L1+1)
    pos_μ2 = 1+(μ2-1)*(2L2+1):2L2+1+(μ2-1)*(2L2+1)

    return D[pos_I, pos_J][pos_L1, pos_L2][pos_μ1, pos_μ2]
end