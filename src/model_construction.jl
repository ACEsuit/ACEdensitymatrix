using EquivariantModels, Polynomials4ML, Lux, Random
using EquivariantModels: simple_radial_basis, Radial_basis, append_layer, simple_extension
using Polynomials4ML: ScalarPoly4MLBasis, lux, natural_indices

include("utils/transformations.jl")
include("utils/extended_eqm.jl")

# Should be removed after the next release of EquivariantModels
import EquivariantModels: specnlm2spec1p
function specnlm2spec1p(spec_nlm)
    spec1p = []
    for spec_nlm_i in spec_nlm
        push!(spec1p, spec_nlm_i...)
        unique!(spec1p)
    end
    lmax = [ spec1p[i].l for i = 1:length(spec1p) ] |> maximum
    nmax = [ spec1p[i].n for i = 1:length(spec1p) ] |> maximum
    return spec1p, lmax, nmax + 1
end

abstract type AbstractModel end

struct On_Model{L} <: AbstractModel where L 
    model::Chain
    ps::NamedTuple
    st::NamedTuple
    n_orbs::SVector{L,Int64}
    fitted::Bool
end

struct Off_Model{L1,L2} <: AbstractModel where {L1,L2} 
    model::Chain
    ps::NamedTuple
    st::NamedTuple
    n_orbs1::SVector{L1,Int64}
    n_orbs2::SVector{L2,Int64}
    fitted::Bool
end

struct Density_Model{T}
    Models::Dict{Union{T,Tuple{T,T}},AbstractModel}
end

isfitted(model::AbstractModel) = model.fitted
isfitted(model::Density_Model) = all(isfitted.(model.On_Models)) && all(isfitted.(model.Off_Models))
get_L(model::On_Model{L}) where L = (L-1,L-1)
get_L(model::Off_Model{L1,L2}) where {L1, L2} = (L1-1,L2-1)
get_norbs(model::On_Model) = (model.n_orbs,model.n_orbs)
get_norbs(model::Off_Model) = (model.n_orbs1,model.n_orbs2)
eval_model(model::AbstractModel, x::Union{State{T}, Vector{State{T}}}) where {T} = sub_densitymatrix(model.model(x, model.ps, model.st)[1], get_L(model)..., get_norbs(model)...)

# NOTE: Here we assume that the same type of atoms are discretized in the same way
# Input: `ao_dict`` contains a list of atomic numbers and the number of orbitals for each atom
# including the three parameters that define the basis set (maxdeg, ord, cutoff)
# Output: a Density Model that contains all the On_Models and Off_Models, storing as Dictionary
# When evaluating a Density Model, the model should not only know a whole State, but also the ao_labels
function Density_Model(ao_dict::Dict)
    Zs = Int.(collect(keys(ao_dict))) |> sort # Here we assume that the atoms are symbolized by their atomic numbers which are integers
    T = typeof(Zs[1])
    dict = Dict{Union{T,Tuple{T,T}},AbstractModel}()
    for i = 1:length(Zs)
        push!(dict, Zs[i] => On_Model(ao_dict[Zs[i]]["maxdeg"], ao_dict[Zs[i]]["ord"], ao_dict[Zs[i]]["rcut"], Zs[i], Zs, length(ao_dict[Zs[i]]["n_orbs"])-1, ao_dict[Zs[i]]["n_orbs"]))
        # in principle, j should start from i because we can then use the symmetry of the density matrix but let's keep it for now
        for j = i:length(Zs)
            # cutoff here is not so correct but let's keep it for now
            push!(dict, (Zs[i],Zs[j]) => Off_Model(ao_dict[Zs[i]]["maxdeg"], ao_dict[Zs[i]]["ord"], maximum([ao_dict[Zs[i]]["rcut"], ao_dict[Zs[j]]["rcut"]]), maximum([ao_dict[Zs[i]]["zcut"], ao_dict[Zs[j]]["zcut"]]), Zs[i], Zs[j], Zs, length(ao_dict[Zs[i]]["n_orbs"])-1, length(ao_dict[Zs[j]]["n_orbs"])-1, ao_dict[Zs[i]]["n_orbs"], ao_dict[Zs[j]]["n_orbs"]))
        end
    end
    return Density_Model(dict)
end

function eval_model(model::Density_Model, R::Union{State{T}, Vector{State{T}}}, ao_labels::Union{Vector{String},Matrix{String}}) where {T}
    # R is a global configuration - a State or a vector of State objects
    # ao_labels is the labels of atoms and the corresponding basis sets
    # the output is the density matrix, ordering as the ao_labels

    ao_labels = apply_reorder(ao_labels) # with this line, we fit the reordered Density matrix in the correct order but need to map it back to the original order

    atom_ids, atom_symbols, shells, ls, ms = unpack(ao_labels)
    atom_ids .+= 1

    D = zeros(Float64,length(atom_ids),length(atom_ids))

    for I = 1:length(R)
        for J = I:length(R)
            pos_I = findall(x->x==I, atom_ids)
            pos_J = findall(x->x==J, atom_ids)
            if I == J
                md = model.Models[R[I].Z]
                D[pos_I,pos_J] = sub_densitymatrix(md.model(get_state(R,I,I), md.ps, md.st)[1],get_L(md)...,get_norbs(md)...)
                D[pos_J,pos_I] = (D[pos_J,pos_I] + D[pos_I,pos_J]') / 2
            else
                if R[I].Z > R[J].Z
                    md = model.Models[(R[J].Z,R[I].Z)]
                    D[pos_J,pos_I] = sub_densitymatrix(md.model(get_state(R,J,I), md.ps, md.st)[1],get_L(md)...,get_norbs(md)...)
                    D[pos_I,pos_J] = D[pos_J,pos_I]'
                else
                    md = model.Models[(R[I].Z,R[J].Z)]
                    D[pos_I,pos_J] = sub_densitymatrix(md.model(get_state(R,I,J), md.ps, md.st)[1],get_L(md)...,get_norbs(md)...)
                    D[pos_J,pos_I] = D[pos_I,pos_J]'
                end
            end
        end
    end

    return D
    
end

# Standard cutoff function
fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)

# radial basis for Onsite
function onsite_radial(maxdeg::Int64,rcut::Float64; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2)
    # fcut(rcut::Float64,pin::Int=pin,pout::Int=pout) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
    ftrans(r0::Float64=r0,p::Int=p) = r -> ( (1+r0)/(1+r) )^p
    return EquivariantModels.simple_radial_basis(legendre_basis(maxdeg),fcut(rcut),ftrans())
    # TODO: something wrong with simple_radial_basis - cutoff is done for something before transformation
end

# An onsite submodel - input is a (local) one center environment, output is the corresponding onsite block of the density matrix
On_Model(maxdeg::Int64, ord::Int64, rcut::Float64, Zi::T, Zs::Vector{T}, Lmax::Int64, n_orbs::Vector{Int64}=ones(Int64,Lmax+1)) where{T} = 
                On_Model{Lmax+1}(equivariant_operator(maxdeg,ord,onsite_radial(maxdeg, rcut),Lmax,n_orbs;categories=unique([(Zi,Z) for Z in Zs]))..., SVector{Lmax+1}(n_orbs),false)


# radial basis for Offsite
function f_env_offsite(r,rbond,be::Bool,rcut::Float64,zcut::Float64,pin::Int=2,pout::Int=2)
    lbond = norm(rbond)
    z = dot(r,rbond)/lbond
    rr = norm(r - z*rbond)
    if be == true
        return fcut(rcut,pin,pout)(norm(r))
    else be == false
        return fcut(rcut,pin,pout)(rr) * fcut(zcut+lbond/2,pin,pout)(z)
    end
end

# function offsite_radial_basis(basis::ScalarPoly4MLBasis,f_cut::Function=r->1,f_trans::Function=r->r; spec = nothing)
function offsite_radial_basis(maxdeg::Int64, rcut::Float64=5.0, zcut::Float64=5.0; r0::Float64=2.0, p::Int=2)
    basis = legendre_basis(maxdeg)
    spec = natural_indices(basis)
    # if isnothing(spec)
    #    try 
    #       spec = natural_indices(basis)
    #    catch 
    #       error("The specification of this Radial_basis should be given explicitly!")
    #    end
    # end
    ftrans = r -> ( (1+r0)/(1+r) )^p
    _norm(x) = norm(x.rr)
    return Radial_basis(Chain(split = Lux.Parallel(nothing; trans = WrappedFunction(x -> ftrans.(_norm.(x))), id = WrappedFunction(identity)), evaluation = Lux.Parallel(nothing; poly = lux(basis), cutoff = WrappedFunction(x -> [ f_env_offsite(x[i].rr,x[i].rr0,x[i].bond,rcut,zcut) for i = 1:length(x)])), env = WrappedFunction(x -> x[1].*x[2]), ), spec)
 end

# get the categories of a offsite state
 _get_cat_offsite(x) = [ (x[i].Zi,x[i].Zj,x[i].Zk,x[i].bond) for i = 1:length(x) ]
 count_bond(bb) = sum( bb[i].s[4] for i = 1:length(bb) )
 function offsite_extension(AAspec, catagories)
    AAspec = simple_extension(AAspec, catagories)
    filter!(bb -> length(bb) > 0 && count_bond(bb) == 1, AAspec)
    return AAspec
 end

# An offsite submodel - input is a (local) two-center environment, output is the corresponding offsite block of the density matrix
Off_Model(maxdeg::Int64, ord::Int64, rcut::Float64, zcut::Float64, Zi::T, Zj::T, Zs::Vector{T}, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1)) where {T} = 
                Off_Model{L1+1,L2+1}(equivariant_operator(maxdeg,ord,offsite_radial_basis(maxdeg, rcut, zcut),L1,L2,n_orbs1,n_orbs2;categories=union([(Zi,Zj,Zj,true)],unique([(Zi,Zj,Zk,false) for Zk in Zs])),_get_cat = _get_cat_offsite, cat_extension = offsite_extension)..., SVector{L1+1}(n_orbs1), SVector{L2+1}(n_orbs2), false)

# adhoc code transforming an output of on or off model to a sub density matrix
function sub_densitymatrix(x::NTuple{Len,Vector{Matrix{T}}},L1::Int64,L2::Int64,n_orbs1::Union{Vector{Int64},SVector{L3,Int64}},n_orbs2::Union{Vector{Int64},SVector{L4,Int64}};sym = false) where {Len, L3, L4, T}
    @assert Len == (L1+1)*(L2+1) && L3 == L1 + 1 && L4 == L2 + 1
    LLset = [(l1,l2) for l1 = 0:L1 for l2 = 0:L2]

    # @assert unique(LLset) == LLset
    len1 = sum( (2i-1) * n_orbs1[i] for i = 1:length(n_orbs1) )
    len2 = sum( (2i-1) * n_orbs2[i] for i = 1:length(n_orbs2) )
    H = zeros(T,len1,len2)
    for (t,(l1,l2)) in enumerate(LLset)
        pos_init_x = l1 == 0 ? 1 : sum( (2i+1) * n_orbs1[i+1] for i = 0:l1-1 ) + 1
        pos_init_y = l2 == 0 ? 1 : sum( (2i+1) * n_orbs2[i+1] for i = 0:l2-1 ) + 1
        ijset = [(i,j) for i = 1:n_orbs1[l1+1] for j = 1:n_orbs2[l2+1]]
        for (k,(i,j)) in enumerate(ijset)
            H[pos_init_x + (2l1+1) * (i-1) : pos_init_x + (2l1+1) * i - 1, pos_init_y + (2l2+1) * (j-1) : pos_init_y + (2l2+1) * j - 1] = x[t][k]
        end
    end

    return sym == true ? (H + H')/2 : H
end
