using EquivariantModels, Polynomials4ML, Lux, Random
using EquivariantModels: simple_radial_basis, Radial_basis, append_layer, simple_extension
using Polynomials4ML: ScalarPoly4MLBasis, lux, natural_indices

include("utils/transformations.jl")
include("utils/extended_eqm.jl")
include("radial_basis.jl")

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

get_cutoff(model::On_Model) = model.model.layers.embed.layers.Rn.rcut
get_cutoff(model::Off_Model) = [model.model.layers.embed.layers.Rn.rcut1, model.model.layers.embed.layers.Rn.rcut2]

struct Density_Model{T}
    Models::Dict{Union{T,Tuple{T,T}},AbstractModel}
end

function get_cutoff(model::Density_Model)
    rcut_on = 0.0
    rcut_off = 0.0
    zcut = 0.0
    for key in keys(model.Models)
        if !(typeof(key) <: Tuple)
            rcut_on = max(rcut_on, get_cutoff(model.Models[key]))
        else
            rcut_off = max(rcut_off, get_cutoff(model.Models[key])[1])
            zcut = max(zcut, get_cutoff(model.Models[key])[2])
        end
    end
    return rcut_on, rcut_off, zcut
end

isfitted(model::AbstractModel) = model.fitted
isfitted(model::Density_Model) = all(isfitted(model.Models[key]) for key in keys(model.Models))
get_L(model::On_Model{L}) where L = (L-1,L-1)
get_L(model::Off_Model{L1,L2}) where {L1, L2} = (L1-1,L2-1)
get_norbs(model::On_Model) = (model.n_orbs,model.n_orbs)
get_norbs(model::Off_Model) = (model.n_orbs1,model.n_orbs2)
eval_model(model::AbstractModel, x::Union{PState{T}, Vector{PState{T}}}) where {T} = sub_densitymatrix(model.model(x, model.ps, model.st)[1], get_L(model)..., get_norbs(model)...)

# NOTE: Here we assume that the same type of atoms are discretized in the same way
# Input: `ao_dict`` contains a list of atomic numbers and the number of orbitals for each atom
# including the three parameters that define the basis set (maxdeg, ord, cutoff)
# Output: a Density Model that contains all the On_Models and Off_Models, storing as Dictionary
# When evaluating a Density Model, the model should not only know a whole State, but also the ao_labels

function classify_ao_dict_on(ao_dict::Dict{TP, Dict{String, Any}}) where TP
    Zs = collect(keys(ao_dict)) |> sort
    on_classes = Vector{Vector{TP}}()
    for Z in Zs
        pushed = false
        for C in on_classes
            if ao_dict[Z]["n_orbs"] == ao_dict[C[1]]["n_orbs"] # It is too strict, actually the cutoffs do not matter
                push!(C, Z)
                pushed = true
                break
            end
        end
        if !pushed
            push!(on_classes, [Z])
        end
    end
    return on_classes
end

function classify_ao_dict_off(ao_dict::Dict{TP, Dict{String, Any}}) where TP
    on_classes = classify_ao_dict_on(ao_dict)
    off_classes = Vector{Vector{Tuple{TP,TP}}}()
    for (k,Zs1) in enumerate(on_classes)
        for Zs2 in on_classes[k:end]
            push!(off_classes, [ Z1≤Z2 ? (Z1,Z2) : (Z2, Z1) for Z1 in Zs1 for Z2 in Zs2])
        end
    end
    return unique.(off_classes)
end

function Density_Model(ao_dict::Dict{TP, Dict{String, Any}}) where TP # There is a potential risk that DM tries to convert an unknown type dictionary to a Density_Model
    Zs = collect(keys(ao_dict)) |> sort 
    on_classes = classify_ao_dict_on(ao_dict)
    off_classes = classify_ao_dict_off(ao_dict)
    # Here we assume that the atoms are symbolized by their atomic numbers which are integers
    
    # T = typeof(Zs[1])
    # @assert TP == T
    dict = Dict{Union{TP,Tuple{TP,TP}},AbstractModel}()

    for zs in on_classes
        onsite_cutoff = haskey(ao_dict[zs[1]], "rcut_on") ? "rcut_on" : "rcut"
        push!(dict, zs[1] => On_Model(ao_dict[zs[1]]["maxdeg"], ao_dict[zs[1]]["ord"], ao_dict[zs[1]][onsite_cutoff], zs[1], Zs, length(ao_dict[zs[1]]["n_orbs"])-1, ao_dict[zs[1]]["n_orbs"]))
        
        if length(zs) > 1
            AA2BB = Dict("AA2BBmap" => [ dict[zs[1]].model.layers.AA2BB.layers[i].op for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)],
                         "AA2BBpos" => [ dict[zs[1]].model.layers.AA2BB.layers[i].pos for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)])
            for k = 2:length(zs)
                push!(dict, zs[k] => On_Model(ao_dict[zs[k]]["maxdeg"], ao_dict[zs[k]]["ord"], ao_dict[zs[k]][onsite_cutoff], zs[k], Zs, length(ao_dict[zs[k]]["n_orbs"])-1, ao_dict[zs[k]]["n_orbs"], AA2BB = AA2BB))
            end
        end
    end

    for zs in off_classes
        offsite_cutoff = haskey(ao_dict[zs[1][1]], "rcut_off") ? "rcut_off" : "rcut"
        push!(dict, zs[1] => Off_Model(maximum([ao_dict[zs[1][1]]["maxdeg"],ao_dict[zs[1][2]]["maxdeg"]]), maximum([ao_dict[zs[1][1]]["ord"],ao_dict[zs[1][2]]["ord"]]), ao_dict[zs[1][1]][offsite_cutoff], ao_dict[zs[1][2]][offsite_cutoff], maximum([ao_dict[zs[1][1]]["zcut"], ao_dict[zs[1][2]]["zcut"]]), zs[1][1], zs[1][2], Zs, length(ao_dict[zs[1][1]]["n_orbs"])-1, length(ao_dict[zs[1][2]]["n_orbs"])-1, ao_dict[zs[1][1]]["n_orbs"], ao_dict[zs[1][2]]["n_orbs"]))

        if length(zs) > 1
            AA2BB = Dict("AA2BBmap" => [ dict[zs[1]].model.layers.AA2BB.layers[i].op for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)],
                         "AA2BBpos" => [ dict[zs[1]].model.layers.AA2BB.layers[i].pos for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)])
            for k = 2:length(zs)
                push!(dict, zs[k] => Off_Model(maximum([ao_dict[zs[k][1]]["maxdeg"],ao_dict[zs[k][2]]["maxdeg"]]), maximum([ao_dict[zs[k][1]]["ord"],ao_dict[zs[k][2]]["ord"]]), ao_dict[zs[k][1]][offsite_cutoff], ao_dict[zs[k][2]][offsite_cutoff], maximum([ao_dict[zs[k][1]]["zcut"], ao_dict[zs[k][2]]["zcut"]]), zs[k][1], zs[k][2], Zs, length(ao_dict[zs[k][1]]["n_orbs"])-1, length(ao_dict[zs[k][2]]["n_orbs"])-1, ao_dict[zs[k][1]]["n_orbs"], ao_dict[zs[k][2]]["n_orbs"], AA2BB = AA2BB) )
            end
        end
    end

    return Density_Model{TP}(dict)
end

function eval_model(model::Density_Model, R::Union{PState{T}, Vector{PState{T}}}, ao_labels::Union{Vector{String},Matrix{String}}; retraction::Function=identity) where {T}
    # R is a global configuration - a State or a vector of State objects
    # ao_labels is the labels of atoms and the corresponding basis sets
    # the output is the density matrix, ordering as the ao_labels

    rcut_on, r_cut_off, zcut = get_cutoff(model)
    ao_labels = apply_reorder(ao_labels) # with this line, we fit the reordered Density matrix in the correct order but need to map it back to the original order

    atom_ids, atom_symbols, shells, ls, ms = unpack(ao_labels)
    atom_ids .+= 1

    D = zeros(Float64,length(atom_ids),length(atom_ids))
    TP = typeof(D)

    for I = 1:length(R)
        for J = I:length(R)
            pos_I = findall(x->x==I, atom_ids)
            pos_J = findall(x->x==J, atom_ids)
            if I == J
                md = model.Models[R[I].Z]
                D[pos_I,pos_J] = sub_densitymatrix(md.model(get_state(R,I,I;atom_filter=filter_on(rcut_on)), md.ps, md.st)[1],get_L(md)...,get_norbs(md)...; sym = true)
            else
                if R[I].Z > R[J].Z
                    md = model.Models[(R[J].Z,R[I].Z)]
                    D[pos_J,pos_I] = sub_densitymatrix(md.model(get_state(R,J,I;atom_filter=filter_off(r_cut_off, zcut)), md.ps, md.st)[1],get_L(md)...,get_norbs(md)...)
                    D[pos_I,pos_J] = D[pos_J,pos_I]'
                else
                    md = model.Models[(R[I].Z,R[J].Z)]
                    D[pos_I,pos_J] = sub_densitymatrix(md.model(get_state(R,I,J;atom_filter=filter_off(r_cut_off, zcut)), md.ps, md.st)[1],get_L(md)...,get_norbs(md)...)
                    D[pos_J,pos_I] = D[pos_I,pos_J]'
                end
            end
        end
    end

    return retraction(D)
    
end

# reset_cutoff function is used to reset the cutoff of the model (it does not change the model itself but create a new one with new cutoffs)
# after resetting the cutoff, we need to refit the model so isfitted model is always set to be false
reset_cutoff(model::Density_Model, r_cut::Float64, z_cut::Float64) = Density_Model( Dict([ (key => reset_cutoff(model.Models[key], r_cut, z_cut)) for key in keys(model.Models)] ) )
reset_cutoff(model::Density_Model, r_cut_on::Float64, r_cut_off1::Float64, r_cut_off2::Float64, z_cut::Float64) = 
            Density_Model( Dict([ (key => length(key) == 1 ? reset_cutoff(model.Models[key], r_cut_on, z_cut) : reset_cutoff(model.Models[key], r_cut_off1, r_cut_off2, z_cut)) for key in keys(model.Models)] ) )

function reset_cutoff(model::On_Model, r_cut::Float64, z_cut::Float64)
    degree = model.model.layers.embed.layers.Rn.maxdeg
    r_cut_old = model.model.layers.embed.layers.Rn.rcut
    if r_cut_old == r_cut
        @warn("The cutoff is already set to $r_cut. No change is made.")
        return model
    end
    Rn_new = onsite_radial_basis(degree, r_cut)
    embed_new = Lux.Parallel(nothing; Rn = Rn_new.Rnl, Ylm = model.model.layers.embed.layers.Ylm, δs = model.model.layers.embed.layers.δs)
    luxchain = Chain(embed = embed_new, A = model.model.layers.A, AA = model.model.layers.AA, AA2BB = model.model.layers.AA2BB, stablize = model.model.layers.stablize, dot = model.model.layers.dot)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)
    return On_Model(luxchain, ps, st, model.n_orbs, false)
end

function reset_cutoff(model::Off_Model, r_cut1::Float64, r_cut2::Float64, z_cut::Float64)
    degree = model.model.layers.embed.layers.Rn.maxdeg
    r_cut_old = try; (model.model.layers.embed.layers.Rn.rcut, model.model.layers.embed.layers.Rn.rcut); catch; (model.model.layers.embed.layers.Rn.rcut1, model.model.layers.embed.layers.Rn.rcut2); end
    z_cut_old = model.model.layers.embed.layers.Rn.zcut
    if r_cut_old == (r_cut1, r_cut2) && z_cut_old == z_cut
        @warn("The cutoffs are already set as is. No change is made.")
        return model
    end
    Rn_new = offsite_radial_basis(degree, r_cut1, r_cut2, z_cut)
    embed_new = Lux.Parallel(nothing; Rn = Rn_new.Rnl, Ylm = model.model.layers.embed.layers.Ylm, δs = model.model.layers.embed.layers.δs)
    luxchain = Chain(embed = embed_new, A = model.model.layers.A, AA = model.model.layers.AA, AA2BB = model.model.layers.AA2BB, stablize = model.model.layers.stablize, dot = model.model.layers.dot)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)
    return Off_Model(luxchain, ps, st, model.n_orbs1, model.n_orbs2, false)
end

reset_cutoff(model::Off_Model, r_cut::Float64, z_cut::Float64) = reset_cutoff(model, r_cut, r_cut, z_cut)

# An onsite submodel - input is a (local) one center environment, output is the corresponding onsite block of the density matrix
On_Model(maxdeg::Int64, ord::Int64, rcut::Float64, Zi::T, Zs::Vector{T}, Lmax::Int64, n_orbs::Vector{Int64}=ones(Int64,Lmax+1); AA2BB=nothing) where{T} = 
                On_Model{Lmax+1}(equivariant_operator(maxdeg,ord,onsite_radial_basis(maxdeg, rcut),Lmax,n_orbs;categories=unique([(Zi,Z) for Z in Zs]), AA2BB = AA2BB)..., SVector{Lmax+1}(n_orbs),false)


# get the categories of a offsite state
 _get_cat_offsite(x) = [ (x[i].Zi,x[i].Zj,x[i].Zk,x[i].bond) for i = 1:length(x) ]
 count_bond(bb) = sum( bb[i].s[4] for i = 1:length(bb) )
 function offsite_extension(AAspec, catagories)
    AAspec = simple_extension(AAspec, catagories)
    filter!(bb -> length(bb) > 0 && count_bond(bb) == 1, AAspec)
    return AAspec
 end

# An offsite submodel - input is a (local) two-center environment, output is the corresponding offsite block of the density matrix
Off_Model(maxdeg::Int64, ord::Int64, rcut::Float64, zcut::Float64, Zi::T, Zj::T, Zs::Vector{T}, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); AA2BB=nothing) where {T} = 
                Off_Model{L1+1,L2+1}(equivariant_operator(maxdeg,ord,offsite_radial_basis(maxdeg, rcut, zcut),L1,L2,n_orbs1,n_orbs2;categories=union([(Zi,Zj,Zj,true)],unique([(Zi,Zj,Zk,false) for Zk in Zs])),_get_cat = _get_cat_offsite, cat_extension = offsite_extension, AA2BB = AA2BB)..., SVector{L1+1}(n_orbs1), SVector{L2+1}(n_orbs2), false)

Off_Model(maxdeg::Int64, ord::Int64, rcut1::Float64, rcut2::Float64, zcut::Float64, Zi::T, Zj::T, Zs::Vector{T}, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); AA2BB=nothing) where {T} = 
                Off_Model{L1+1,L2+1}(equivariant_operator(maxdeg,ord,offsite_radial_basis(maxdeg, rcut1, rcut2, zcut),L1,L2,n_orbs1,n_orbs2;categories=union([(Zi,Zj,Zj,true)],unique([(Zi,Zj,Zk,false) for Zk in Zs])),_get_cat = _get_cat_offsite, cat_extension = offsite_extension, AA2BB = AA2BB)..., SVector{L1+1}(n_orbs1), SVector{L2+1}(n_orbs2), false)

# The above rcut1 and rcut2 could have different meaning
# in the different choices of offsite environment, in the 
# cylinder case, they mean the cutoff of the the cutoff 
# of the bond direction and radial direction (that is 
# perpendicular to the bond). In the "two-sphereres" case, 
# they are radial cutoffs of the two spheres.

# adhoc code transforming an output of on or off model to a sub density matrix
function sub_densitymatrix(x::NTuple{Len,Vector},L1::Int64,L2::Int64,n_orbs1::Union{Vector{Int64},SVector{L3,Int64}},n_orbs2::Union{Vector{Int64},SVector{L4,Int64}};sym = false) where {Len, L3, L4}
    @assert Len == (L1+1)*(L2+1) && L3 == L1 + 1 && L4 == L2 + 1
    LLset = [(l1,l2) for l1 = 0:L1 for l2 = 0:L2]

    # @assert unique(LLset) == LLset
    len1 = sum( (2i-1) * n_orbs1[i] for i = 1:length(n_orbs1) )
    len2 = sum( (2i-1) * n_orbs2[i] for i = 1:length(n_orbs2) )
    D = zeros(Float64,len1,len2)
    for (t,(l1,l2)) in enumerate(LLset)
        pos_init_x = l1 == 0 ? 1 : sum( (2i+1) * n_orbs1[i+1] for i = 0:l1-1 ) + 1
        pos_init_y = l2 == 0 ? 1 : sum( (2i+1) * n_orbs2[i+1] for i = 0:l2-1 ) + 1
        ijset = [(i,j) for i = 1:n_orbs1[l1+1] for j = 1:n_orbs2[l2+1]]
        for (k,(i,j)) in enumerate(ijset)
            D[pos_init_x + (2l1+1) * (i-1) : pos_init_x + (2l1+1) * i - 1, pos_init_y + (2l2+1) * (j-1) : pos_init_y + (2l2+1) * j - 1] = x[t][k]
        end
    end

    return sym == true ? (D + D')/2 : D
end
