using EquivariantModels, Polynomials4ML, Lux, Random
using EquivariantModels: simple_radial_basis, Radial_basis, append_layer

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

struct On_Model <: AbstractModel
    model_on::Chain
    ps::NamedTuple
    st::NamedTuple
    fitted::Bool
end

struct Off_Model <: AbstractModel
    model_off::Chain
    ps::NamedTuple
    st::NamedTuple
    fitted::Bool
end

struct density_model
    On_Models::Vector{On_Model}
    Off_Models::Vector{Off_Model}
    # fitted::Bool
end

isfitted(model::AbstractModel) = model.fitted
get_L(model::On_Model) = Int(sqrt(length(model.ps.dot)))-1
function get_norbs(model::On_Model)
    n_orbs = [Int(sqrt(size(model.ps.dot[1].W,1)))]
    for i = 2:Int(sqrt(length(model.ps.dot)))
        push!(n_orbs, Int(size(model.ps.dot[i].W,1)/ n_orbs[1]))
    end
    return n_orbs
end
eval_model(model::AbstractModel, x::Union{State{T}, Vector{State{T}}}) where {T} = sub_densitymatrix(model.model_on(x, model.ps, model.st)[1])


# function density_model(maxdeg, ord, cutoffs, ao_labels)
#     Zs, n_orbs = get_info(ao_labels)
#     @assert length(Zs) == length(n_orbs)

#     return density_model(maxdeg, ord, cutoffs, Zs, n_orbs)
# end

# function density_model(maxdeg, ord, cutoffs, Zs, n_orbs)
#     on_bases = []
#     on_params = []
#     for (i,Zi) in enumerate(Zs)
#         Lmax = length(n_orbs[i])-1
#         radial = onsite_radial(maxdeg, cutoffs)
#         push!(on_bases, onsite_basis(maxdeg, ord, radial, Zi, Zs, Lmax))
#         on_param = [ zeros(size(basis.xx2BB.contents.layers.AA2BB.layers[i].op,1)) for i in 1:(Lmax+1)^2 ]
#         push!(on_params, on_param)
#     end

#     on_model = on_model(on_bases, on_params, false)

#     return density_model(on_model, [])
# end

# radial basis for onsite
function onsite_radial(maxdeg::Int64,rcut::Float64; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2)
    fcut(rcut::Float64,pin::Int=pin,pout::Int=pout) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
    ftrans(r0::Float64=r0,p::Int=p) = r -> ( (1+r0)/(1+r) )^p
    return EquivariantModels.simple_radial_basis(legendre_basis(maxdeg),fcut(rcut),ftrans())
end

# An onsite submodel - input is a (local) one center environment, output is the corresponding onsite block of the density matrix
On_Model(maxdeg::Int64, ord::Int64, rcut::Float64, Zi::T, Zs::Vector{T}, Lmax::Int64, n_orbs::Vector{Int64}=ones(Int64,Lmax+1)) where{T} = 
                On_Model(equivariant_operator(maxdeg,ord,onsite_radial(maxdeg, rcut),Lmax,n_orbs;categories=unique([(Zi,Z) for Z in Zs]))..., false)


# radial basis for Offsite

# single offsite basis

# adhoc code transforming an output of on or off model to a sub density matrix
function sub_densitymatrix(x::NTuple{Len,Vector{Matrix{T}}}) where {Len, T}
    L = Int(sqrt(Len)-1)
    LLset = [(l1,l2) for l1 = 0:L for l2 = 0:L]
    pos = findall(x->x[1]==x[2],LLset)
    n_orbs = Int.(sqrt.(length.(x[pos])))

    @assert length(x) == length(LLset) # == length(n_orbs)
    # @assert unique(LLset) == LLset
    len = sum( (2i-1) * n_orbs[i] for i = 1:length(n_orbs) )
    H = zeros(T,len,len)
    for (t,(l1,l2)) in enumerate(LLset)
        pos_init_x = l1 == 0 ? 1 : sum( (2i+1) * n_orbs[i+1] for i = 0:l1-1 ) + 1
        pos_init_y = l2 == 0 ? 1 : sum( (2i+1) * n_orbs[i+1] for i = 0:l2-1 ) + 1
        ijset = [(i,j) for i = 1:n_orbs[l1+1] for j = 1:n_orbs[l2+1]]
        for (k,(i,j)) in enumerate(ijset)
            H[pos_init_x + (2l1+1) * (i-1) : pos_init_x + (2l1+1) * i - 1, pos_init_y + (2l2+1) * (j-1) : pos_init_y + (2l2+1) * j - 1] = x[t][k]
        end
    end

    return (H + H')/2
end
