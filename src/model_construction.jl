using EquivariantModels, Polynomials4ML
using EquivariantModels: simple_radial_basis, Radial_basis

## Temporal fix for an utils function in EquivariantModels
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

function onsite_basis(maxdeg::Int64, ord::Int64, radial::Radial_basis, Zi::Int64, Zs::Vector{Int64}, Lmax::Int64; islong = false)
    cats_ext = [(Zi,Z) for Z in Zs] |> unique
    Aspec, AAspec = degord2spec(radial; totaldegree = maxdeg, 
                                  order = ord, 
                                  Lmax = Lmax, catagories = cats_ext)

    luxchain, ps, st = equivariant_model(AAspec, radial, Lmax; categories=cats_ext, isState = true, islong = islong)

    #TODO: add one more LinearLayer to to complex to real transformation

    F(X) = luxchain(X, ps, st)[1]
    return F
end

function onsite_radial(maxdeg::Int64,rcut::Float64; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2)
    fcut(rcut::Float64,pin::Int=pin,pout::Int=pout) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
    ftrans(r0::Float64=r0,p::Int=p) = r -> ( (1+r0)/(1+r) )^p
    return EquivariantModels.simple_radial_basis(legendre_basis(maxdeg),fcut(rcut),ftrans())
end