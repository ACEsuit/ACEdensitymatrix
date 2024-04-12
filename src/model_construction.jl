using EquivariantModels, Polynomials4ML, Lux, Random
using EquivariantModels: simple_radial_basis, Radial_basis, append_layer

include("utils/transformations.jl")
include("utils/extended_eqm.jl")

function onsite_basis(maxdeg::Int64, ord::Int64, radial::Radial_basis, Zi::Int64, Zs::Vector{Int64}, Lmax::Int64)
    cats_ext = [(Zi,Z) for Z in Zs] |> unique
    Aspec, AAspec = degord2spec(radial; totaldegree = maxdeg, 
                                  order = ord, 
                                  Lmax = Lmax, catagories = cats_ext)

    xx2BB, ps, st, LLset = equivariant_model_loc(AAspec, radial, Lmax; categories=cats_ext, isState = true)

    #TODO: add one more LinearLayer to to complex to real transformation
    l_c2r = Lux.Parallel(nothing, [WrappedFunction(x -> real.(Ref(ctran(l1)) .* x .* Ref(ctran(l2)'))) for (l1,l2) in LLset]... )
    xx2BB = append_layer(xx2BB, l_c2r; l_name = :complex2real)
    
    ps, st = Lux.setup(MersenneTwister(1234), xx2BB)

    F(X) = xx2BB(X, ps, st)[1]
    return F
end

function onsite_radial(maxdeg::Int64,rcut::Float64; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2)
    fcut(rcut::Float64,pin::Int=pin,pout::Int=pout) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
    ftrans(r0::Float64=r0,p::Int=p) = r -> ( (1+r0)/(1+r) )^p
    return EquivariantModels.simple_radial_basis(legendre_basis(maxdeg),fcut(rcut),ftrans())
end