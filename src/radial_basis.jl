## A file that contains all the codes that are related to radial basis construction
using Polynomials4ML, Lux
using Polynomials4ML: AbstractP4MLBasis, AbstractExplicitLayer, lux, natural_indices
using EquivariantModels: Radial_basis

export onsite_radial_basis, offsite_radial_basis,
       Sample_Onsite_Radial, Sample_Offsite_Radial,
       Identity_Onsite_Radial, Identity_Offsite_Radial,
       Default_Polynomial_Type

const Default_Polynomial_Type = legendre_basis

# TODO: A patch to EquivariantModels.simple_radial_basis - cutoff is done for something before transformation
# import EquivariantModels: simple_radial_basis
function simple_radial_basis_loc(basis::AbstractP4MLBasis,f_cut::Function=r->1,f_trans::Function=r->r; spec = nothing, isState = true)
    if isnothing(spec)
        try 
           spec = natural_indices(basis)
        catch 
           error("The specification of this Radial_basis should be given explicitly!")
        end
     end
  
     _norm(x) = isState ? norm(x.rr) : norm(x)
     return Radial_basis(Chain(split = Lux.Parallel(nothing; trans = WrappedFunction(x -> f_trans.(_norm.(x))), get_norm = WrappedFunction(x -> _norm.(x))), evaluation = Lux.Parallel(nothing; poly = lux(basis), cutoff = WrappedFunction(x -> f_cut.(x))), env = WrappedFunction(x -> x[1].*x[2]), ), spec)
end

# Standard cutoff function
fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)

# An example for radial basis for Onsite - decorated as an AbstractExplicitLayer

struct Sample_Onsite_Radial <: AbstractExplicitLayer
    maxdeg::Int
    rcut::Float64
    pin::Int # =2
    pout::Int # =2
    r0::Float64 # =2.0
    p::Int # =2
 end
 
(l::Sample_Onsite_Radial)(x::AbstractArray, ps, st) = (l(x), st)
(l::Sample_Onsite_Radial)(x::AbstractArray) = begin 
    c = simple_radial_basis_loc(Default_Polynomial_Type(l.maxdeg),fcut(l.rcut,l.pin,l.pout),r -> ( (1+l.r0)/(1+r) )^l.p).Rnl
    ps, st = Lux.setup(MersenneTwister(1234), c)
    return c(x,ps,st)[1]
end

Sample_Onsite_Radial(maxdeg::Int, rcut::Float64) = Sample_Onsite_Radial(maxdeg, rcut, 2, 2, 2.0, 2)
get_spec(l::Sample_Onsite_Radial) = natural_indices(Default_Polynomial_Type(l.maxdeg))

struct Identity_Onsite_Radial <: AbstractExplicitLayer
    maxdeg::Int
    rcut::Float64
end

function _identity_coordinate(radius, scale, label)
    scale > 0 || throw(ArgumentError("$label scale must be positive"))
    radius <= scale + sqrt(eps(Float64)) * max(scale, 1.0) ||
        throw(DomainError(radius, "$label radius exceeds its identity-basis scale $scale"))
    return clamp(2radius / scale - 1, -1.0, 1.0)
end

(l::Identity_Onsite_Radial)(x::AbstractArray, ps, st) = (l(x), st)
function (l::Identity_Onsite_Radial)(x::AbstractArray)
    radial = simple_radial_basis_loc(
        Default_Polynomial_Type(l.maxdeg), r -> one(r),
        r -> _identity_coordinate(r, l.rcut, "onsite")).Rnl
    ps, st = Lux.setup(MersenneTwister(1234), radial)
    return radial(x, ps, st)[1]
end

get_spec(l::Identity_Onsite_Radial) = natural_indices(Default_Polynomial_Type(l.maxdeg))

# function onsite_radial(maxdeg::Int64,rcut::Float64; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2)
#     # fcut(rcut::Float64,pin::Int=pin,pout::Int=pout) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
#     ftrans(r0::Float64=r0,p::Int=p) = r -> ( (1+r0)/(1+r) )^p
#     return EquivariantModels.simple_radial_basis_loc(legendre_basis(maxdeg),fcut(rcut),ftrans())
# end

function onsite_radial_basis(maxdeg::Int64, rcut::Float64; pin::Int=2, pout::Int=2,
                             r0::Float64=2.0, p::Int=2,
                             radial_mode::Symbol=:sample)
    layer = if radial_mode === :sample
        Sample_Onsite_Radial(maxdeg, rcut, pin, pout, r0, p)
    elseif radial_mode === :identity
        Identity_Onsite_Radial(maxdeg, rcut)
    else
        throw(ArgumentError("radial_mode $radial_mode not implemented"))
    end
    return Radial_basis(layer, natural_indices(Default_Polynomial_Type(maxdeg)))
end

# An example for radial basis for Offsite - decorated as an AbstractExplicitLayer

struct Sample_Offsite_Radial <: AbstractExplicitLayer
    maxdeg::Int
    rcut1::Float64
    rcut2::Float64
    # The above rcut1 and rcut2 could have different meaning
    # in the different choices of offsite environment, in the 
    # cylinder case, they mean the cutoff of the the cutoff 
    # of the bond direction and radial direction (that is 
    # perpendicular to the bond). In the "two-sphereres" case, 
    # they are radial cutoffs of the two spheres.
    zcut::Float64
    pin::Int # =2
    pout::Int # =2
    r0::Float64 # =2.0
    p::Int # =2
 end

(l::Sample_Offsite_Radial)(x::AbstractArray, ps, st) = (l(x), st)
(l::Sample_Offsite_Radial)(x::AbstractArray) = begin 
    ftrans = r -> ( (1+l.r0)/(1+r) )^l.p
    _norm(x) = norm(x.rr)
    c = Chain(split = Lux.Parallel(nothing; trans = WrappedFunction(x -> ftrans.(_norm.(x))), id = WrappedFunction(identity)), evaluation = Lux.Parallel(nothing; poly = lux(Default_Polynomial_Type(l.maxdeg)), cutoff = WrappedFunction(x -> [ f_env_offsite_new(x[i].rr,x[i].rr0,x[i].bond,l.rcut1,l.rcut2,l.zcut,l.pin,l.pout) for i = 1:length(x)])), env = WrappedFunction(x -> x[1].*x[2]), )
    ps, st = Lux.setup(MersenneTwister(1234), c)
    return c(x,ps,st)[1]
end

Sample_Offsite_Radial(maxdeg::Int, rcut::Float64, zcut::Float64) = Sample_Offsite_Radial(maxdeg, rcut, rcut, zcut, 2, 2, 2.0, 2)
Sample_Offsite_Radial(maxdeg::Int, rcut_I::Float64, rcut_J::Float64, zcut::Float64) = Sample_Offsite_Radial(maxdeg, rcut_I, rcut_J, zcut, 2, 2, 2.0, 2)
get_spec(l::Sample_Offsite_Radial) = natural_indices(Default_Polynomial_Type(l.maxdeg))

struct Identity_Offsite_Radial <: AbstractExplicitLayer
    maxdeg::Int
    rcut1::Float64
    rcut2::Float64
    zcut::Float64
end

(l::Identity_Offsite_Radial)(x::AbstractArray, ps, st) = (l(x), st)

function _identity_offsite_coordinate(l::Identity_Offsite_Radial, state)
    scale = state.bond ? l.zcut : (l.rcut1 + l.rcut2) / 2
    radius = state.bond ? norm(state.rr0) : norm(state.rr)
    return _identity_coordinate(radius, scale, "offsite")
end

function (l::Identity_Offsite_Radial)(x::AbstractArray)
    radial = Chain(
        split=Lux.Parallel(nothing;
            trans=WrappedFunction(states ->
                [_identity_offsite_coordinate(l, state) for state in states]),
            id=WrappedFunction(identity)),
        evaluation=Lux.Parallel(nothing;
            poly=lux(Default_Polynomial_Type(l.maxdeg)),
            cutoff=WrappedFunction(states -> ones(Float64, length(states)))),
        env=WrappedFunction(parts -> parts[1] .* parts[2]))
    ps, st = Lux.setup(MersenneTwister(1234), radial)
    return radial(x, ps, st)[1]
end

get_spec(l::Identity_Offsite_Radial) = natural_indices(Default_Polynomial_Type(l.maxdeg))

# radial basis for Offsite
# zcut - cutoff of the bond length
# rcut - cutoff of the radial direction (that is perpendicular to the bond)
# rzcut - cutoff on the bond direction - the cylinder has total height of 2*rzcut+lbond
function f_env_offsite(r,rbond,be::Bool,rcut::Float64,rzcut::Float64,zcut::Float64,pin::Int=2,pout::Int=2)
    lbond = norm(rbond) # length of bond
    z = dot(r,rbond)/lbond
    rr = norm(r - z*rbond)
    z = abs(z)# - lbond/2)
    if be == true
        return fcut(zcut,pin,pout)(norm(r))
    else be == false
        return fcut(rcut,pin,pout)(rr) * fcut(rzcut+lbond/2,pin,pout)(z)
    end
end

f_env_offsite(r,rbond,be::Bool,rcut::Float64,zcut::Float64,pin::Int=2,pout::Int=2) = f_env_offsite(r,rbond,be,rcut,rcut,zcut,pin,pout)

# new radial basis for Offsite - two seperated spheries with the same cutoff (should be different though)
# TODO: Type for r and rbond should be specified
function f_env_offsite_new(r,rbond,be::Bool,rcut_I::Float64,rcut_J::Float64,zcut::Float64,pin::Int=2,pout::Int=2)
    # lbond = norm(rbond) # length of bond
    rrI = norm(r + rbond)
    rrJ = norm(r - rbond)
    if be == true
        return fcut(zcut,pin,pout)(norm(r))
    else be == false
        return fcut(rcut_I,pin,pout)(rrI) + fcut(rcut_J,pin,pout)(rrJ)
    end
end

f_env_offsite_new(r,rbond,be::Bool,rcut::Float64,zcut::Float64,pin::Int=2,pout::Int=2) = f_env_offsite_new(r,rbond,be,rcut,rcut,zcut,pin,pout)

# function offsite_radial_basis(basis::AbstractP4MLBasis,f_cut::Function=r->1,f_trans::Function=r->r; spec = nothing)
function offsite_radial_basis(maxdeg::Int64, rcut_I::Float64, rcut_J::Float64,
                              zcut::Float64; pin::Int=2, pout::Int=2,
                              r0::Float64=2.0, p::Int=2,
                              radial_mode::Symbol=:sample)
    layer = if radial_mode === :sample
        Sample_Offsite_Radial(maxdeg, rcut_I, rcut_J, zcut, pin, pout, r0, p)
    elseif radial_mode === :identity
        Identity_Offsite_Radial(maxdeg, rcut_I, rcut_J, zcut)
    else
        throw(ArgumentError("radial_mode $radial_mode not implemented"))
    end
    return Radial_basis(layer, natural_indices(Default_Polynomial_Type(maxdeg)))
end

offsite_radial_basis(maxdeg::Int64, rcut::Float64, zcut::Float64; kwargs...) =
    offsite_radial_basis(maxdeg, rcut, rcut, zcut; kwargs...)
# Radial_basis(Sample_Offsite_Radial(maxdeg, rcut, zcut, pin, pout, r0, p), natural_indices(Default_Polynomial_Type(maxdeg)))