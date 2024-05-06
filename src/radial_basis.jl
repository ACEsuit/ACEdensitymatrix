## A file that contains all the codes that are related to radial basis construction

const Default_Polynomial_Type = legendre_basis

# TODO: A patch to EquivariantModels.simple_radial_basis - cutoff is done for something before transformation
import EquivariantModels: simple_radial_basis
function simple_radial_basis(basis::ScalarPoly4MLBasis,f_cut::Function=r->1,f_trans::Function=r->r; spec = nothing, isState = true)
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
    c = simple_radial_basis(Default_Polynomial_Type(l.maxdeg),fcut(l.rcut,l.pin,l.pout),r -> ( (1+l.r0)/(1+r) )^l.p).Rnl
    ps, st = Lux.setup(MersenneTwister(1234), c)
    return c(x,ps,st)[1]
end

Sample_Onsite_Radial(maxdeg::Int, rcut::Float64) = Sample_Onsite_Radial(maxdeg, rcut, 2, 2, 2.0, 2)
get_spec(l::Sample_Onsite_Radial) = natural_indices(Default_Polynomial_Type(l.maxdeg))

# function onsite_radial(maxdeg::Int64,rcut::Float64; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2)
#     # fcut(rcut::Float64,pin::Int=pin,pout::Int=pout) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
#     ftrans(r0::Float64=r0,p::Int=p) = r -> ( (1+r0)/(1+r) )^p
#     return EquivariantModels.simple_radial_basis(legendre_basis(maxdeg),fcut(rcut),ftrans())
# end

onsite_radial_basis(maxdeg::Int64,rcut::Float64; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2) = Radial_basis(Sample_Onsite_Radial(maxdeg, rcut, pin, pout, r0, p), natural_indices(Default_Polynomial_Type(maxdeg)))

# An example for radial basis for Offsite - decorated as an AbstractExplicitLayer

struct Sample_Offsite_Radial <: AbstractExplicitLayer
    maxdeg::Int
    rcut::Float64
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
    c = Chain(split = Lux.Parallel(nothing; trans = WrappedFunction(x -> ftrans.(_norm.(x))), id = WrappedFunction(identity)), evaluation = Lux.Parallel(nothing; poly = lux(Default_Polynomial_Type(l.maxdeg)), cutoff = WrappedFunction(x -> [ f_env_offsite_new(x[i].rr,x[i].rr0,x[i].bond,l.rcut,l.zcut,l.pin,l.pout) for i = 1:length(x)])), env = WrappedFunction(x -> x[1].*x[2]), )
    ps, st = Lux.setup(MersenneTwister(1234), c)
    return c(x,ps,st)[1]
end

Sample_Offsite_Radial(maxdeg::Int, rcut::Float64, zcut::Float64) = Sample_Offsite_Radial(maxdeg, rcut, zcut, 2, 2, 2.0, 2)
get_spec(l::Sample_Offsite_Radial) = natural_indices(Default_Polynomial_Type(l.maxdeg))

# radial basis for Offsite
function f_env_offsite(r,rbond,be::Bool,rcut::Float64,zcut::Float64,pin::Int=2,pout::Int=2)
    lbond = norm(rbond) # length of bond
    z = dot(r,rbond)/lbond
    rr = norm(r - z*rbond)
    z = abs(z)# - lbond/2)
    if be == true
        return fcut(rcut,pin,pout)(norm(r))
    else be == false
        return fcut(rcut,pin,pout)(rr) * fcut(zcut+lbond/2,pin,pout)(z)
    end
end

# new radial basis for Offsite - two seperated spheries with the same cutoff (should be different though)
function f_env_offsite_new(r,rbond,be::Bool,rcut::Float64,zcut::Float64,pin::Int=2,pout::Int=2)
    # lbond = norm(rbond) # length of bond
    rrI = norm(r + rbond)
    rrJ = norm(r - rbond)
    if be == true
        return fcut(zcut,pin,pout)(norm(r))
    else be == false
        return fcut(rcut,pin,pout)(rrI) * fcut(rcut,pin,pout)(rrJ)
    end
end

# function offsite_radial_basis(basis::ScalarPoly4MLBasis,f_cut::Function=r->1,f_trans::Function=r->r; spec = nothing)
offsite_radial_basis(maxdeg::Int64, rcut::Float64=5.0, zcut::Float64=5.0; pin::Int=2, pout::Int=2, r0::Float64=2.0, p::Int=2) = Radial_basis(Sample_Offsite_Radial(maxdeg, rcut, zcut, pin, pout, r0, p), natural_indices(Default_Polynomial_Type(maxdeg)))