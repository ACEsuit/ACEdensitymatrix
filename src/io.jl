## A file that contains the codes to save and load the Lux model
import ACEbase: write_dict, read_dict

write_dict(l::Sample_Onsite_Radial) = Dict("__id__"=>"Sample_Onsite_Radial", "maxdeg" => l.maxdeg, "rcut" => l.rcut, "pin" => l.pin, "pout" => l.pout, "r0" => l.r0, "p" => l.p)
read_dict(::Val{:Sample_Onsite_Radial}, dict::Dict) = Sample_Onsite_Radial(dict["maxdeg"], dict["rcut"], dict["pin"], dict["pout"], dict["r0"], dict["p"])

write_dict(l::Sample_Offsite_Radial) = begin 
    try 
        Dict("__id__"=>"Sample_Offsite_Radial", "maxdeg" => l.maxdeg, "rcut1" => l.rcut1, "rcut2" => l.rcut2, "zcut" => l.zcut, "pin" => l.pin, "pout" => l.pout, "r0" => l.r0, "p" => l.p)
    catch
        Dict("__id__"=>"Sample_Offsite_Radial", "maxdeg" => l.maxdeg, "rcut" => l.rcut, "zcut" => l.zcut, "pin" => l.pin, "pout" => l.pout, "r0" => l.r0, "p" => l.p)
        # This line is added to make it compatible with the old version of the model
    end
end

read_dict(::Val{:Sample_Offsite_Radial}, dict::Dict) = begin
    try 
        Sample_Offsite_Radial(dict["maxdeg"], dict["rcut1"], dict["rcut2"], dict["zcut"], dict["pin"], dict["pout"], dict["r0"], dict["p"])
    catch 
        Sample_Offsite_Radial(dict["maxdeg"], dict["rcut"], dict["rcut"], dict["zcut"], dict["pin"], dict["pout"], dict["r0"], dict["p"]) 
        # This line is added to make it compatible with the old version of the model
    end
end

write_dict(m::On_Model{L}) where L = Dict("__id__"=>"On_Model", 
                "maxdeg" => m.model.layers.embed.layers.Rn.maxdeg,
                "ord" => m.model.layers.AA.basis.specs |> length,
                "cutoff" => write_dict(m.model.layers.embed.layers.Rn),
                "categories" => m.model.layers.embed.layers.δs.layers.categorical.basis.categories.list,
                "parameters" => [ m.ps.dot[i].W for i =1:length(m.ps.dot)], 
                "AA2BBmap" => [ m.model.layers.AA2BB.layers[i].op for i = 1:length(m.model.layers.AA2BB.layers)],
                "AA2BBpos" => [ m.model.layers.AA2BB.layers[i].pos for i = 1:length(m.model.layers.AA2BB.layers)],
                "n_orbs" => m.n_orbs,
                "L" => L-1, 
                "fitted" => m.fitted)

function read_dict(::Val{:On_Model}, dict::Dict)

    # recover radial basis
    sor = read_dict(dict["cutoff"])
    spec = natural_indices(Default_Polynomial_Type(sor.maxdeg))
    radial = Radial_basis(sor, spec)

    L = identity.(dict["L"])
    n_orbs = identity.(dict["n_orbs"])
    categories = identity.(dict["categories"])

    luxchain, ps, st = equivariant_operator(dict["maxdeg"], dict["ord"], radial, L, L, Vector(n_orbs), Vector(n_orbs); categories = categories, _get_cat = _get_cat_default, cat_extension = simple_extension, AA2BB = Dict("AA2BBmap" => dict["AA2BBmap"], "AA2BBpos" => dict["AA2BBpos"]))

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)

    # replace the parameters
    layer_set = ["layer_$i" for i in 1:length(dict["AA2BBmap"])]
    layer_set = Symbol.(layer_set)
    
    for i = 1:length(layer_set)
        @set! ps.dot.$(layer_set[i]).W = dict["parameters"][i]
    end

    return On_Model(luxchain, ps, st, SVector{L+1}(identity.(n_orbs)), dict["fitted"])

end

write_dict(m::Off_Model{L1,L2}) where {L1,L2} = Dict("__id__"=>"Off_Model", 
                "maxdeg" => m.model.layers.embed.layers.Rn.maxdeg,
                "ord" => m.model.layers.AA.basis.specs |> length,
                "cutoff" => write_dict(m.model.layers.embed.layers.Rn),
                "categories" => m.model.layers.embed.layers.δs.layers.categorical.basis.categories.list,
                "parameters" => [ m.ps.dot[i].W for i =1:length(m.ps.dot)], 
                "AA2BBmap" => [ m.model.layers.AA2BB.layers[i].op for i = 1:length(m.model.layers.AA2BB.layers)],
                "AA2BBpos" => [ m.model.layers.AA2BB.layers[i].pos for i = 1:length(m.model.layers.AA2BB.layers)],
                "n_orbs1" => m.n_orbs1,
                "n_orbs2" => m.n_orbs2,
                "L1" => L1-1,
                "L2" => L2-1,
                "fitted" => m.fitted)

function read_dict(::Val{:Off_Model}, dict::Dict)
    
    # recover radial basis
    sor = read_dict(dict["cutoff"])
    spec = natural_indices(Default_Polynomial_Type(sor.maxdeg))
    radial = Radial_basis(sor, spec)

    L1, L2 = dict["L1"], dict["L2"]
    n_orbs1, n_orbs2 = dict["n_orbs1"], dict["n_orbs2"]
    categories = identity.(dict["categories"])

    luxchain, ps, st = equivariant_operator(dict["maxdeg"], dict["ord"], radial, L1, L2, Vector(n_orbs1), Vector(n_orbs2); categories = categories, _get_cat = _get_cat_offsite, cat_extension = offsite_extension, AA2BB = Dict("AA2BBmap" => dict["AA2BBmap"], "AA2BBpos" => dict["AA2BBpos"]))

    # replace the parameters
    layer_set = ["layer_$i" for i in 1:length(dict["AA2BBmap"])]
    layer_set = Symbol.(layer_set)
    
    for i = 1:length(layer_set)
        @set! ps.dot.$(layer_set[i]).W = dict["parameters"][i]
    end

    return Off_Model(luxchain, ps, st, SVector{L1+1}(identity.(n_orbs1)), SVector{L2+1}(identity.(n_orbs2)), dict["fitted"])

end

write_dict(m::Density_Model{T}) where T = Dict("__id__"=>"Density_Model", "Models" => Dict([ (key => write_dict(m.Models[key])) for key in keys(m.Models)]))
read_dict(::Val{:Density_Model}, dict::Dict) = Density_Model(Dict([ (key => read_dict(dict["Models"][key]) ) for key in setdiff(keys(dict["Models"]),["__id__"])]))

# NOTE: To save a DM dict (write_dict(DM::Density_Model)), JLD.save could be a feasible choice - cf. "test/io_tests.jl".