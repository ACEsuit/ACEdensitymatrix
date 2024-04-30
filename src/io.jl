## A file that contains the codes to save and load the Lux model
import ACEbase: write_dict, read_dict

write_dict(l::Sample_Onsite_Radial) = Dict("__id__"=>"Sample_Onsite_Radial", "maxdeg" => l.maxdeg, "rcut" => l.rcut, "pin" => l.pin, "pout" => l.pout, "r0" => l.r0, "p" => l.p)
read_dict(::Val{:Sample_Onsite_Radial}, dict::Dict) = Sample_Onsite_Radial(dict["maxdeg"], dict["rcut"], dict["pin"], dict["pout"], dict["r0"], dict["p"])

write_dict(l::Sample_Offsite_Radial) = Dict("__id__"=>"Sample_Offsite_Radial", "maxdeg" => l.maxdeg, "rcut" => l.rcut, "zcut" => l.zcut, "pin" => l.pin, "pout" => l.pout, "r0" => l.r0, "p" => l.p)
read_dict(::Val{:Sample_Offsite_Radial}, dict::Dict) = Sample_Offsite_Radial(dict["maxdeg"], dict["rcut"], dict["zcut"], dict["pin"], dict["pout"], dict["r0"], dict["p"])

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

    # recover xx2AA map
    spec_nlm = degord2spec(radial; totaldegree = dict["maxdeg"], order = dict["ord"], Lmax = 2L, catagories = categories, filtered_extension = simple_extension, islong = true)[2]
    # first filt out those unfeasible spec_nlm
    filter_init = RPE_filter_long(2L)
    spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]
   
    # sort!(spec_nlm, by = x -> length(x))
    spec_nlm = closure(spec_nlm,filter_init; categories = categories)
    luxchain = EquivariantModels.xx2AA(spec_nlm, radial; categories = categories, _get_cat = _get_cat_default, d = 3, rSH = false, isState = true)[1]
    
    # recover AA2BB map
    LLset = [(l1,l2) for l1 = 0:L for l2 = 0:L]
    C = dict["AA2BBmap"]
    pos = dict["AA2BBpos"]

    l_sym = Lux.Parallel(nothing, [ConstLinearLayer_loc(identity.(C[i]),identity.(pos[i])) for i in 1:length(C)]... )
    # C - A2Bmap
    luxchain = append_layer(luxchain, l_sym; l_name = :AA2BB)

    l_real = WrappedFunction(cc -> Tuple([identity.(real.(cc[i])) for i = 1:length(cc) ]))
    luxchain = append_layer(luxchain, l_real; l_name = :stablize)

    ext_n_orbs = extend_n_orbs(n_orbs, n_orbs, LLset)

    len = [size(C[i],1) for i = 1:length(C)]
    
    # recover dot layer
    Linear_layer = Lux.Parallel(nothing, [Polynomials4ML.LinearLayer(len[i], ext_n_orbs[i]) for i = 1:(L+1)^2]... )
    luxchain = append_layer(luxchain, Linear_layer; l_name = :dot)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)

    # replace the parameters
    layer_set = ["layer_$i" for i in 1:length(C)]
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

    # recover xx2AA map
    spec_nlm = degord2spec(radial; totaldegree = dict["maxdeg"], order = dict["ord"], Lmax = L1+L2, catagories = categories, filtered_extension = offsite_extension, islong = true)[2]
    # first filt out those unfeasible spec_nlm
    filter_init = RPE_filter_long(L1+L2)
    spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]
   
    # sort!(spec_nlm, by = x -> length(x))
    spec_nlm = closure(spec_nlm,filter_init; categories = categories)
    luxchain = EquivariantModels.xx2AA(spec_nlm, radial; categories = categories, _get_cat = _get_cat_offsite, d = 3, rSH = false, isState = true)[1]
    
    # recover AA2BB map
    LLset = [(l1,l2) for l1 = 0:L1 for l2 = 0:L2]
    C = dict["AA2BBmap"]
    pos = dict["AA2BBpos"]

    l_sym = Lux.Parallel(nothing, [ConstLinearLayer_loc(identity.(C[i]),identity.(pos[i])) for i in 1:length(C)]... )
    # C - A2Bmap
    luxchain = append_layer(luxchain, l_sym; l_name = :AA2BB)

    l_real = WrappedFunction(cc -> Tuple([identity.(real.(cc[i])) for i = 1:length(cc) ]))
    luxchain = append_layer(luxchain, l_real; l_name = :stablize)

    ext_n_orbs = extend_n_orbs(n_orbs1, n_orbs2, LLset)

    len = [size(C[i],1) for i = 1:length(C)]
    
    # recover dot layer
    Linear_layer = Lux.Parallel(nothing, [Polynomials4ML.LinearLayer(len[i], ext_n_orbs[i]) for i = 1:(L1+1)*(L2+1)]... )
    luxchain = append_layer(luxchain, Linear_layer; l_name = :dot)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)

    # replace the parameters
    layer_set = ["layer_$i" for i in 1:length(C)]
    layer_set = Symbol.(layer_set)
    
    for i = 1:length(layer_set)
        @set! ps.dot.$(layer_set[i]).W = dict["parameters"][i]
    end

    return Off_Model(luxchain, ps, st, SVector{L1+1}(identity.(n_orbs1)), SVector{L2+1}(identity.(n_orbs2)), dict["fitted"])

end

write_dict(m::Density_Model{T}) where T = Dict("__id__"=>"Density_Model", "Models" => Dict([ (key => write_dict(m.Models[key])) for key in keys(m.Models)]))
read_dict(::Val{:Density_Model}, dict::Dict) = Density_Model(Dict([ (key => read_dict(dict["Models"][key]) ) for key in setdiff(keys(dict["Models"]),["__id__"])]))

# NOTE: To save a DM dict (write_dict(DM::Density_Model)), JLD.save could be a feasible choice - cf. "test/io_tests.jl".