function Density_Model_tuned(ao_dict::Dict{TP, Dict{String, Any}}) where TP # There is a potential risk that DM tries to convert an unknown type dictionary to a Density_Model
    Zs = collect(keys(ao_dict)) |> sort 
    on_classes = classify_ao_dict_on(ao_dict)
    off_classes = classify_ao_dict_off(ao_dict)
    # Here we assume that the atoms are symbolized by their atomic numbers which are integers
    
    # T = typeof(Zs[1])
    # @assert TP == T
    dict = Dict{Union{TP,Tuple{TP,TP}},AbstractModel}()

    for zs in on_classes
        push!(dict, zs[1] => On_Model_tuned(ao_dict[zs[1]]["maxdeg"], ao_dict[zs[1]]["ord"], ao_dict[zs[1]]["rcut"], zs[1], Zs, length(ao_dict[zs[1]]["n_orbs"])-1, ao_dict[zs[1]]["n_orbs"]))
        
        if length(zs) > 1
            AA2BB = Dict("AA2BBmap" => [ dict[zs[1]].model.layers.AA2BB.layers[i].op for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)],
                         "AA2BBpos" => [ dict[zs[1]].model.layers.AA2BB.layers[i].pos for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)])
            for k = 2:length(zs)
                push!(dict, zs[k] => On_Model_tuned(ao_dict[zs[k]]["maxdeg"], ao_dict[zs[k]]["ord"], ao_dict[zs[k]]["rcut"], zs[k], Zs, length(ao_dict[zs[k]]["n_orbs"])-1, ao_dict[zs[k]]["n_orbs"], AA2BB = AA2BB))
            end
        end
    end

    for zs in off_classes
        push!(dict, zs[1] => Off_Model_tuned(maximum([ao_dict[zs[1][1]]["maxdeg"],ao_dict[zs[1][2]]["maxdeg"]]), maximum([ao_dict[zs[1][1]]["ord"],ao_dict[zs[1][2]]["ord"]]), maximum([ao_dict[zs[1][1]]["rcut"], ao_dict[zs[1][2]]["rcut"]]), maximum([ao_dict[zs[1][1]]["zcut"], ao_dict[zs[1][2]]["zcut"]]), zs[1][1], zs[1][2], Zs, length(ao_dict[zs[1][1]]["n_orbs"])-1, length(ao_dict[zs[1][2]]["n_orbs"])-1, ao_dict[zs[1][1]]["n_orbs"], ao_dict[zs[1][2]]["n_orbs"]))

        if length(zs) > 1
            AA2BB = Dict("AA2BBmap" => [ dict[zs[1]].model.layers.AA2BB.layers[i].op for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)],
                         "AA2BBpos" => [ dict[zs[1]].model.layers.AA2BB.layers[i].pos for i = 1:length(dict[zs[1]].model.layers.AA2BB.layers)])
            for k = 2:length(zs)
                push!(dict, zs[k] => Off_Model_tuned(maximum([ao_dict[zs[k][1]]["maxdeg"],ao_dict[zs[k][2]]["maxdeg"]]), maximum([ao_dict[zs[k][1]]["ord"],ao_dict[zs[k][2]]["ord"]]), maximum([ao_dict[zs[k][1]]["rcut"], ao_dict[zs[k][2]]["rcut"]]), maximum([ao_dict[zs[k][1]]["zcut"], ao_dict[zs[k][2]]["zcut"]]), zs[k][1], zs[k][2], Zs, length(ao_dict[zs[k][1]]["n_orbs"])-1, length(ao_dict[zs[k][2]]["n_orbs"])-1, ao_dict[zs[k][1]]["n_orbs"], ao_dict[zs[k][2]]["n_orbs"], AA2BB = AA2BB) )
            end
        end
    end

    return Density_Model{TP}(dict)
end

On_Model_tuned(maxdeg::Int64, ord::Int64, rcut::Float64, Zi::T, Zs::Vector{T}, Lmax::Int64, n_orbs::Vector{Int64}=ones(Int64,Lmax+1); AA2BB=nothing) where{T} = 
                On_Model{Lmax+1}(equivariant_operator_tuned(maxdeg,ord,onsite_radial_basis(maxdeg, rcut),Lmax,n_orbs;categories=unique([(Zi,Z) for Z in Zs]), AA2BB = AA2BB)..., SVector{Lmax+1}(n_orbs),false)

Off_Model_tuned(maxdeg::Int64, ord::Int64, rcut::Float64, zcut::Float64, Zi::T, Zj::T, Zs::Vector{T}, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); AA2BB=nothing) where{T} =
                Off_Model{L1+1,L2+1}(equivariant_operator_tuned(maxdeg,ord,offsite_radial_basis(maxdeg, rcut, zcut),L1,L2,n_orbs1,n_orbs2;categories=union([(Zi,Zj,Zj,true)],unique([(Zi,Zj,Zk,false) for Zk in Zs])),_get_cat = _get_cat_offsite, cat_extension = offsite_extension, AA2BB = AA2BB)..., SVector{L1+1}(n_orbs1), SVector{L2+1}(n_orbs2), false)

equivariant_operator_tuned(totdeg::Int64, ν::Int64, radial::Radial_basis, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, cat_extension = simple_extension) = 
                equivariant_operator(totdeg, ν+1, radial, L1, L2, n_orbs1, n_orbs2; categories = categories, _get_cat = _get_cat, AA2BB = AA2BB, d = d, group = group, isState = isState, isreal = isreal, cat_extension = cat_extension, tuned_filter = (l1,l2,u) -> filter_tuned(l1,l2,u))

equivariant_operator_tuned(totdeg::Int64, ν::Int64, radial::Radial_basis, L::Int64, n_orbs::Vector{Int64}=ones(Int64,L1+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, cat_extension = simple_extension) = 
                equivariant_operator_tuned(totdeg, ν, radial, L, L, n_orbs, n_orbs; categories = categories, _get_cat = _get_cat, AA2BB = AA2BB, d = d, group = group, isState = isState, isreal = isreal, cat_extension = cat_extension)


function filter_tuned(l1::Int64, l2::Int64,ν::Int64)
    if l1 == l2 == 1
        return RPE_filter(2)
    else
        return bb -> (length(bb) == 0) || (length(bb) == 0) || ((length(bb) < ν) && (abs(sum(b.m for b in bb)) <= l1+l2) && iseven(sum(b.l for b in bb)+l1+l2)) && ( length(bb) == 1 && l1+l2 == 0 ? bb[1].l == 0 : true )
    end
end