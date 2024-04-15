using EquivariantModels: _get_cat_default,RPE_filter_long, closure, _linear_operator_L, _close, rpe_basis, _nlms2b, _gramian, LinearSearch, ConstLinearLayer, genmul!
using Polynomials4ML: LinearLayer
import EquivariantModels: _rpi_A2B_matrix, _valtype, rpe_basis, RPE_filter

## Construct a new EQM that generates also tensorial basis
_valtype(op::AbstractMatrix{<: AbstractMatrix}, x::AbstractArray{<: Number}) = SMatrix{size(op[1],1), size(op[1],2), promote_type(eltype(op[1]), eltype(x[1][1]))}
_valtype(op::AbstractMatrix{<: AbstractMatrix}, x::AbstractArray{<: AbstractMatrix}) = SMatrix{size(op[1],1), size(op[1],2), promote_type(eltype(op[1]), eltype(x[1][1]))}

function _linear_operator_loc(L1, L2, C, pos, len)
    T = SMatrix{2L1+1,2L2+1,ComplexF64} 
    fL = let C=C, idx=pos#, T=T
        (res, aa) -> genmul!(res, C, aa[idx], *)
    end
    return LinearOperator{T}(size(C,1), len, false, false, fL, nothing, nothing; S = Vector{T})
 end

function rpe_basis(A::Rot3DCoeffs_loc{L1,L2,T}, nn::SVector{N, TN}, ll::SVector{N, Int}) where {L1, L2, T, N, TN}
    Ure, Mre = re_basis(A, ll)
    G = _gramian(nn, ll, Ure, Mre)
    S = svd(G)
    rk = rank(Diagonal(S.S); rtol =  1e-7)
    Urpe = S.U[:, 1:rk]'
    return Diagonal(sqrt.(S.S[1:rk])) * Urpe * Ure, Mre
end

function _rpi_A2B_matrix(cgen::Rot3DCoeffs_loc{L1,L2,T}, spec) where {L1,L2,T}
    # allocate triplet format
    Irow, Jcol = Int[], Int[]
    
    vals =  SMatrix{2L1+1,2L2+1,ComplexF64}[]
    
    # count the number of PI basis functions = number of rows
    idxB = 0
    # loop through all (zz, kk, ll) tuples; each specifies 1 to several B
    nnllset = []
    for i = 1:length(spec)
       # get the specification of the ith basis function, which is a tuple/vec of NamedTuples
       pib = spec[i]
       
       # get the rotation-coefficients for this basis group
       # the bs are the basis functions corresponding to the columns
       
       # The nnlllist is created because we want to consider each
       # (nn, ll) block only once.
       nn = SVector([onep.n for onep in pib]...)
       ll = SVector([onep.l for onep in pib]...) # get a SVector of ll index
       if haskey(pib[1],:s)
          ss = [onep.s for onep in pib]
       end
       
       if haskey(pib[1],:s)
          
          if (nn,ll,ss) in nnllset; continue; end
 
          # get the Mll indices and coeffs
          U, Mll = rpe_basis(cgen, nn, ll)
          # conver the Mlls into basis functions (NamedTuples)
       
          rpibs = [_nlms2b(nn, ll, mm, ss) for mm in Mll]
       
          if size(U, 1) == 0; continue; end
          # loop over the rows of Ull -> each specifies a basis function
          for irow = 1:size(U, 1)
             idxB += 1
             # loop over the columns of U / over brows
             for (icol, bcol) in enumerate(rpibs)
                # look for the index of basis bcol in spec
                bcol = sort(bcol)
                idxAA = LinearSearch(spec, bcol)
                if !isnothing(idxAA)
                   push!(Irow, idxB)
                   push!(Jcol, idxAA)
                   push!(vals, U[irow, icol])
                end
             end
          end
          push!(nnllset,(nn,ll,ss))
          
       else
          
          if (nn,ll) in nnllset; continue; end
 
          # get the Mll indices and coeffs
          # U, Mll = re_basis(cgen, ll)
          U, Mll = rpe_basis(cgen, nn, ll)
          # conver the Mlls into basis functions (NamedTuples)
       
          rpibs = [_nlms2b(nn, ll, mm) for mm in Mll]
       
          if size(U, 1) == 0; continue; end
          # loop over the rows of Ull -> each specifies a basis function
          for irow = 1:size(U, 1)
             idxB += 1
             # loop over the columns of U / over brows
             for (icol, bcol) in enumerate(rpibs)
                # look for the index of basis bcol in spec
                bcol = sort(bcol)
                idxAA = LinearSearch(spec, bcol)
                if !isnothing(idxAA)
                   push!(Irow, idxB)
                   push!(Jcol, idxAA)
                   if norm(U[irow, icol] - real.(U[irow, icol]))<1e-12
                      push!(vals, real.(U[irow, icol]))
                   else
                      push!(vals, U[irow, icol])
                   end
                   # push!(vals, U[irow, icol])
                end
             end
          end
          push!(nnllset,(nn,ll))
       
       end
       
    end
    # create CSC: [   triplet    ]  nrows   ncols
    return sparse(Irow, Jcol, vals, idxB, length(spec))
end

function equivariant_model_loc(spec_nlm, radial::Radial_basis, L::Int64; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true)
    # if rSH && L > 0
    #    error("rSH is only implemented (for now) for L = 0")
    # end
 
    # first filt out those unfeasible spec_nlm
    filter_init = RPE_filter_long(2*L)
    spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]
    
    # sort!(spec_nlm, by = x -> length(x))
    spec_nlm = closure(spec_nlm,filter_init; categories = categories)
    
    luxchain, ps, st = EquivariantModels.xx2AA(spec_nlm, radial; categories = categories, _get_cat = _get_cat, d = d, rSH = false, isState = isState)
    # F(X) = luxchain_tmp(X, ps, st)[1]
 
    LLset = [(l1,l2) for l1 = 0:L for l2 = 0:L]
    C = Vector{Any}(undef, length(LLset))
    pos = Vector{Any}(undef, length(LLset))
        
    
    for (l,(l1,l2)) in enumerate(LLset)
        filter = RPE_filter(l1+l2)
        cgen = Rot3DCoeffs_loc(l1,l2) # TODO: this should be made group related
 
        tmp = spec_nlm[findall(x -> filter(x) == 1, spec_nlm)]

        C[l] = _rpi_A2B_matrix(cgen, tmp)
        pos[l] = findall(x -> filter(x) == 1, spec_nlm) # [ dict[tmp[j]] for j = 1:length(tmp)]
    end

    l_sym = Lux.Parallel(nothing, [ConstLinearLayer(_linear_operator_loc(l1,l2,C[i],pos[i],length(spec_nlm))) for (i,(l1,l2)) in enumerate(LLset)]... )
    # C - A2Bmap
    luxchain = append_layer(luxchain, l_sym; l_name = :AA2BB)

    if isreal
        l_c2r = Lux.Parallel(nothing, [WrappedFunction(x -> real.(Ref(ctran(l1)) .* x .* Ref(ctran(l2)'))) for (l1,l2) in LLset]... )
        luxchain = append_layer(luxchain, l_c2r; l_name = :complex2real)
    end

    l_id = Lux.Parallel(nothing, [WrappedFunction(x -> identity.(x)) for i in 1:length(LLset)]... )
    luxchain = append_layer(luxchain, l_id; l_name = :TypeStablization)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)
    
    return luxchain, ps, st, LLset
 end
 
 # more constructors equivariant_model
 equivariant_model_loc(totdeg::Int64, ν::Int64, radial::Radial_basis, L::Int64; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true) = 
      equivariant_model_loc(degord2spec(radial; totaldegree = totdeg, order = ν, Lmax=L, islong = islong)[2], radial, L; categories, _get_cat, d, group, isState, isreal)
 
 # With the _close function, the input could simply be an nnlllist (nlist,llist)
 equivariant_model_loc(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L::Int64; categories=[], _get_cat = _get_cat_default, d=3, group = "O3", isState = true, isreal = true) = begin
    filter = RPE_filter_long(2*L)
    equivariant_model_loc(_close(nn, ll; filter = filter), radial, L; categories, _get_cat, d, group, isState, isreal)
 end

extend_n_orbs(n_orbs, LLset) = [ n_orbs[l1+1] * n_orbs[l2+1] for (l1,l2) in LLset]
extend_n_orbs(n_orbs) = extend_n_orbs(n_orbs, [(l1,l2) for l1 = 0:length(n_orbs)-1 for l2 = 0:length(n_orbs)-1])

function equivariant_operator(spec_nlm, radial::Radial_basis, Lmax::Int64, n_orbs::Vector{Int64}=ones(Int64,Lmax+1); categories=[], d=3, group="O3", isState=true, isreal = true)
    luxchain, ps, st, LLset = equivariant_model_loc(spec_nlm, radial, Lmax; categories, d = d, group = group, isState = isState, isreal)
    @assert length(n_orbs) == Lmax + 1
    @assert length(LLset)  == (Lmax + 1)^2
    ext_n_orbs = extend_n_orbs(n_orbs, LLset)

    len = [size(luxchain.layers.AA2BB.layers[i].op,1) for i = 1:(Lmax+1)^2]
    
    Linear_layer = Lux.Parallel(nothing, [Polynomials4ML.LinearLayer(len[i], ext_n_orbs[i]) for i = 1:(Lmax+1)^2]... )
    luxchain = append_layer(luxchain, Linear_layer; l_name = :dot)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)

    return luxchain, ps, st
end

function equivariant_operator(totdeg::Int64, ν::Int64, radial::Radial_basis, Lmax::Int64, n_orbs::Vector{Int64}=ones(Int64,Lmax+1); categories=[], d=3, group="O3", isState=true, isreal = true)
    equivariant_operator(degord2spec(radial; totaldegree = totdeg, order = ν, Lmax=Lmax, catagories = categories, islong = true)[2], radial, Lmax, n_orbs; categories, d = d, group = group, isState = isState, isreal)
end

function equivariant_operator(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, Lmax::Int64, n_orbs::Vector{Int64}=ones(Int64,Lmax+1); categories=[], d=3, group="O3", isState=true, isreal = true)
    filter = RPE_filter_long(2*Lmax)
    equivariant_operator(_close(nn, ll; filter = filter), radial, Lmax, n_orbs; categories, d = d, group = group, isState = isState, isreal)
end