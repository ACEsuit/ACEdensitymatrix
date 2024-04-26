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

function _rpi_A2B_matrix_real(cgen::Rot3DCoeffs_loc{L1,L2,T}, spec) where {L1,L2,T}
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
    return sparse(Irow, Jcol, SMatrix{2L1+1,2L2+1}.(Ref(ctran(L1)) .* vals .* Ref(ctran(L2)')), idxB, length(spec))
end

# function degord2spec_loc(radial::Radial_basis; totaldegree, order, Lmax, catagories = [], filtered_extension = simple_extension, wL = 1, islong = true, rSH = false, admissible = default_admissible)
#    # Rn = radial.radial_basis(totaldegree)
#    Ylm = CYlmBasis(totaldegree)

#    spec1p = make_nlms_spec(radial, Ylm; totaldegree = totaldegree, admissible = (br, by) -> br.n + wL * by.l <= totaldegree)
#    spec1p = sort(spec1p, by = (x -> x.n + x.l * wL))
#    spec1pidx = getspec1idx(spec1p, radial.Radialspec, Ylm)

#    # define sparse for n-correlations
#    tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
#    default_admissible = bb -> length(bb) == 0 || sum(b.n for b in bb) + wL * sum(b.l for b in bb) <= totaldegree

#    # to construct SS, SD blocks
#    if rSH
#       filter_ = RPE_filter_real(Lmax)
#    else
#       filter_ = islong ? RPE_filter_long(Lmax) : RPE_filter(Lmax)
#    end

#    specAA = gensparse(; NU = order, tup2b = tup2b, filter = filter_, 
#                         admissible = default_admissible,
#                         minvv = fill(0, order), 
#                         maxvv = fill(length(spec1p), order), 
#                         ordered = true)

#    spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
#    # map back to nlm
#    AAspec = getspecnlm(spec1p, spec)
#    if !isempty(catagories)
#       AAspec = filtered_extension(AAspec, catagories)
#    end
#    Aspec = specnlm2spec1p(AAspec)[1]
#    return Aspec, AAspec # Aspecgetspecnlm(spec1p, spec)
# end

# Can add a reduce = true/false option to simplify onsite basis
function equivariant_model_loc(spec_nlm, radial::Radial_basis, L1::Int64, L2::Int64; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true)

   # first filt out those unfeasible spec_nlm
   filter_init = RPE_filter_long(L1+L2)
   spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]
   
   # sort!(spec_nlm, by = x -> length(x))
   spec_nlm = closure(spec_nlm,filter_init; categories = categories)
   
   luxchain, ps, st = EquivariantModels.xx2AA(spec_nlm, radial; categories = categories, _get_cat = _get_cat, d = d, rSH = false, isState = isState)
   # F(X) = luxchain_tmp(X, ps, st)[1]

   LLset = [(l1,l2) for l1 = 0:L1 for l2 = 0:L2]
   C = Vector{Any}(undef, length(LLset))
   pos = Vector{Any}(undef, length(LLset))
       
   
   for (l,(l1,l2)) in enumerate(LLset)
       filter = RPE_filter(l1+l2)
       cgen = Rot3DCoeffs_loc(l1,l2) # TODO: this should be made group related

       tmp = spec_nlm[findall(x -> filter(x) == 1, spec_nlm)]

       C[l] = isreal ? _rpi_A2B_matrix_real(cgen, tmp) : _rpi_A2B_matrix(cgen, tmp)
       pos[l] = findall(x -> filter(x) == 1, spec_nlm) # [ dict[tmp[j]] for j = 1:length(tmp)]
   end

   l_sym = Lux.Parallel(nothing, [ConstLinearLayer(_linear_operator_loc(l1,l2,identity(C[i]),identity(pos[i]),length(spec_nlm))) for (i,(l1,l2)) in enumerate(LLset)]... )
   # C - A2Bmap
   luxchain = append_layer(luxchain, l_sym; l_name = :AA2BB)

   # if isreal
   #     l_c2r = Lux.Parallel(nothing, [WrappedFunction(x -> identity.(real.(Ref(ctran(l1)) .* x .* Ref(ctran(l2)')))) for (l1,l2) in LLset]... )
   #     luxchain = append_layer(luxchain, l_c2r; l_name = :complex2real)
   # end

   if isreal
      l_real = WrappedFunction(cc -> Tuple([identity.(real.(cc[i])) for i = 1:length(cc) ]))
      luxchain = append_layer(luxchain, l_real; l_name = :stablize)
   end

   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
   
   return luxchain, ps, st, LLset
end

equivariant_model_loc(spec_nlm, radial::Radial_basis, L::Int64; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true) = 
      equivariant_model_loc(spec_nlm, radial, L, L; categories, _get_cat, d, group, isState, isreal)
 
 # more constructors equivariant_model
equivariant_model_loc(totdeg::Int64, ν::Int64, radial::Radial_basis, L1::Int64, L2::Int64; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true, cat_extension = simple_extension) = 
      equivariant_model_loc(degord2spec(radial; totaldegree = totdeg, order = ν, Lmax = L1+L2, catagories = categories, filtered_extension = cat_extension, islong = islong)[2], radial, L1, L2; categories, _get_cat, d, group, isState, isreal)

 # With the _close function, the input could simply be an nnlllist (nlist,llist)

equivariant_model_loc(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L1::Int64, L2::Int64; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true) = begin
    filter = RPE_filter_long(L1+L2)
    equivariant_model_loc(_close(nn, ll; filter = filter), radial, L1, L2; categories, _get_cat, d, group, isState, isreal)
end

equivariant_model_loc(totdeg::Int64, ν::Int64, radial::Radial_basis, L; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true) = 
      equivariant_model_loc(totdeg, ν, radial, L, L; categories, _get_cat, d, group, isState, isreal)

equivariant_model_loc(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L; categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState = true, isreal = true) = 
      equivariant_model_loc(nn, ll, radial, L, L; categories, _get_cat, d, group, isState, isreal)

# extend_n_orbs(n_orbs, LLset) = [ n_orbs[l1+1] * n_orbs[l2+1] for (l1,l2) in LLset]
# extend_n_orbs(n_orbs) = extend_n_orbs(n_orbs, [(l1,l2) for l1 = 0:length(n_orbs)-1 for l2 = 0:length(n_orbs)-1])
extend_n_orbs(n_orbs1, n_orbs2) = [ n_orbs1[l1+1] * n_orbs2[l2+1] for l1 = 0:length(n_orbs1)-1, l2 = 0:length(n_orbs2)-1]
extend_n_orbs(n_orbs1, n_orbs2, LLset) = [ n_orbs1[l1+1] * n_orbs2[l2+1] for (l1,l2) in LLset]

function equivariant_operator(spec_nlm, radial::Radial_basis, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState=true, isreal = true)
    luxchain, ps, st, LLset = equivariant_model_loc(spec_nlm, radial, L1, L2; categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)
    @assert length(n_orbs1) == L1 + 1 && length(n_orbs2) == L2 + 1
    @assert length(LLset)  == (L1 + 1) * (L2 + 1)
    ext_n_orbs = extend_n_orbs(n_orbs1, n_orbs2, LLset)

    len = [size(luxchain.layers.AA2BB.layers[i].op,1) for i = 1:(L1+1)*(L2+1)]
    
    Linear_layer = Lux.Parallel(nothing, [Polynomials4ML.LinearLayer(len[i], ext_n_orbs[i]) for i = 1:(L1+1)*(L2+1)]... )
    luxchain = append_layer(luxchain, Linear_layer; l_name = :dot)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)

    return luxchain, ps, st
end

equivariant_operator(spec_nlm, radial::Radial_basis, L::Int64, n_orbs::Vector{Int64}=ones(Int64,L+1); categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState=true, isreal = true) = 
    equivariant_operator(spec_nlm, radial, L, L, n_orbs, n_orbs; categories = categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)

function equivariant_operator(totdeg::Int64, ν::Int64, radial::Radial_basis, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState=true, isreal = true, cat_extension = simple_extension)
   # equivariant_operator(degord2spec(radial; totaldegree = totdeg, order = ν, Lmax=maximum(L1+L2), catagories = categories, islong = true)[2], radial, L1, L2, n_orbs1, n_orbs2; categories = categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)
   equivariant_operator(degord2spec(radial; totaldegree = totdeg, order = ν, Lmax = L1+L2, catagories = categories, filtered_extension = cat_extension, islong = true)[2], radial, L1, L2, n_orbs1, n_orbs2; categories = categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)
end

equivariant_operator(totdeg::Int64, ν::Int64, radial::Radial_basis, L::Int64, n_orbs::Vector{Int64}=ones(Int64,L+1); categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState=true, isreal = true) = 
    equivariant_operator(totdeg, ν, radial, L, L, n_orbs, n_orbs; categories = categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)


function equivariant_operator(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L1::Int64, L2, Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); categories=[], d=3, group="O3", isState=true, isreal = true)
    filter = RPE_filter_long(L1+L2)
    # equivariant_operator(_close(nn, ll; filter = filter), radial, maximum(L1,L2), n_orbs; categories = categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)
    equivariant_operator(_close(nn, ll; filter = filter), radial, L1+L2, n_orbs; categories = categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)
end

equivariant_operator(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L::Int64, n_orbs::Vector{Int64}=ones(Int64,L+1); categories=[], _get_cat = _get_cat_default, d=3, group="O3", isState=true, isreal = true) = 
    equivariant_operator(nn, ll, radial, L, L, n_orbs, n_orbs; categories = categories, _get_cat = _get_cat, d = d, group = group, isState = isState, isreal = isreal)