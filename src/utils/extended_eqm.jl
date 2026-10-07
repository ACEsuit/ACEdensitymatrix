using Random, EquivariantModels, Lux
using EquivariantModels: _get_cat_default,RPE_filter_long, closure, _linear_operator_L, _close, rpe_basis, _nlms2b, _gramian, LinearSearch, ConstLinearLayer, genmul!
using Polynomials4ML: LinearLayer
using Lux: AbstractExplicitLayer
import EquivariantModels: _rpi_A2B_matrix, _valtype, rpe_basis, RPE_filter, simple_extension

export equivariant_operator, extend_n_orbs, LinearLayer_loc, ConstLinearLayer_loc,
       _gepi_A2B_matrix_real

function _normalize_coupling_backend(backend::Symbol)
   backend in (:old, :legacy) && return :old
   backend in (:new, :gepi) && return :new
   throw(ArgumentError(
      "coupling_backend must be :old or :new " *
      "(:legacy and :gepi remain accepted as compatibility aliases)",
   ))
end

## Construct a new EQM that generates also tensorial basis
# TODO: Lux has also a ReshapeLayer that could be used - might be a way to get rid of Rot3DCoeffs_loc
_valtype(op::AbstractMatrix{<: AbstractMatrix}, x::AbstractArray{<: Number}) = SMatrix{size(op[1],1), size(op[1],2), promote_type(eltype(op[1]), eltype(x[1][1]))}
_valtype(op::AbstractMatrix{<: AbstractMatrix}, x::AbstractArray{<: AbstractMatrix}) = SMatrix{size(op[1],1), size(op[1],2), promote_type(eltype(op[1]), eltype(x[1][1]))}

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

# -----------------------------------------------------------------------------
# Category-aware GE-PI coupling construction
# -----------------------------------------------------------------------------

_nonmagnetic_label(onep) = haskey(onep, :s) ?
    (n=onep.n, l=onep.l, s=onep.s) : (n=onep.n, l=onep.l)

function _label_sort_key(label)
    return haskey(label, :s) ? (label.l, label.n, label.s) :
                              (label.l, label.n, ())
end

function _canonical_group_key(pib)
    labels = _nonmagnetic_label.(pib)
    sort!(labels; by=_label_sort_key)
    return Tuple(labels)
end

function _canonical_magnetic_tuple(pib)
    ordered = sort(
        collect(pib);
        by=onep -> (_label_sort_key(_nonmagnetic_label(onep)), onep.m),
    )
    return SVector{length(ordered),Int}(Tuple(onep.m for onep in ordered))
end

function _canonical_magnetic_tuple(pib, ::Val{N}) where {N}
    ordered = sort(
        collect(pib);
        by=onep -> (_label_sort_key(_nonmagnetic_label(onep)), onep.m),
    )
    return SVector{N,Int}(Tuple(onep.m for onep in ordered))
end

function _gepi_matrix_value(coupling, lambda, coefficient)
    complex_vector = lambda == 0 ?
        SVector{1,Float64}((coefficient,)) : coefficient
    real_vector = complex_to_real_vector(lambda, complex_vector)
    value = transform_λ(coupling, lambda, real_vector)

    # ACEoperators' odd-parity real matrix channel differs by an imaginary
    # phase from the complex spherical-harmonic AA input basis used here.
    # Without this phase, the subsequent `real.(...)` stabilization silently
    # discards the odd channel.
    return channel_parity(coupling.l, coupling.lp, lambda) == :odd ?
           im * value : value
end

function _gepi_channel_pattern(key)
    channel_map = Dict{Any,Int}()
    next_channel = 1
    return Tuple(
        get!(channel_map, haskey(label, :s) ? (label.n, label.s) : label.n) do
            channel = next_channel
            next_channel += 1
            channel
        end
        for label in key
    )
end

function _gepi_coefficients!(cache, lambda, ll, channels)
    cache_key = (lambda, ll, channels)
    return get!(cache, cache_key) do
        GEPICouplings.gepi_coupling_coeffs(lambda, ll, channels)
    end
end

function _append_gepi_coefficients!(
    row_indices::Vector{Int},
    column_indices::Vector{Int},
    values::Vector{MT},
    row_offset::Int,
    target_index::Dict{SVector{N,Int},Int},
    coupling,
    ::Val{K},
    coefficients,
    magnetic_set,
    tolerance::Real,
) where {MT,N,K}
    for local_row in axes(coefficients, 1)
        row_offset += 1
        for source_column in axes(coefficients, 2)
            column = get(target_index, magnetic_set[source_column], 0)
            column == 0 && continue
            value = MT(_gepi_matrix_value(
                coupling,
                K,
                coefficients[local_row, source_column],
            ))
            norm(value) <= tolerance && continue
            push!(row_indices, row_offset)
            push!(column_indices, column)
            push!(values, value)
        end
    end
    return row_offset
end

function _append_gepi_group!(
    row_indices,
    column_indices,
    values,
    row_offset,
    coupling,
    spec,
    key::NTuple{N},
    columns,
    coefficient_cache,
    lambdas,
    tolerance,
) where {N}
    target_index = Dict{SVector{N,Int},Int}()
    for column in columns
        magnetic = _canonical_magnetic_tuple(spec[column], Val(N))
        haskey(target_index, magnetic) &&
            error("duplicate canonical magnetic tuple in GE-PI group")
        target_index[magnetic] = column
    end

    ll = ntuple(index -> key[index].l, Val(N))
    channels = _gepi_channel_pattern(key)
    for lambda in lambdas
        coefficients, magnetic_set = _gepi_coefficients!(
            coefficient_cache, lambda, ll, channels,
        )
        row_offset = _append_gepi_coefficients!(
            row_indices,
            column_indices,
            values,
            row_offset,
            target_index,
            coupling,
            Val(lambda),
            coefficients,
            magnetic_set,
            tolerance,
        )
    end
    return row_offset
end

"""
    _gepi_A2B_matrix_real(Rot3DCoeffs_loc(L1, L2), spec)

Construct the real `(L1,L2)` matrix-valued A-to-B coupling map using the
local GE-PI Lie-algebra kernel.  Products are grouped by the complete
non-magnetic one-particle labels.  In particular, categorical labels `s` are
folded into the GE-PI channel identity, matching the EquivariantTensors
convention that its integer channel label distinguishes every non-magnetic
one-particle feature.
"""
function _gepi_A2B_matrix_real(
    ::Rot3DCoeffs_loc{L1,L2,T}, spec;
    tolerance::Real=1e-12,
    coefficient_cache=Dict{Any,Any}(),
) where {L1,L2,T}
    matrix_type = SMatrix{
        2L1 + 1, 2L2 + 1, ComplexF64, (2L1 + 1) * (2L2 + 1),
    }
    isempty(spec) && return sparse(Int[], Int[], matrix_type[], 0, 0)

    group_order = Any[]
    group_columns = Dict{Any,Vector{Int}}()
    for (column, pib) in enumerate(spec)
        key = _canonical_group_key(pib)
        if !haskey(group_columns, key)
            push!(group_order, key)
            group_columns[key] = Int[]
        end
        push!(group_columns[key], column)
    end

    coupling = BlockCoupling(L1, L2)
    row_indices = Int[]
    column_indices = Int[]
    values = matrix_type[]
    row_offset = 0

    for key in group_order
        columns = group_columns[key]
        row_offset = _append_gepi_group!(
            row_indices,
            column_indices,
            values,
            row_offset,
            coupling,
            spec,
            key,
            columns,
            coefficient_cache,
            abs(L1-L2):(L1+L2),
            tolerance,
        )
    end

    return sparse(row_indices, column_indices, values, row_offset, length(spec))
end

# Locally defined ConstLinearLayer and LinearLayer ========================
using Polynomials4ML: @reqfields, _make_reqfields
using ObjectPools: ArrayPool, FlexArray, FlexArrayCache, unwrap, release!

struct ConstLinearLayer_loc{T} <: AbstractExplicitLayer
   op::T
   pos::Vector{Int}

   ConstLinearLayer_loc{T}(op::T, pos::Vector{Int}) where {T} =
      new{T}(op, pos)
end

function _concrete_static_sparse(
   op::SparseMatrixCSC{TM,TI},
) where {TM<:StaticMatrix,TI<:Integer}
   concrete_type = typeof(zero(TM))
   TM === concrete_type && return op
   return SparseMatrixCSC(
      size(op, 1), size(op, 2),
      op.colptr, op.rowval, concrete_type.(op.nzval),
   )
end

function ConstLinearLayer_loc(
   op::SparseMatrixCSC{<:StaticMatrix},
   pos::AbstractVector{<:Integer},
)
   concrete_op = _concrete_static_sparse(op)
   return ConstLinearLayer_loc{typeof(concrete_op)}(
      concrete_op, Vector{Int}(pos),
   )
end

ConstLinearLayer_loc(op, pos::AbstractVector{<:Integer}) =
   ConstLinearLayer_loc{typeof(op)}(op, Vector{Int}(pos))

(l::ConstLinearLayer_loc)(x::AbstractArray, ps, st) = (l(x), st)
(l::ConstLinearLayer_loc)(x) = l.op * x[l.pos]

# The matrix-valued sparse entries use a fully concrete SMatrix element type.
# Their inline storage can therefore be viewed as scalar coefficients without
# copying. Accumulating directly into one flat scalar buffer avoids allocating
# an immutable SMatrix temporary for every sparse multiply-add.
function (l::ConstLinearLayer_loc{<:SparseMatrixCSC{TM}})(
    x::AbstractVector{TX},
) where {TM<:StaticMatrix,TX<:Number}
    @assert isbitstype(TM)
    output_type = typeof(zero(TM) * zero(TX))
    output_scalar_type = eltype(output_type)
    coefficient_scalar_type = eltype(TM)
    component_count = length(output_type)
    flat_result = zeros(
        output_scalar_type, component_count * size(l.op, 1),
    )
    scalar_values = reinterpret(coefficient_scalar_type, l.op.nzval)
    for local_column in axes(l.op, 2)
        input_value = x[l.pos[local_column]]
        for pointer in l.op.colptr[local_column]:(l.op.colptr[local_column+1]-1)
            output_offset = (l.op.rowval[pointer] - 1) * component_count
            value_offset = (pointer - 1) * component_count
            @inbounds @simd for component in 1:component_count
                flat_result[output_offset + component] +=
                    scalar_values[value_offset + component] * input_value
            end
        end
    end
    return reinterpret(output_type, flat_result)
end

struct LinearLayer_loc{FEATFIRST} <: AbstractExplicitLayer
   in_dim::Int
   out_dim::Int
   use_cache::Bool
end

LinearLayer_loc(in_dim::Int, out_dim::Int; feature_first = false, use_cache = true) = LinearLayer_loc{feature_first}(in_dim, out_dim, use_cache)

function (l::LinearLayer_loc)(x::AbstractVector, ps, st)
   out = ps.W * unwrap(x)
   release!(x)
   return out, st
end

function (l::LinearLayer_loc{true})(x::AbstractMatrix, ps, st)
   out = ps.W * unwrap(x)
   release!(x)
   return out, st
end

(l::LinearLayer_loc{false})(x::AbstractMatrix, ps, st) = begin
   out = unwrap(x) * transpose(ps.W)
   release!(x)
   return out, st
end

LuxCore.initialparameters(rng::AbstractRNG, l::LinearLayer_loc) = 
      ( W = randn(rng, l.out_dim, l.in_dim), )

LuxCore.initialstates(rng::AbstractRNG, l::LinearLayer_loc) = 
      ( l.use_cache ?  (pool =  ArrayPool(FlexArrayCache), ) 
                    : (pool =  ArrayPool(FlexArray), ))
## =====================================================================================
# fix something stupid in EQM
using EquivariantModels: make_nlms_spec, getspec1idx, gensparse, getspecnlm, specnlm2spec1p

_is_zero_channel(b) = b.n == 0 && b.l == 0 && b.m == 0

function _remove_redundant_zero_padding(AAspec)
   available = Set(Tuple(bb) for bb in AAspec)
   return filter(AAspec) do bb
      length(bb) == 1 && return true
      !any(eachindex(bb)) do i
         _is_zero_channel(bb[i]) &&
            Tuple(bb[j] for j in eachindex(bb) if j != i) in available
      end
   end
end

function degord2spec_loc(radial::Radial_basis; totaldegree, order, Lmax, catagories = [], filtered_extension = simple_extension, wL = 1, rSH = false, fixed_particle_number::Bool = false)
   # Rn = radial.radial_basis(totaldegree)
   if typeof(totaldegree) == Int64
      totaldegree = repeat([totaldegree], order)
   end
   Ylm = complex_sphericalharmonics(maximum(totaldegree))

   spec1p = make_nlms_spec(radial, Ylm; totaldegree = maximum(totaldegree), admissible = (br, by) -> br.n + wL * by.l <= maximum(totaldegree))
   spec1p = sort(spec1p, by = (x -> x.n + x.l * wL))
   spec1pidx = getspec1idx(spec1p, radial.Radialspec, Ylm)

   # define sparse for n-correlations
   tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
   default_admissible = bb -> length(bb) == 0 || sum(b.n for b in bb) + wL * sum(b.l for b in bb) <= totaldegree[length(bb)]

   # Matrix-valued density models construct all (l1, l2) output blocks.
   # Keep the complete admissible complex-SH product specification here;
   # each output block applies its own RPE_filter later.
   if rSH
      filter_ = bb -> (length(bb) == 0) || iseven(sum(b.l for b in bb) + Lmax) && ( length(bb) == 1 && Lmax == 0 ? bb[1].l == 0 : true )
   else
      filter_ = bb -> true
   end

   specAA = gensparse(; NU = order, tup2b = tup2b, filter = filter_, 
                        admissible = default_admissible,
                        minvv = fill(0, order), 
                        maxvv = fill(length(spec1p), order), 
                        ordered = true)

   spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
   # map back to nlm
   AAspec = getspecnlm(spec1p, spec)
   if !isempty(catagories)
      AAspec = filtered_extension(AAspec, catagories)
   end
   if fixed_particle_number
      AAspec = _remove_redundant_zero_padding(AAspec)
   end
   Aspec = specnlm2spec1p(AAspec)[1]
   return Aspec, AAspec # Aspecgetspecnlm(spec1p, spec)
end

# Can add a reduce = true/false option to simplify onsite basis
function equivariant_model_loc(spec_nlm, radial::Radial_basis, L1::Int64, L2::Int64; categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState = true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old)

   # # first filt out those unfeasible spec_nlm
   # filter_init = RPE_filter_long(L1+L2)
   # spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]
   
   # # sort!(spec_nlm, by = x -> length(x))
   # spec_nlm = closure(spec_nlm,filter_init; categories = categories)
   
   luxchain, ps, st = EquivariantModels.xx2AA(spec_nlm, radial; categories = categories, _get_cat = _get_cat, d = d, rSH = false, isState = isState)
   # F(X) = luxchain_tmp(X, ps, st)[1]

   LLset = [(l1,l2) for l1 = 0:L1 for l2 = 0:L2]
   if isnothing(AA2BB)
      coupling_backend = _normalize_coupling_backend(coupling_backend)
      C = Vector{Any}(undef, length(LLset))
      pos = Vector{Any}(undef, length(LLset))
      gepi_coefficient_cache = Dict{Any,Any}()
       
      for (l,(l1,l2)) in enumerate(LLset)
         filter = isnothing(tuned_filter) ? RPE_filter(l1+l2) : tuned_filter(l1,l2,maximum(length.(spec_nlm)))
         cgen = Rot3DCoeffs_loc(l1,l2) # TODO: this should be made group related

         tmp = spec_nlm[findall(x -> filter(x) == 1, spec_nlm)]

         C[l] = if coupling_backend == :new
            isreal || throw(ArgumentError(
               "the GE-PI backend currently supports isreal=true only",
            ))
            _gepi_A2B_matrix_real(
               cgen, tmp; coefficient_cache=gepi_coefficient_cache,
            )
         else
            isreal ? _rpi_A2B_matrix_real(cgen, tmp) :
                     _rpi_A2B_matrix(cgen, tmp)
         end
         pos[l] = findall(x -> filter(x) == 1, spec_nlm) # [ dict[tmp[j]] for j = 1:length(tmp)]
      end
   else
      C = AA2BB["AA2BBmap"]
      pos = AA2BB["AA2BBpos"]
   end

   # l_sym = Lux.Parallel(nothing, [ConstLinearLayer(_linear_operator_loc(l1,l2,identity(C[i]),identity(pos[i]),length(spec_nlm))) for (i,(l1,l2)) in enumerate(LLset)]... )
   # A temporary fix for the issue of the cost of the linear operator (a lack of suitable ConstLinearLayer)
   l_sym = Lux.Parallel(nothing, [ConstLinearLayer_loc(identity(C[i]),identity(pos[i])) for i in 1:length(LLset)]... )
   # # C - A2Bmap
   luxchain = append_layer(luxchain, l_sym; l_name = :AA2BB)

   # if isreal
   #     l_c2r = Lux.Parallel(nothing, [WrappedFunction(x -> identity.(real.(Ref(ctran(l1)) .* x .* Ref(ctran(l2)')))) for (l1,l2) in LLset]... )
   #     luxchain = append_layer(luxchain, l_c2r; l_name = :complex2real)
   # end

   if isreal
      l_real = WrappedFunction(cc -> real.(cc)) # WrappedFunction(cc -> Tuple([identity.(real.(cc[i])) for i = 1:length(cc) ]))
      luxchain = append_layer(luxchain, l_real; l_name = :stablize)
   end

   ps, st = Lux.setup(MersenneTwister(1234), luxchain)
   
   return luxchain, ps, st, LLset
end

equivariant_model_loc(spec_nlm, radial::Radial_basis, L::Int64; categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState = true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old) =
      equivariant_model_loc(spec_nlm, radial, L, L; categories, _get_cat, AA2BB, d, group, isState, isreal, tuned_filter, coupling_backend)
 
 # more constructors equivariant_model
equivariant_model_loc(totdeg::Int64, ν::Int64, radial::Radial_basis, L1::Int64, L2::Int64; categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState = true, isreal = true, cat_extension = simple_extension, tuned_filter = nothing, coupling_backend::Symbol = :old, fixed_particle_number::Bool = false) =
      equivariant_model_loc(degord2spec_loc(radial; totaldegree = totdeg, order = ν, Lmax = L1+L2, catagories = categories, filtered_extension = cat_extension, fixed_particle_number = fixed_particle_number)[2], radial, L1, L2; categories, _get_cat, AA2BB, d, group, isState, isreal, tuned_filter, coupling_backend)

 # With the _close function, the input could simply be an nnlllist (nlist,llist)

equivariant_model_loc(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L1::Int64, L2::Int64; categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState = true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old) = begin
    filter = RPE_filter_long(L1+L2)
    equivariant_model_loc(_close(nn, ll; filter = filter), radial, L1, L2; categories, _get_cat, AA2BB, d, group, isState, isreal, tuned_filter, coupling_backend)
end

equivariant_model_loc(totdeg::Int64, ν::Int64, radial::Radial_basis, L; categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState = true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old, fixed_particle_number::Bool = false) =
      equivariant_model_loc(totdeg, ν, radial, L, L; categories, _get_cat, AA2BB, d, group, isState, isreal, tuned_filter, coupling_backend, fixed_particle_number)

equivariant_model_loc(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L; categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState = true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old) =
      equivariant_model_loc(nn, ll, radial, L, L; categories, _get_cat, AA2BB, d, group, isState, isreal, tuned_filter, coupling_backend)

# extend_n_orbs(n_orbs, LLset) = [ n_orbs[l1+1] * n_orbs[l2+1] for (l1,l2) in LLset]
# extend_n_orbs(n_orbs) = extend_n_orbs(n_orbs, [(l1,l2) for l1 = 0:length(n_orbs)-1 for l2 = 0:length(n_orbs)-1])
extend_n_orbs(n_orbs1, n_orbs2) = [ n_orbs1[l1+1] * n_orbs2[l2+1] for l1 = 0:length(n_orbs1)-1, l2 = 0:length(n_orbs2)-1]
extend_n_orbs(n_orbs1, n_orbs2, LLset) = [ n_orbs1[l1+1] * n_orbs2[l2+1] for (l1,l2) in LLset]

function equivariant_operator(spec_nlm, radial::Radial_basis, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old)
    luxchain, ps, st, LLset = equivariant_model_loc(spec_nlm, radial, L1, L2; categories, _get_cat = _get_cat, AA2BB= AA2BB, d = d, group = group, isState = isState, isreal = isreal, tuned_filter = tuned_filter, coupling_backend = coupling_backend)
    @assert length(n_orbs1) == L1 + 1 && length(n_orbs2) == L2 + 1
    @assert length(LLset)  == (L1 + 1) * (L2 + 1)
    ext_n_orbs = extend_n_orbs(n_orbs1, n_orbs2, LLset)

    len = [size(luxchain.layers.AA2BB.layers[i].op,1) for i = 1:(L1+1)*(L2+1)]
    
    Linear_layer = Lux.Parallel(nothing, [LinearLayer_loc(len[i], ext_n_orbs[i]) for i = 1:(L1+1)*(L2+1)]... )
    luxchain = append_layer(luxchain, Linear_layer; l_name = :dot)

    ps, st = Lux.setup(MersenneTwister(1234), luxchain)

    return luxchain, ps, st
end

equivariant_operator(spec_nlm, radial::Radial_basis, L::Int64, n_orbs::Vector{Int64}=ones(Int64,L+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old) =
    equivariant_operator(spec_nlm, radial, L, L, n_orbs, n_orbs; categories = categories, _get_cat = _get_cat, AA2BB = AA2BB, d = d, group = group, isState = isState, isreal = isreal, tuned_filter = tuned_filter, coupling_backend = coupling_backend)

function equivariant_operator(totdeg::Union{Int64,Vector{Int64}}, ν::Int64, radial::Radial_basis, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, cat_extension = simple_extension, tuned_filter = nothing, coupling_backend::Symbol = :old, fixed_particle_number::Bool = false)
   equivariant_operator(degord2spec_loc(radial; totaldegree = totdeg, order = ν, Lmax = L1+L2, catagories = categories, filtered_extension = cat_extension, fixed_particle_number = fixed_particle_number)[2], radial, L1, L2, n_orbs1, n_orbs2; categories = categories, _get_cat = _get_cat, AA2BB = AA2BB, d = d, group = group, isState = isState, isreal = isreal, tuned_filter = tuned_filter, coupling_backend = coupling_backend)
end

equivariant_operator(totdeg::Union{Int64,Vector{Int64}}, ν::Int64, radial::Radial_basis, L::Int64, n_orbs::Vector{Int64}=ones(Int64,L+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old, fixed_particle_number::Bool = false) =
    equivariant_operator(totdeg, ν, radial, L, L, n_orbs, n_orbs; categories = categories, _get_cat = _get_cat, AA2BB = AA2BB, d = d, group = group, isState = isState, isreal = isreal, tuned_filter = tuned_filter, coupling_backend = coupling_backend, fixed_particle_number = fixed_particle_number)


function equivariant_operator(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L1::Int64, L2::Int64, n_orbs1::Vector{Int64}=ones(Int64,L1+1), n_orbs2::Vector{Int64}=ones(Int64,L2+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old)
    filter = RPE_filter_long(L1+L2)
    equivariant_operator(_close(nn, ll; filter = filter), radial, L1, L2,
                         n_orbs1, n_orbs2; categories = categories,
                         _get_cat = _get_cat, AA2BB = AA2BB, d = d,
                         group = group, isState = isState, isreal = isreal,
                         tuned_filter = tuned_filter,
                         coupling_backend = coupling_backend)
end

equivariant_operator(nn::Vector{Int64}, ll::Vector{Int64}, radial::Radial_basis, L::Int64, n_orbs::Vector{Int64}=ones(Int64,L+1); categories=[], _get_cat = _get_cat_default, AA2BB = nothing, d=3, group="O3", isState=true, isreal = true, tuned_filter = nothing, coupling_backend::Symbol = :old) =
    equivariant_operator(nn, ll, radial, L, L, n_orbs, n_orbs; categories = categories, _get_cat = _get_cat, AA2BB = AA2BB, d = d, group = group, isState = isState, isreal = isreal, tuned_filter = tuned_filter, coupling_backend = coupling_backend)
