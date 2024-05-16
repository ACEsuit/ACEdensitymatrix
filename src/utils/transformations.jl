using RepLieGroups, SparseArrays, LinearOperators, StaticArrays, LinearAlgebra

using RepLieGroups.O3: ClebschGordan, wigner_D_indices, adjoint, Rot3DCoeffs, dicttype, _key
import RepLieGroups.O3: _mrange
import EquivariantModels: coco_dot

# The transformation matrix from complex SHs to real SHs
function ctran(L)
    AA = spzeros(ComplexF64,2L+1, 2L+1)
    for i = 1:2L+1
        for j in [i, 2L+2-i]
            # @show i,j
            AA[i,j] = begin
                if i == j == L+1
                    1
                elseif i > L+1 && j > L+1
                    (-1)^(i-L-1)/sqrt(2)
                elseif i < L+1 && j < L+1
                    im/sqrt(2)
                elseif i < L+1 && j > L+1
                    (-1)^(i-L)/sqrt(2)*im
                elseif i > L+1 && j < L+1
                    1/sqrt(2)
                end
            end
            # @show AA[i,j]
        end
    end
    return AA
end

function flat(a)
    tmp = ones(length(a[1]),length(a))
    for i in 1:length(a)
        tmp[:,i] = vec(a[i])
    end
    return tmp
end

function k2ij(k, n, m)
    i = div(k-1, m) + 1
    j = k - (i-1)*m
    return i, j
end

cg = ClebschGordan(ComplexF64)

# The transformation matrix from a long equivariant vector to (L1, L2) equivariant tensors
function cgmatrix(L1, L2)
   cgm = zeros((2L1+1) * (2L2+1),(L1+L2+1)^2)
   for (i,(p,q)) in enumerate(collect(Iterators.product(-L1:L1, L2:-1:-L2)))
      ν = p+q
      for l = 0:L1+L2
         if abs(ν)<=l
            position = ν + l + 1
            cgm[i, l^2+position] = (-1)^q * cg(L1, p, L2, q, l, ν) 
            # cgm[i, l^2+position] = cg(L1,p,L2,q,l,ν) 
            # cgm[i, l^2+position] = (-1)^q * sqrt( (2L1+1) * (2L2+1) ) / 2 / sqrt(π * (2l+1)) * cg(L1,0,L2,0,l,0) * cg(L1,p,L2,q,l,ν) 
         end
      end
   end
   return sparse(cgm)
end

function _linear_operator(len_out, C, pos, len)
   T = SVector{len_out,ComplexF64} 
   fL = let C=C, idx=pos#, T=T
         (res, aa) -> begin
            res[:] .= C * aa[idx]
         end
   end
   return LinearOperator{T}(size(C,1), len, false, false, fL, nothing, nothing; S = Vector{T})
end

function _linear_operator_loc(L1, L2, C, pos, len)
    T = SMatrix{2L1+1,2L2+1,ComplexF64} 
    fL = let C=C, idx=pos#, T=T
        (res, aa) -> genmul!(res, C, aa[idx], *)
    end
    return LinearOperator{T}(size(C,1), len, false, false, fL, nothing, nothing; S = Vector{T})
 end

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## The above transformation somewhat does not work - add ad hoc code to construct equivariant tensors

struct Rot3DCoeffs_loc{L1, L2, T}
    vals::Vector{Dict}      # val[N] = coeffs for correlation order N
    cg::ClebschGordan{T}
 end

Rot3DCoeffs_loc(L1, L2, T=Float64) = Rot3DCoeffs_loc{L1, L2, T}(Dict[], ClebschGordan(T))

function mat_cou_coe(rotc::Rot3DCoeffs,
    l::Integer, m::Integer, μ::Integer,
    a::Integer, b::Integer, L1::Integer, L2::Integer)
    @assert (0 < a <= 2L1 + 1) && (0 < b <= 2L2 + 1)
    Z = zeros(ComplexF64, 2L1+1, 2L2+1)  # zeros(2 * L1 + 1, 2 * L2 + 1)
    Dp = wigner_D_indices(L1)'
    Dq = wigner_D_indices(L2)
    LL = SA[l, L1, L2]
    for i = 1:(2 * L1 + 1)
       for j = 1:(2 * L2 + 1)
          MM = SA[μ, Dp[i,a].m, Dq[b,j].m]
          KK = SA[m, Dp[i,a].μ, Dq[b,j].μ]
          cc = rotc(LL, MM, KK)
          Z[i,j] = Dp[i,a].sign * Dq[b,j].sign * cc
       end
    end
    return SMatrix{2L1+1,2L2+1}(Z)
end
 
 
function _select_ab(L1, L2, L, M, K)
    Dp = wigner_D_indices(L1)'
    Dq = wigner_D_indices(L2)
    list_ab = Tuple{Int, Int}[]
    for a = 1:2L1+1
       for b = 1:2L2+1
          # pm = prod( ma[i] + mb[j] + K for i = 1:2L1+1, j = 1:2L2+1)
          pm = prod( Dp[i,a].m + Dq[b,j].m + K for i = 1:2L1+1, j = 1:2L2+1)
          # pμ = prod(μa[i] + μb[j] + M for i = 1:2L1+1, j = 1:2L2+1)
          pμ = prod( Dp[i,a].μ + Dq[b,j].μ + M for i = 1:2L1+1, j = 1:2L2+1)
          if pμ == pm ==0
             push!(list_ab, (a,b))
          end
       end
    end
    return list_ab
end

coco_type(::Val{L1}, ::Val{L2}, T::Type{<: Number}) where {L1, L2} = SMatrix{2L1+1,2L2+1,T} 

coco_init(::Val{L1}, ::Val{L2}, T) where {L1, L2} = []

function coco_init(::Val{L1}, ::Val{L2}, T, l, m, μ) where {L1, L2} 
    if abs(m) <= L1+L2 && abs(μ) <= L1+L2
        return [ mat_cou_coe(Rot3DCoeffs(0), l, m, μ, a, b, L1, L2) for (a,b) in _select_ab(L1, L2, l, m, μ)]
    end
end

coco_zeros(::Val{L1}, ::Val{L2}, T, ll, mm, kk) where {L1, L2}  = [ SMatrix{2L1+1,2L2+1}(zeros(T,2L1+1,2L2+1)) for i = 1:length(_select_ab(L1, L2, sum(ll), sum(mm), sum(kk))) ]
# coco_zeros(φ::TP, ll, mm, kk, T, A) where{TP <: SphericalMatrix} =
#             zeros(TP, length(_select_ab(φ, sum(mm), sum(kk))))

coco_filter(::Val{L1}, ::Val{L2}, ll, mm) where {L1, L2} = iseven(sum(ll)+L1+L2) && abs(sum(mm)) <= L1+L2

coco_filter(::Val{L1}, ::Val{L2}, ll, mm, kk) where {L1, L2} = iseven(sum(ll)+L1+L2) && abs(sum(mm)) <= L1+L2 && abs(sum(kk)) <= L1+L2

coco_dot(u1::SMatrix{L1,L2,T}, u2::SMatrix{L1,L2,T}) where {L1,L2,T} = dot(u1, u2)
coco_dot(u1::SMatrix{L1,L2,T,L}, u2::SMatrix{L1,L2,T,L}) where {L1,L2,T,L} = dot(u1, u2)

Rot3DCoeffs_loc(L1, L2, T=Float64) = Rot3DCoeffs_loc{L1, L2, T}(Dict[], ClebschGordan(T))

_ValL1(::Rot3DCoeffs_loc{L1,L2,T}) where {L1,L2,T} = Val{L1}()
_ValL2(::Rot3DCoeffs_loc{L1,L2,T}) where {L1,L2,T} = Val{L2}() 

function get_vals(A::Rot3DCoeffs_loc{L1, L2, T}, valN::Val{N}) where {L1,L2,T,N}
	# make up an ll, kk, mm and compute a dummy coupling coeff
	ll, mm, kk = SVector(0), SVector(0), SVector(0)
	cc0 = coco_zeros(_ValL1(A), _ValL2(A), T, ll, mm, kk)
	TP = typeof(cc0)
	if length(A.vals) < N
		# create more dictionaries of the correct type
		for n = length(A.vals)+1:N
			push!(A.vals, dicttype(n, TP)())
		end
	end
   return (A.vals[N])::(dicttype(valN, TP))
end

function (A::Rot3DCoeffs_loc{L1, L2, T})(ll::StaticVector{N},
    mm::StaticVector{N},
    kk::StaticVector{N}) where {L1, L2, T, N}
    vals = get_vals(A, Val(N))  # this should infer the type!
    key = _key(ll, mm, kk)
    if haskey(vals, key)
        val  = vals[key]
    else
        val = _compute_val(A, key...)
        vals[key] = val
    end
    return val
end

# the recursion has two steps so we need to define the
# coupling coefficients for N = 1, 2
# TODO: actually this seems false; it is only one recursion step, and a bit
#       or reshuffling should allow us to get rid of the {N = 2} case.

(A::Rot3DCoeffs_loc{L1, L2, T})(ll::StaticVector{1},
                                mm::StaticVector{1},
                                kk::StaticVector{1}) where {L1, L2, T} =
                                coco_init(_ValL1(A), _ValL2(A), T, ll[1], mm[1], kk[1])

function _compute_val(A::Rot3DCoeffs_loc{L1, L2, T}, ll::StaticVector{N},
    mm::StaticVector{N},
    kk::StaticVector{N}) where {L1, L2, T, N}
    val = coco_zeros(_ValL1(A), _ValL2(A), T, ll, mm, kk)
    TV = typeof(val)

    tmp = zero(MVector{N-1, Int})

    function _get_pp(aa, ap)
        for i = 1:N-2
            @inbounds tmp[i] = aa[i]
        end
        tmp[N-1] = ap
        return SVector(tmp)
    end

    jmin = maximum( ( abs(ll[N-1]-ll[N]),
    abs(kk[N-1]+kk[N]),
    abs(mm[N-1]+mm[N]) ) )
    jmax = ll[N-1]+ll[N]
    for j = jmin:jmax
        cgk = A.cg(ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
        cgm = A.cg(ll[N-1], mm[N-1], ll[N], mm[N], j, mm[N-1]+mm[N])
        if cgk * cgm  != 0
            llpp = _get_pp(ll, j) # SVector(llp..., j)
            mmpp = _get_pp(mm, mm[N-1]+mm[N]) # SVector(mmp..., mm[N-1]+mm[N])
            kkpp = _get_pp(kk, kk[N-1]+kk[N]) # SVector(kkp..., kk[N-1]+kk[N])
            a = TV(A(llpp, mmpp, kkpp))::TV
            val += cgk * cgm * a
        end
    end
    return val
end

# ----------------------------------------------------------------------
#   construction of a possible set of generalised CG coefficient;
#   numerically via SVD; this could be done analytically which might
#   be more efficient.
# ----------------------------------------------------------------------


function re_basis(A::Rot3DCoeffs_loc{L1, L2, T}, ll::SVector) where {L1, L2, T}
    TCC = coco_type(_ValL1(A), _ValL2(A), T)
    CC, Mll = compute_Al(A, ll)  # CC::Vector{Vector{...}}
    G = [ sum( coco_dot(CC[a][i], CC[b][i]) for i = 1:length(Mll) ) for a = 1:length(CC), b = 1:length(CC) ]
    svdC = svd(G)
    rk = rank(Diagonal(svdC.S), rtol = 1e-7)
    # Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]' * CC
    # construct the new basis
    Ured = Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]'
    Ure = Matrix{TCC}(undef, rk, length(Mll))
    for i = 1:rk
        Ure[i, :] = sum(Ured[i, j] * CC[j]  for j = 1:length(CC))
    end
    return Ure, Mll
end

_mrange(::Val{L1}, ::Val{L2}, ll::SVector; args...) where {L1, L2} = RepLieGroups.O3._mrange(Val(L1+L2), ll; args...)

# function barrier
function compute_Al(A::Rot3DCoeffs_loc{L1, L2, T}, ll::SVector) where {L1, L2, T}
    # fil = mm -> abs(sum(mm)) <= L1+L2
    Mll = collect(_mrange(_ValL1(A), _ValL2(A), ll))#; mm_filter = fil))
    TP = coco_type(_ValL1(A), _ValL2(A), T)
    if length(Mll) == 0
        return Vector{TP}[], Mll
    end

    TA = typeof(A(ll, Mll[1], Mll[1]))
    return __compute_Al(A, ll, Mll, TP, TA)
end

# TODO: what was TA for? Can we get rid of it via coco_type? 

function __compute_Al(A::Rot3DCoeffs_loc{L1, L2, T}, ll, Mll, TP, TA) where {L1, L2, T}	
    fil = coco_filter

    lenMll = length(Mll)
    # each element of CC will be one row of the coupling coefficients
    TCC = coco_type(_ValL1(A), _ValL2(A), T)
    CC = Vector{TCC}[]
    # some utility funcions to allow coco_init to return either a property
    # or a vector of properties
    # function __into_cc!(cc, cc0, im)   # cc0: ::AbstractProperty
    #     @assert length(cc) == length(cc0)
    #     cc[1][im] = cc0
    # end
    # # NOTE: We won't have this in the current setting???
    function __into_cc!(cc, cc0::AbstractVector, im)
    	@assert length(cc) == length(cc0)
    	for p = 1:length(cc)
    		cc[p][im] = cc0[p]
    	end
    end

    for (ik, kk) in enumerate(Mll)  # loop over possible basis functions
        # do a dummy calculation to determine how many coefficients we will get
        cc0 = A(ll, Mll[1], kk)::TA
        @assert [ size(cc0[i]) == (2L1+1, 2L2+1) for i = 1:length(cc0) ] |> all
        numcc = length(cc0)
        cc = [ Vector{TCC}(undef, lenMll) for _=1:numcc ]
        for (im, mm) in enumerate(Mll) # loop over possible indices
            if !fil(_ValL1(A), _ValL2(A), ll, mm, kk)
                cc00 = zeros(TP, length(cc))#::TA
                __into_cc!(cc, cc00, im)
            else
            # get all possible coupling coefficients
                cc0 = A(ll, mm, kk)#::TA
                __into_cc!(cc, cc0, im)
            end
        end
        # and now push them onto the big stack.
        append!(CC, cc)
    end

    return CC, Mll
end

function svd_retraction(D::Matrix, n_pairs::Int)
    u, s, v = svd(D)
    s[1:n_pairs] .= 1.0
    s[n_pairs+1:end] .= 0.0
    retr_D = u * diagm(s) * v'
    @assert tr(retr_D) - n_pairs < 1e-10
    @assert norm(retr_D * retr_D - retr_D, Inf) < 1e-10
    @assert norm(retr_D' - retr_D, Inf) < 1e-10
    return retr_D
end

function eigen_retraction(D::Matrix, n_pairs::Int)
    @assert D ≈ D'
    s, q = eigen(D; sortby=x->-x)
    s[1:n_pairs] .= 1.0
    s[n_pairs+1:end] .= 0.0
    retr_D = q * diagm(s) * q'
    @assert tr(retr_D) - n_pairs < 1e-10
    @assert norm(retr_D * retr_D - retr_D, Inf) < 1e-10
    @assert norm(retr_D' - retr_D, Inf) < 1e-10
    return retr_D
end
