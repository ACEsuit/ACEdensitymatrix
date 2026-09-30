module GEPICouplings

using LinearAlgebra
using SparseArrays
using StaticArrays

export gepi_coupling_coeffs, categorical_channel_ids

# Adapted from constructing_GE-PI_basis_via_lie_alg.jl in the code archive
# https://doi.org/10.5281/zenodo.20331400.  The implementation is kept local
# so that ACEDensityMatrix can retain its Julia 1.9 environment.  In
# particular, `binomial` is provided by Base, so Combinatorics is not needed.

@inline function _fill_mm!(out, current, lset, nset, block, local_depth,
                           global_depth, min_val, idx)
    if global_depth > length(current)
        @inbounds out[idx] = SVector(current)
        return idx + 1
    end

    if local_depth > @inbounds nset[block]
        next_l = @inbounds lset[block + 1]
        return _fill_mm!(out, current, lset, nset, block + 1, 1,
                         global_depth, -next_l, idx)
    end

    l = @inbounds lset[block]
    for value in min_val:l
        @inbounds current[global_depth] = value
        idx = _fill_mm!(out, current, lset, nset, block, local_depth + 1,
                        global_depth + 1, value, idx)
    end
    return idx
end

function _all_mm_blocks(lset::AbstractVector{Int}, nset::AbstractVector{Int},
                        ::Val{N}) where {N}
    @assert length(lset) == length(nset)
    @assert sum(nset) == N

    total = prod(binomial(2 * l + n, n) for (l, n) in zip(lset, nset))
    out = Vector{SVector{N,Int}}(undef, total)
    current = MVector{N,Int}(undef)
    _fill_mm!(out, current, lset, nset, 1, 1, 1, -lset[1], 1)
    return out
end

struct _PermutableBlocks{N,T1,T2}
    nn::SVector{N,T1}
    ll::SVector{N,T2}
end

function Base.iterate(iter::_PermutableBlocks{N}, state=1) where {N}
    state > N && return nothing
    start_idx = state
    @inbounds for i in (start_idx + 1):N
        if iter.ll[i] != iter.ll[start_idx] || iter.nn[i] != iter.nn[start_idx]
            return (start_idx:(i - 1), i)
        end
    end
    return (start_idx:N, N + 1)
end

_permutable_blocks(nn::SVector{N,T1}, ll::SVector{N,T2}) where {N,T1,T2} =
    _PermutableBlocks(nn, ll)

"""
    categorical_channel_ids(nn, ss)

Encode the complete non-angular one-particle label `(n, s)` as an integer
channel index.  EquivariantTensors uses its integer `n` label in precisely
this role: permutations are allowed only inside blocks with identical
`(channel, l)` labels.  ACEDensityMatrix stores the categorical label `s`
separately, so it must be folded into the channel index before constructing
permutation-invariant coupling coefficients.
"""
function categorical_channel_ids(nn, ss)
    length(nn) == length(ss) ||
        throw(DimensionMismatch("nn and ss must have the same length"))
    channel_map = Dict{Any,Int}()
    next_channel = 1
    return [get!(channel_map, (nn[i], ss[i])) do
                channel = next_channel
                next_channel += 1
                channel
            end for i in eachindex(nn)]
end

function _block_multiplicities(nn::SVector{N,Int}, ll::SVector{N,Int}) where {N}
    lset = Int[]
    nset = Int[]
    for block in _permutable_blocks(nn, ll)
        push!(lset, ll[first(block)])
        push!(nset, length(block))
    end
    return lset, nset
end

# Derivative with respect to the second Euler angle at the identity.
_db(l::Int, m::Int, mu::Int) =
    m - mu == 1  ? -0.5 * sqrt((l - mu) * (l + m)) :
    m - mu == -1 ?  0.5 * sqrt((l + mu) * (l - m)) : 0.0

function _lie_matrix(K::Int, ll::SVector{N,Int}, nn::SVector{N,Int}) where {N}
    lset, nset = _block_multiplicities(nn, ll)
    all_mm = _all_mm_blocks(lset, nset, Val(N))

    muset = all_mm[findall(x -> abs(sum(x)) <= K, all_mm)]
    mmset = if K != 0
        all_mm[findall(x -> (-K <= sum(x) <= K - 1) || sum(x) == K + 1,
                           all_mm)]
    else
        all_mm[findall(x -> sum(x) == K + 1, all_mm)]
    end

    sort!(muset, by=sum)
    sort!(mmset, by=sum)
    mu_index = Dict{SVector{N,Int},Int}(mu => i for (i, mu) in enumerate(muset))

    row_idx = Int[]
    col_idx = Int[]
    values = Float64[]
    ncols = 0

    for mm in mmset
        S = sum(mm)
        if -K <= S <= K - 1
            ncols += 1
            I = S + 1
            push!(row_idx, mu_index[mm])
            push!(col_idx, ncols)
            push!(values, -_db(K, I - 1, I))

            block_end = 0
            for (l, n) in zip(lset, nset)
                block_start = block_end + 1
                block_end += n
                for i in block_end:-1:block_start
                    m = mm[i]
                    if m < l && (i == block_end || m < mm[i + 1])
                        mu = setindex(mm, m + 1, i)
                        multiplicity = count(==(mu[i]), view(mu, block_start:block_end))
                        push!(row_idx, mu_index[mu])
                        push!(col_idx, ncols)
                        push!(values, multiplicity * _db(l, mu[i] - 1, mu[i]))
                    end
                end
            end
        elseif S == K + 1
            ncols += 1
            block_end = 0
            for (l, n) in zip(lset, nset)
                block_start = block_end + 1
                block_end += n
                for i in block_start:block_end
                    m = mm[i]
                    if m > -l && (i == block_start || m > mm[i - 1])
                        mu = setindex(mm, m - 1, i)
                        multiplicity = count(==(mu[i]), view(mu, block_start:block_end))
                        push!(row_idx, mu_index[mu])
                        push!(col_idx, ncols)
                        push!(values, multiplicity * _db(l, mu[i] + 1, mu[i]))
                    end
                end
            end
        end
    end

    @assert ncols == length(mmset)
    return sparse(row_idx, col_idx, values, length(muset), ncols), muset, mmset
end

function _truncate_zeros!(C::AbstractMatrix, tol::Real=1e-12)
    @inbounds @simd for i in eachindex(C)
        abs(C[i]) < tol && (C[i] = zero(eltype(C)))
    end
    return C
end

function _nullspace_upper_sparse(U::AbstractMatrix{T}) where {T<:Number}
    m, n = size(U)
    @assert m <= n
    nullity = n - m
    nullity == 0 && return zeros(T, n, 0)

    rhs = -Matrix(U[1:m, (m + 1):n])
    X = UpperTriangular(U[1:m, 1:m]) \ rhs
    basis = zeros(T, n, nullity)
    basis[1:m, :] .= X
    @inbounds for j in 1:nullity
        basis[m + j, j] = one(T)
    end
    return basis ./ norm(basis)
end

function _solve_kernel(M::AbstractMatrix{T}, mmset::Vector{SVector{N,Int}},
                       muset::Vector{SVector{N,Int}}) where {N,T<:Number}
    M = sparse(transpose(M))
    C = zeros(Float64, size(M, 2) - size(M, 1), size(M, 2))

    row_sum = sum.(mmset)
    column_sum = sum.(muset)
    row_range = findall(i -> i == 1 || row_sum[i] != row_sum[i - 1],
                        eachindex(mmset))
    column_range = findall(i -> i == 1 || column_sum[i] != column_sum[i - 1],
                           eachindex(muset))
    push!(row_range, length(mmset) + 1)
    push!(column_range, length(muset) + 1)

    row_block = row_range[end - 1]:(row_range[end] - 1)
    previous_columns = column_range[end - 1]:(column_range[end] - 1)

    if length(row_range) == length(column_range)
        B = M[row_block, previous_columns]
        F = lu(transpose(B))
        invp = invperm(F.p)
        sparse_ns = _nullspace_upper_sparse(sparse(transpose(F.L)))
        C[:, previous_columns] .=
            (Diagonal(F.Rs) * sparse_ns[invp, :])'
        start_block = length(row_range) - 2
    else
        for (i, column) in enumerate(previous_columns)
            C[i, column] = 1.0
        end
        start_block = length(row_range) - 1
    end

    for block in start_block:-1:1
        row_block = row_range[block]:(row_range[block + 1] - 1)
        current_columns = column_range[block]:(column_range[block + 1] - 1)
        scale = -1 / M[first(row_block), first(current_columns)]

        for (j_local, j_global) in enumerate(previous_columns)
            for p in M.colptr[j_global]:(M.colptr[j_global + 1] - 1)
                i_global = M.rowval[p]
                if first(row_block) <= i_global <= last(row_block)
                    i_local = i_global - first(row_block) + 1
                    value = M.nzval[p] * scale
                    for k in axes(C, 1)
                        C[k, current_columns[i_local]] +=
                            value * C[k, previous_columns[j_local]]
                    end
                end
            end
        end
        previous_columns = current_columns
    end

    return _truncate_zeros!(cholesky(Symmetric(C * C')).L \ C)
end

function _embed_onehot(C::AbstractMatrix{T}, mmset::AbstractVector,
                       ::Val{K}) where {T,K}
    rows, columns = size(C)
    Vec = SVector{2 * K + 1,T}
    result = Matrix{Vec}(undef, rows, columns)
    zero_vec = zero(Vec)
    for j in 1:columns
        component = sum(mmset[j]) + K + 1
        for i in 1:rows
            result[i, j] = setindex(zero_vec, C[i, j], component)
        end
    end
    return result
end

function _empty_coefficients(K::Int, ll::SVector{N,Int},
                             nn::SVector{N,Int}) where {N}
    lset, nset = _block_multiplicities(nn, ll)
    all_mm = _all_mm_blocks(lset, nset, Val(N))
    muset = all_mm[findall(x -> abs(sum(x)) <= K, all_mm)]
    sort!(muset, by=sum)
    T = K == 0 ? Float64 : SVector{2 * K + 1,Float64}
    return Matrix{T}(undef, 0, length(muset)), muset
end

"""
    gepi_coupling_coeffs(K, ll, nn)

Construct the permutation-invariant, `K`-equivariant coupling coefficients by
the GEPI Lie-algebra kernel/back-substitution method.  Rows enumerate independent
couplings and columns correspond to the returned ordered magnetic tuples.
"""
function gepi_coupling_coeffs(K::Integer, ll, nn)
    @assert K >= 0
    @assert length(ll) == length(nn)
    @assert all(value -> value isa Integer, ll)
    @assert all(value -> value isa Integer, nn)
    N = length(ll)
    ll_static_input = SVector{N,Int}(ll)
    nn_static_input = SVector{N,Int}(nn)

    # EquivariantTensors' GE-PI kernel assumes lexicographic ordering by the
    # complete `(l, channel)` label.  Sort internally and restore the caller's
    # factor order in the returned magnetic tuples.
    permutation = Vector(sortperm(collect(zip(ll_static_input, nn_static_input))))
    inverse_permutation = invperm(permutation)
    ll_static = SVector{N,Int}(ll_static_input[permutation])
    nn_static = SVector{N,Int}(nn_static_input[permutation])

    if all(iszero, ll_static) && K == 0
        return [1.0;;], [zero(SVector{N,Int})]
    end

    if K > sum(ll_static)
        coefficients, muset =
            _empty_coefficients(Int(K), ll_static, nn_static)
        return coefficients, [mu[inverse_permutation] for mu in muset]
    end

    M, muset, mmset = _lie_matrix(Int(K), ll_static, nn_static)
    if size(M, 1) == size(M, 2)
        coefficients, empty_muset =
            _empty_coefficients(Int(K), ll_static, nn_static)
        return coefficients,
               [mu[inverse_permutation] for mu in empty_muset]
    end

    C = _solve_kernel(M, mmset, muset)
    coefficients = K == 0 ? C : _embed_onehot(C, muset, Val(Int(K)))
    return coefficients, [mu[inverse_permutation] for mu in muset]
end

"""
    gepi_coupling_coeffs(K, ll, nn, ss)

Category-aware GE-PI coupling coefficients.  Two factors are permutable only
when their radial labels, categorical labels, and angular momenta are all
identical.  This mirrors the EquivariantTensors convention in which the
one-particle channel index contains every non-magnetic label.
"""
function gepi_coupling_coeffs(K::Integer, ll, nn, ss)
    return gepi_coupling_coeffs(K, ll, categorical_channel_ids(nn, ss))
end

end # module GEPICouplings
