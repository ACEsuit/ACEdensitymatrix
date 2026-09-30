# Coupling transformations used to compare the legacy matrix-valued
# construction with the vector-valued GEPI construction.

# This follows EquivariantTensors.O3.Ctran:
# https://github.com/ACEsuit/EquivariantTensors.jl/blob/main/src/O3/O3_utils.jl
function _Ctran_entry(m::Integer, mu::Integer, ::Type{T}=ComplexF64) where {T}
    abs(m) == abs(mu) || return zero(T)
    m == mu == 0 && return one(T)
    m > 0 && mu > 0 && return T((-1)^m / sqrt(2))
    m < 0 && mu < 0 && return T(im / sqrt(2))
    m < 0 && mu > 0 && return T((-1)^(m + 1) * im / sqrt(2))
    @assert m > 0 && mu < 0
    return T(1 / sqrt(2))
end

"""
    Ctran(L)

Sparse complex-to-real spherical-harmonic transformation in the SpheriCart
ordering.  For component vectors, `v_real = Ctran(L) * v_complex`.
"""
Ctran(L::Integer) = sparse([
    _Ctran_entry(m, mu) for m in -L:L, mu in -L:L
])

"""
    complex_to_real_vector(L, vector)

Transform the output irrep of complex-basis GEPI coefficients to the real
spherical-harmonic convention used for ACEDensityMatrix output blocks.  The
input-product columns remain complex because `xx2AA(...; rSH=false)` constructs
the package's one-particle basis from complex spherical harmonics.
"""
function complex_to_real_vector(L::Integer, vector::AbstractVector)
    @assert length(vector) == 2L + 1
    return Ctran(L) * vector
end

channel_parity(l::Integer, lp::Integer, lambda::Integer) =
    iseven(l + lp + lambda) ? :even : :odd

"""
    cg_block([T=Float64], l, lp, lambda)

Construct the sparse real Clebsch-Gordan map whose transpose maps a real
`lambda`-vector to a `(2l+1) x (2lp+1)` matrix block.  The convention and
`transform_λ` name follow ACEoperators.jl:
https://github.com/ACEsuit/ACEoperators.jl/blob/main/src/linear2c/coupling.jl
"""
function cg_block(::Type{T}, l::Integer, lp::Integer,
                  lambda::Integer) where {T<:Real}
    block = zeros(T, 2lambda + 1, (2l + 1) * (2lp + 1))
    abs(l - lp) <= lambda <= l + lp || return sparse(block)

    Tl = Ctran(l)
    Tlp = Ctran(lp)
    Tlambda = Ctran(lambda)
    even_channel = channel_parity(l, lp, lambda) == :even

    for nu in -lambda:lambda, a in -l:l, b in -lp:lp
        value = 0.0im
        for m in -l:l, mp in -lp:lp
            mu = m + mp
            abs(mu) <= lambda || continue
            coefficient = cg(l, m, lp, mp, lambda, mu)
            iszero(coefficient) && continue
            value += conj(Tlambda[nu + lambda + 1, mu + lambda + 1]) *
                     Tl[a + l + 1, m + l + 1] *
                     Tlp[b + lp + 1, mp + lp + 1] * coefficient
        end
        matrix_column = (b + lp) * (2l + 1) + (a + l + 1)
        block[nu + lambda + 1, matrix_column] =
            even_channel ? T(real(value)) : T(imag(value))
    end
    return sparse(block)
end

cg_block(l::Integer, lp::Integer, lambda::Integer) =
    cg_block(Float64, l, lp, lambda)

struct BlockCoupling{T}
    l::Int
    lp::Int
    lambdas::UnitRange{Int}
    maps::Vector{SparseMatrixCSC{T,Int}}
end

function BlockCoupling(::Type{T}, l::Integer, lp::Integer) where {T<:Real}
    lambdas = abs(l - lp):(l + lp)
    maps = [cg_block(T, l, lp, lambda) for lambda in lambdas]
    return BlockCoupling{T}(Int(l), Int(lp), lambdas, maps)
end

BlockCoupling(l::Integer, lp::Integer) = BlockCoupling(Float64, l, lp)

function _lambda_index(coupling::BlockCoupling, lambda::Integer)
    index = Int(lambda) - first(coupling.lambdas) + 1
    1 <= index <= length(coupling.lambdas) ||
        error("lambda = $lambda is not admissible for " *
              "(l, lp) = ($(coupling.l), $(coupling.lp))")
    return index
end

"""
    transform_λ(coupling, lambda, vector)

Map one real-spherical `lambda` channel to its contribution to the matrix block.
The multiplication uses the sparse linear map stored by `BlockCoupling`.
"""
function transform_λ(coupling::BlockCoupling, lambda::Integer,
                     vector::AbstractVector)
    @assert length(vector) == 2lambda + 1
    map = coupling.maps[_lambda_index(coupling, lambda)]
    matrix_vector = transpose(map) * vector
    return reshape(matrix_vector, 2coupling.l + 1, 2coupling.lp + 1)
end
