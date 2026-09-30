module TestBackendEquivalence

using ACEdensitymatrix
using LinearAlgebra
using StaticArrays
using Test

const ADM = ACEdensitymatrix
const RANK_RTOL = 1e-8

function category_distinct_spec(case)
    spec = [
        sort([
            (n=case.nn[1], l=case.ll[1], m=m1, s=case.ss[1]),
            (n=case.nn[2], l=case.ll[2], m=m2, s=case.ss[2]),
        ])
        for m1 in -case.ll[1]:case.ll[1]
        for m2 in -case.ll[2]:case.ll[2]
    ]
    return unique(spec)
end

function clebsch_gordan_reference(case, spec)
    matrix_type = SMatrix{
        2case.L1 + 1,
        2case.L2 + 1,
        ComplexF64,
    }
    lambdas = abs(case.L1 - case.L2):(case.L1 + case.L2)
    result = fill(zero(matrix_type), length(lambdas), length(spec))
    coupling = ADM.BlockCoupling(case.L1, case.L2)

    for (column, pib) in enumerate(spec)
        mm = ADM._canonical_magnetic_tuple(pib)
        for (row, lambda) in enumerate(lambdas)
            mu = sum(mm)
            abs(mu) > lambda && continue
            coefficient = ADM.cg(
                case.ll[1], mm[1], case.ll[2], mm[2], lambda, mu,
            )
            iszero(coefficient) && continue
            complex_vector = setindex(
                zero(SVector{2lambda + 1,Float64}),
                coefficient,
                mu + lambda + 1,
            )
            real_vector = ADM.complex_to_real_vector(lambda, complex_vector)
            result[row, column] = matrix_type(
                ADM.transform_λ(coupling, lambda, real_vector),
            )
        end
    end
    return result
end

function coefficient_gramian(coefficients)
    rows, columns = size(coefficients)
    gramian = zeros(ComplexF64, rows, rows)
    for i in 1:rows, j in i:rows
        value = sum(dot(coefficients[i, column], coefficients[j, column])
                    for column in 1:columns)
        gramian[i, j] = value
        gramian[j, i] = conj(value)
    end
    return Matrix(Hermitian(gramian))
end

@testset "old and new coupling spaces" begin
    cases = (
        (L1=1, L2=2, nn=(0, 0), ll=(1, 2), ss=(:C, :O)),
        (L1=2, L2=2, nn=(0, 0), ll=(2, 2), ss=(:C, :O)),
        (L1=2, L2=3, nn=(0, 0), ll=(2, 3), ss=(:C, :O)),
    )

    for case in cases
        spec = category_distinct_spec(case)
        reference = clebsch_gordan_reference(case, spec)
        gepi = Matrix(ADM._gepi_A2B_matrix_real(
            ADM.Rot3DCoeffs_loc(case.L1, case.L2), spec,
        ))

        @test size(reference) == size(gepi)
        @test size(first(reference)) == size(first(gepi))

        reference_rank = rank(coefficient_gramian(reference); rtol=RANK_RTOL)
        gepi_rank = rank(coefficient_gramian(gepi); rtol=RANK_RTOL)
        stacked_rank = rank(
            coefficient_gramian(vcat(reference, gepi)); rtol=RANK_RTOL,
        )
        @test reference_rank == gepi_rank == stacked_rank
    end
end

end
