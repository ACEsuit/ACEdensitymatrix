module TestGEPICouplings

using ACEdensitymatrix
using Test

const ADM = ACEdensitymatrix

@testset "category-aware GE-PI couplings" begin
    cases = (
        (L1=1, L2=2, ll=(1, 2), nn=(0, 0), ss=(:C, :O)),
        (L1=2, L2=2, ll=(2, 2), nn=(0, 0), ss=(:C, :O)),
        (L1=2, L2=3, ll=(2, 3), nn=(0, 0), ss=(:C, :O)),
    )

    for case in cases
        for lambda in abs(case.L1 - case.L2):(case.L1 + case.L2)
            coefficients, magnetic_set =
                ADM.GEPICouplings.gepi_coupling_coeffs(
                    lambda, case.ll, case.nn, case.ss,
                )
            @test size(coefficients, 1) == 1
            @test size(coefficients, 2) == length(magnetic_set)
        end
    end

    # Equal categories form a symmetric square. For l=(2,2), only even
    # output channels remain, whereas distinct categories retain 0:4.
    same_category_rows = [
        size(ADM.GEPICouplings.gepi_coupling_coeffs(
            lambda, (2, 2), (0, 0), (:C, :C),
        )[1], 1)
        for lambda in 0:4
    ]
    distinct_category_rows = [
        size(ADM.GEPICouplings.gepi_coupling_coeffs(
            lambda, (2, 2), (0, 0), (:C, :O),
        )[1], 1)
        for lambda in 0:4
    ]
    @test same_category_rows == [1, 0, 1, 0, 1]
    @test distinct_category_rows == ones(Int, 5)
end

end
