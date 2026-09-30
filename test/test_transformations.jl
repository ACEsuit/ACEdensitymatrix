module TestCouplingTransformations

using ACEdensitymatrix
using LinearAlgebra
using SparseArrays
using Test

const ADM = ACEdensitymatrix

@testset "coupling transformations" begin
    @testset "complex-to-real convention" begin
        for L in 0:5
            @test Matrix(ADM.Ctran(L)) ≈ Matrix(ADM.ctran(L))
            @test Matrix(ADM.Ctran(L) * ADM.Ctran(L)') ≈
                  Matrix{ComplexF64}(I, 2L + 1, 2L + 1)
        end
    end

    @testset "vector-to-matrix maps" begin
        for (L1, L2) in ((1, 2), (2, 2), (2, 3))
            coupling = ADM.BlockCoupling(L1, L2)
            @test coupling.lambdas == abs(L1 - L2):(L1 + L2)
            for (index, lambda) in enumerate(coupling.lambdas)
                map = coupling.maps[index]
                @test issparse(map)
                @test size(map) == (2lambda + 1,
                                    (2L1 + 1) * (2L2 + 1))
                @test Matrix(map * transpose(map)) ≈
                      Matrix{Float64}(I, 2lambda + 1, 2lambda + 1)
            end
        end
    end
end

end
