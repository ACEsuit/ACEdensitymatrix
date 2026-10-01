module TestCouplingTransformations

using ACEdensitymatrix
using ACEfit
using LinearAlgebra
using Random
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

    @testset "allocation-free feature assembly" begin
        values = [
            [1.0 2.0; 3.0 4.0],
            [-1.0 0.5; 0.25 2.0],
        ]
        reference = ADM.flat(values)
        destination = fill(NaN, size(reference, 1) + 2, size(reference, 2))

        ADM.flat!(destination, 2, values)

        @test destination[2:(end - 1), :] == reference
        @test all(isnan, destination[[1, end], :])
        @test_throws DimensionMismatch ADM.flat!(
            zeros(size(reference, 1), size(reference, 2) + 1), 1, values,
        )
    end

    @testset "batched QR matches repeated solves" begin
        rng = MersenneTwister(1234)
        data_matrix = randn(rng, 18, 6)
        regularization = 1e-4 * Matrix{Float64}(I, 6, 6)
        design_matrix = vcat(data_matrix, regularization)
        data_targets = randn(rng, 18, 5)
        targets = vcat(data_targets, zeros(6, 5))

        batched = qr!(copy(design_matrix)) \ targets
        repeated = hcat([
            ACEfit.solve(
                ACEfit.QR(), design_matrix, targets[:, column],
            )["C"]
            for column in axes(targets, 2)
        ]...)

        @test batched ≈ repeated rtol=1e-12 atol=1e-12
        @test data_matrix * batched ≈ data_matrix * repeated
    end
end

end
