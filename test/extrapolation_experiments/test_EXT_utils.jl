module TestEXTUtils

using LinearAlgebra
using Random
using StaticArrays
using Test

include(joinpath(@__DIR__, "EXT_utils.jl"))

@testset "extrapolation utilities" begin
    @testset "Grassmann log-exp round trip" begin
        rng = MersenneTwister(1234)
        origin = Matrix(qr(randn(rng, 7, 3)).Q)[:, 1:3]
        target = Matrix(qr(origin + 0.1randn(rng, 7, 3)).Q)[:, 1:3]

        gamma = grassmann_log(target, origin)
        reconstructed = grassmann_exp(gamma, origin)

        @test origin' * gamma ≈ zeros(3, 3) atol=1e-12
        @test reconstructed' * reconstructed ≈ I(3) atol=1e-12
        @test reconstructed * reconstructed' ≈ target * target' atol=1e-12
    end

    @testset "regularized descriptor coefficients" begin
        train_vals = [1.0 2.0; 3.0 5.0; 7.0 11.0]
        ref_vals = [0.5, 1.0, 2.0]
        test_vals = [1.5, 4.0, 8.0]
        design = train_vals .- ref_vals
        target = test_vals .- ref_vals
        λ = 1e-8

        @test center_design(train_vals, ref_vals) == design
        @test regularized_gram(design; λ=λ) ≈ design' * design + λ * I
        expected = (design' * design + λ * I) \ (design' * target)
        @test fit_coeffs(train_vals, ref_vals, test_vals; λ=λ) ≈ expected rtol=1e-10
        @test fit_coeffs(train_vals, ref_vals, test_vals; λ=0.0) ≈ design \ target
        @test_throws ArgumentError fit_coeffs(train_vals, ref_vals, test_vals; λ=-1.0)
        @test_throws DimensionMismatch fit_coeffs(train_vals, ref_vals[1:2], test_vals)
    end

    @testset "AO block metadata" begin
        labels = [
            "1 C 2p 1",
            "1 C 2s 0",
            "1 C 2p -1",
            "1 C 3s 0",
            "1 C 2p 0",
            "0 H 1s 0",
        ]
        layout = ao_block_labels(labels)

        @test layout.atom_ids == [1, 2, 2, 2, 2, 2]
        @test layout.angular_momenta == [0, 0, 0, 1, 1, 1]
        @test layout.magnetic_indices == [0, 0, 0, -1, 0, 1]

        frame = Dict(
            "ao_labels" => labels,
            "atomic_numbers" => [1, 6],
        )
        @test infer_orbital_multiplicities(frame) == Dict(
            1 => [1],
            6 => [2, 1],
        )
        @test angular_orbital_range([2, 1], 0) == 1:2
        @test angular_orbital_range([2, 1], 1) == 3:5
    end

    @testset "matrix-valued basis packing" begin
        sample_1 = [
            @SMatrix([1.0 2.0; 3.0 4.0]),
            @SMatrix([5.0 6.0; 7.0 8.0]),
        ]
        sample_2 = [
            @SMatrix([9.0 10.0; 11.0 12.0]),
            @SMatrix([13.0 14.0; 15.0 16.0]),
        ]

        packed = pack_basis_evaluations([[sample_1], [sample_2]])

        @test length(packed) == 1
        @test size(only(packed)) == (8, 2)
        @test only(packed)[:, 1] == collect(reinterpret(Float64, sample_1))
        @test only(packed)[:, 2] == collect(reinterpret(Float64, sample_2))
    end
end

end
