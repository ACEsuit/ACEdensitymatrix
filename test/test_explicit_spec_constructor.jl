module TestExplicitSpecConstructor

using ACEdensitymatrix
using Test

const ADM = ACEdensitymatrix

function coupling_maps(model)
    return [layer.op for layer in model.layers.AA2BB.layers]
end

function coupling_positions(model)
    return [layer.pos for layer in model.layers.AA2BB.layers]
end

function output_dimensions(model)
    return [(layer.in_dim, layer.out_dim) for layer in model.layers.dot.layers]
end

@testset "explicit nn,ll operator constructors" begin
    radial = ADM.onsite_radial_basis(2, 4.0)
    nn = [0, 0]
    ll = [1, 2]

    @testset "rectangular output" begin
        L1, L2 = 1, 2
        n_orbs1 = [2, 1]
        n_orbs2 = [1, 2, 1]
        filter = ADM.RPE_filter_long(L1 + L2)
        spec = ADM._close(nn, ll; filter=filter)

        explicit_model, _, _ = ADM.equivariant_operator(
            nn, ll, radial, L1, L2, n_orbs1, n_orbs2;
            coupling_backend=:new,
        )
        reference_model, _, _ = ADM.equivariant_operator(
            spec, radial, L1, L2, n_orbs1, n_orbs2;
            coupling_backend=:new,
        )

        @test length(explicit_model.layers.AA2BB.layers) ==
              (L1 + 1) * (L2 + 1)
        @test coupling_positions(explicit_model) ==
              coupling_positions(reference_model)
        @test coupling_maps(explicit_model) == coupling_maps(reference_model)
        @test output_dimensions(explicit_model) ==
              output_dimensions(reference_model)
    end

    @testset "single-L wrapper" begin
        L = 2
        n_orbs = [1, 2, 1]
        filter = ADM.RPE_filter_long(2L)
        spec = ADM._close(nn, ll; filter=filter)

        wrapped_model, _, _ = ADM.equivariant_operator(
            nn, ll, radial, L, n_orbs; coupling_backend=:new,
        )
        reference_model, _, _ = ADM.equivariant_operator(
            spec, radial, L, L, n_orbs, n_orbs;
            coupling_backend=:new,
        )

        @test coupling_positions(wrapped_model) ==
              coupling_positions(reference_model)
        @test coupling_maps(wrapped_model) == coupling_maps(reference_model)
    end

    @testset "degree/order constructors use the same full specification" begin
        degree, order = 2, 2
        L1, L2 = 1, 2
        n_orbs1 = [2, 1]
        n_orbs2 = [1, 2, 1]

        basis_model, _, _, outputs = ADM.equivariant_model_loc(
            degree, order, radial, L1, L2; coupling_backend=:new,
        )
        operator_model, _, _ = ADM.equivariant_operator(
            degree, order, radial, L1, L2, n_orbs1, n_orbs2;
            coupling_backend=:new,
        )

        @test outputs == [(l1, l2) for l1 in 0:L1 for l2 in 0:L2]
        @test coupling_positions(basis_model) ==
              coupling_positions(operator_model)
        @test coupling_maps(basis_model) == coupling_maps(operator_model)
    end
end

end
