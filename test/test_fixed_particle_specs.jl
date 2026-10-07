module TestFixedParticleSpecs

using ACEdensitymatrix
using Test

const ADM = ACEdensitymatrix

spec_key(bb) = Tuple(bb)
without_index(bb, i) = Tuple(bb[j] for j in eachindex(bb) if j != i)

function removable_zero(bb, available)
    for i in eachindex(bb)
        if ADM._is_zero_channel(bb[i]) && without_index(bb, i) in available
            return i, without_index(bb, i)
        end
    end
    return nothing
end

function test_fixed_count_redundancy(full, reduced, counts)
    full_by_key = Dict(spec_key(bb) => bb for bb in full)
    full_keys = Set(keys(full_by_key))
    reduced_keys = Set(spec_key(bb) for bb in reduced)
    removed = [bb for bb in full if spec_key(bb) ∉ reduced_keys]

    @test !isempty(removed)
    @test reduced_keys == Set(spec_key(bb) for bb in ADM._remove_redundant_zero_padding(full))
    @test all(isnothing(removable_zero(bb, full_keys)) for bb in reduced)

    channel_values = Dict{Any, ComplexF64}()
    next_value = 1
    for bb in full, b in bb
        haskey(channel_values, b) && continue
        if ADM._is_zero_channel(b)
            channel_values[b] = counts[b.s]
        else
            channel_values[b] = complex(next_value, next_value / 7)
            next_value += 1
        end
    end
    feature(bb) = prod(channel_values[b] for b in bb)

    missing_relations = [
        bb for bb in removed if isnothing(removable_zero(bb, full_keys))
    ]
    @test isempty(missing_relations)

    failed_identities = Any[]
    for bb in removed
        relation = removable_zero(bb, full_keys)
        isnothing(relation) && continue
        i, lower_key = relation
        lhs = feature(bb)
        rhs = counts[bb[i].s] * feature(full_by_key[lower_key])
        isapprox(lhs, rhs) || push!(failed_identities,
            (spec=bb, lower=full_by_key[lower_key], lhs=lhs, rhs=rhs))
    end
    @test isempty(failed_identities)
end

@testset "fixed-particle AA specifications" begin
    degree = 3
    order = 3

    @testset "default path is unchanged" begin
        radial = ADM.onsite_radial_basis(degree, 4.0)
        categories = [(6, 1), (6, 6)]
        default = ADM.degord2spec_loc(
            radial; totaldegree=degree, order=order, Lmax=2,
            catagories=categories,
        )
        explicit_false = ADM.degord2spec_loc(
            radial; totaldegree=degree, order=order, Lmax=2,
            catagories=categories, fixed_particle_number=false,
        )
        @test default == explicit_false

        model_default, ps_default, _ = ADM.equivariant_operator(
            2, 2, radial, 1, [1, 1]; categories=categories,
            coupling_backend=:new,
        )
        model_false, ps_false, _ = ADM.equivariant_operator(
            2, 2, radial, 1, [1, 1]; categories=categories,
            coupling_backend=:new, fixed_particle_number=false,
        )
        @test model_default.layers.A.basis.spec == model_false.layers.A.basis.spec
        @test model_default.layers.AA.basis.specs == model_false.layers.AA.basis.specs
        @test ps_default == ps_false
        for (default_layer, false_layer) in zip(
            model_default.layers.AA2BB.layers, model_false.layers.AA2BB.layers,
        )
            @test default_layer.pos == false_layer.pos
            @test default_layer.op == false_layer.op
        end
    end

    @testset "onsite zero padding is redundant at fixed counts" begin
        radial = ADM.onsite_radial_basis(degree, 4.0)
        categories = [(6, 1), (6, 6)]
        _, full = ADM.degord2spec_loc(
            radial; totaldegree=degree, order=order, Lmax=2,
            catagories=categories,
        )
        _, reduced = ADM.degord2spec_loc(
            radial; totaldegree=degree, order=order, Lmax=2,
            catagories=categories, fixed_particle_number=true,
        )

        @test length(reduced) < length(full)
        test_fixed_count_redundancy(
            full, reduced, Dict((6, 1) => 4.0, (6, 6) => 2.0),
        )
    end

    @testset "offsite bond anchors are retained" begin
        radial = ADM.offsite_radial_basis(degree, 4.0, 10.0)
        categories = [
            (6, 8, 8, true),
            (6, 8, 1, false),
            (6, 8, 6, false),
            (6, 8, 8, false),
        ]
        spec_args = (
            totaldegree=degree,
            order=order,
            Lmax=2,
            catagories=categories,
            filtered_extension=ADM.ModelConstruction.offsite_extension,
        )
        _, full = ADM.degord2spec_loc(radial; spec_args...)
        _, reduced = ADM.degord2spec_loc(
            radial; spec_args..., fixed_particle_number=true,
        )

        @test length(reduced) < length(full)
        @test all(sum(b.s[4] for b in bb) == 1 for bb in reduced)
        @test any(reduced) do bb
            any(ADM._is_zero_channel(b) && b.s[4] for b in bb) &&
                isnothing(removable_zero(bb, Set(spec_key.(full))))
        end
        test_fixed_count_redundancy(
            full,
            reduced,
            Dict(
                (6, 8, 8, true) => 1.0,
                (6, 8, 1, false) => 4.0,
                (6, 8, 6, false) => 2.0,
                (6, 8, 8, false) => 1.0,
            ),
        )
    end

    @testset "Density_Model propagates the flag" begin
        ao_dict = Dict(
            1 => Dict(
                "n_orbs" => [1], "maxdeg" => 2, "ord" => 2,
                "rcut" => 4.0, "zcut" => 10.0,
            ),
            6 => Dict(
                "n_orbs" => [1], "maxdeg" => 2, "ord" => 2,
                "rcut" => 4.0, "zcut" => 10.0,
            ),
        )
        full = Density_Model(ao_dict; coupling_backend=:new)
        reduced = Density_Model(
            ao_dict; coupling_backend=:new, fixed_particle_number=true,
        )

        for key in keys(full.Models)
            full_count = sum(length, full.Models[key].model.layers.AA.basis.specs)
            reduced_count = sum(length, reduced.Models[key].model.layers.AA.basis.specs)
            @test reduced_count < full_count
            for layer in reduced.Models[key].model.layers.AA2BB.layers
                @test isempty(layer.pos) || maximum(layer.pos) <= reduced_count
            end
        end
    end
end

end
