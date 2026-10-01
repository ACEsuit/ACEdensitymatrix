module TestFittingSmoke

using ACEdensitymatrix
using ACEfit
using DecoratedParticles
using StaticArrays
using Test

function onsite_environment(distance)
    return [PState(
        rr=SVector(distance, 0.0, 0.0),
        Zi=1,
        Zj=1,
    )]
end

@testset "data-free fitting test" begin
    model = On_Model(
        2, 1, 4.0, 1, [1, 6, 8], 0, [2]; coupling_backend=:new,
    )
    environments = [
        onsite_environment(distance) for distance in (0.8, 1.0, 1.2, 1.4)
    ]
    targets = [
        [1.0 + 0.1i 0.05i; 0.05i 0.7 - 0.05i]
        for i in eachindex(environments)
    ]

    fitted = Ref{Any}()
    redirect_stdout(devnull) do
        fitted[] = fit!(
            model, environments, targets;
            solver=ACEfit.QR(), λ=1e-4, reg=:id,
        )
    end

    @test isfitted(fitted[])
    @test all(isfinite, fitted[].ps.dot[1].W)

    predictions = eval_model.(Ref(fitted[]), environments)
    @test all(size(prediction) == (2, 2) for prediction in predictions)
    @test all(all(isfinite, prediction) for prediction in predictions)
end

end
