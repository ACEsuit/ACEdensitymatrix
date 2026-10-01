module TestMWE

using ACEdensitymatrix
using ACEdensitymatrix.Database: close_traj
using ACEfit
using JLD2
using Test

@testset "reduced real-data MWE" begin
    degree = 6
    order = 1
    rcut = 4.0
    zcut = 10.0
    ao_dict = Dict(
        1 => Dict(
            "n_orbs" => [2], "maxdeg" => degree, "ord" => order,
            "rcut" => rcut, "zcut" => zcut,
        ),
        6 => Dict(
            "n_orbs" => [3, 2, 1], "maxdeg" => degree, "ord" => order,
            "rcut" => rcut, "zcut" => zcut,
        ),
        8 => Dict(
            "n_orbs" => [3, 2, 1], "maxdeg" => degree, "ord" => order,
            "rcut" => rcut, "zcut" => zcut,
        ),
    )

    model = Density_Model(ao_dict; coupling_backend=:new)
    train_trajectory = TrajectoryHDF5(
        joinpath(@__DIR__, "data", "propanol_mwe.h5"),
    )
    test_trajectory = TrajectoryHDF5(
        joinpath(@__DIR__, "data", "ethanol_mwe.h5"),
    )
    training_frames = [read_frame(train_trajectory, i) for i in 0:1]
    test_frame = read_frame(test_trajectory, 0)
    close_traj(train_trajectory)
    close_traj(test_trajectory)

    redirect_stdout(devnull) do
        fit!(
            model, training_frames;
            solver=ACEfit.QR(), λ=1e-4, reg=:smooth,
        )
    end

    converted = convert_frame(test_frame)
    electron_pairs = Int(sum(converted["atomic_numbers"]) / 2)
    direct_prediction = eval_model(
        model, converted["R"], converted["ao_labels"];
        retraction=D -> eigen_retraction(D, electron_pairs),
    )
    frame_prediction = eval_model(model, test_frame)

    @test size(frame_prediction) == size(converted["D"])
    @test all(isfinite, frame_prediction)
    @test frame_prediction == direct_prediction

    metrics = validate_model(model, [test_frame])
    @test all(isfinite, metrics)
    @test metrics[1] < 0.02
    @test metrics[4] < 0.2
    @test metrics[5] < 0.1

    mktempdir() do directory
        model_path = joinpath(directory, "model.jld2")
        save(model_path, write_dict(model))
        loaded_model = load(model_path) |> read_dict
        @test eval_model(loaded_model, test_frame) == frame_prediction
    end
end

end
