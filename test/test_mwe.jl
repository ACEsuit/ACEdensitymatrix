module TestMWE

using ACEdensitymatrix
using ACEdensitymatrix.Database: close_traj
using ACEfit
using JLD2
using LinearAlgebra
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

    threaded_model = Density_Model(ao_dict; coupling_backend=:new)
    previous_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        redirect_stdout(devnull) do
            fit!(
                model, training_frames;
                solver=ACEfit.QR(), λ=1e-4, reg=:smooth,
            )
            fit!(
                threaded_model, training_frames;
                solver=ACEfit.QR(), λ=1e-4, reg=:smooth,
                multi_thread=true,
            )
        end
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end

    @test all(
        isapprox(
            threaded_model.Models[key].ps.dot[i].W,
            model.Models[key].ps.dot[i].W;
            rtol=1e-11, atol=1e-12,
        ) for key in keys(model.Models)
          for i in eachindex(model.Models[key].ps.dot)
    )

    converted = convert_frame(test_frame)
    electron_pairs = Int(sum(converted["atomic_numbers"]) / 2)
    direct_prediction = eval_model(
        model, converted["R"], converted["ao_labels"];
        retraction=D -> eigen_retraction(D, electron_pairs),
    )
    frame_prediction = eval_model(model, test_frame)
    threaded_prediction = eval_model(threaded_model, test_frame)

    @test size(frame_prediction) == size(converted["D"])
    @test all(isfinite, frame_prediction)
    @test frame_prediction == direct_prediction
    @test isapprox(threaded_prediction, frame_prediction; rtol=1e-12, atol=1e-12)

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
