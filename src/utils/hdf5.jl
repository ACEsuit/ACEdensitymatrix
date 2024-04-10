using HDF5

mutable struct TrajectoryHDF5
    filename::String
    file::HDF5.File

    function TrajectoryHDF5(filename::String)
        file = h5open(filename, "r")
        new(filename, file)
    end
end

function read_frame(traj::TrajectoryHDF5, frame)
    matrices = Dict{String, Array}()
    group = traj.file["frame_$frame"]
    for obj in group
        key = split(HDF5.name(obj), "/")[end]
        matrices[key] = read(obj)
    end
    return matrices
end

function read_info(traj::TrajectoryHDF5)
    info = Dict{String, Any}()
    for key in keys(attributes(traj.file))
        info[key] = read_attribute(traj.file, key)
    end
    info
end

function close_traj(traj::TrajectoryHDF5)
    close(traj.file)
end

