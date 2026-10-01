using Test

@testset "ACEdensitymatrix" begin
    include("test_transformations.jl")
    include("test_gepi_couplings.jl")
    include("test_backend_equivalence.jl")
    include("test_sparse_evaluation.jl")
    include("test_explicit_spec_constructor.jl")
    include("test_mwe.jl")
    include("test_reorder.jl")
end
