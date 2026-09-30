module TestSparseEvaluation

using ACEdensitymatrix
using SparseArrays
using StaticArrays
using Test

const ADM = ACEdensitymatrix

@testset "concrete sparse static-matrix evaluation" begin
    matrix_values = SMatrix{2,2,Float64}[
        (@SMatrix [1.0 2.0; 3.0 4.0]),
        (@SMatrix [-1.0 0.5; 0.25 2.0]),
    ]
    operator = SparseMatrixCSC(
        2, 2, [1, 2, 3], [1, 2], matrix_values,
    )
    input = [2.0, -0.5]
    concrete_values = typeof(first(matrix_values)).(matrix_values)
    concrete_operator = SparseMatrixCSC(
        2, 2, [1, 2, 3], [1, 2], concrete_values,
    )
    reference = concrete_operator * input

    layer = ADM.ConstLinearLayer_loc(operator, [1, 2])
    @test isbitstype(eltype(layer.op))
    @test collect(layer(input)) == reference
end

end
