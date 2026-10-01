module TestReorder

using ACEdensitymatrix
using Test

@testset "AO reordering" begin
    labels = [
        "1 C 2p 1",
        "1 C 2s 0",
        "1 C 2p -1",
        "1 C 3s 0",
        "1 C 2p 0",
        "0 H 1s 0",
    ]
    order = [6, 2, 4, 3, 5, 1]
    expected_labels = labels[order]

    reordered_labels, atom_ids, ls, ms =
        apply_reorder(labels; full_info=true)
    @test reordered_labels == expected_labels
    @test atom_ids == [0, 1, 1, 1, 1, 1]
    @test ls == [0, 0, 0, 1, 1, 1]
    @test ms == [0, 0, 0, -1, 0, 1]

    column_quantity = reshape(collect(1.0:12.0), 2, 6)
    reordered_columns = apply_reorder(labels, column_quantity)
    @test reordered_columns == column_quantity[:, order]
    @test apply_reorder(labels, reordered_columns; inverse=true) ==
          column_quantity

    row_quantity = transpose(column_quantity)
    @test apply_reorder(labels, row_quantity; orbital_dim=1) ==
          row_quantity[order, :]

    square_quantity = reshape(collect(1.0:36.0), 6, 6)
    @test apply_reorder(labels, square_quantity; bothsides=true) ==
          square_quantity[order, order]
end

end
