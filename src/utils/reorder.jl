function unpack(ao_labels::Union{Vector{String},Matrix{String}})
    atom_ids = Int64[]
    shells = Int64[]
    ls = Int64[]
    ms = Int64[]
    letter_to_l = Dict{String,Int64}("s" => 0, "p" => 1, "d" => 2, "f" => 3)

    for label in ao_labels
        atom_id, _, orbital, m = split(label)
        push!(atom_ids, parse(Int64, atom_id))
        push!(shells, parse(Int64, orbital[1:1]))
        push!(ls, letter_to_l[orbital[2:2]])
        push!(ms, parse(Int64, m))
    end

    return atom_ids, shells, ls, ms
end


"""
    apply_reorder(ao_labels, matrix; inverse=false, debug=false,
                  bothsides=false, orbital_dim=2)

Reorder an orbital-indexed quantity using a direct permutation derived from
the AO labels. The canonical order is `(atom, l, principal shell, m)`, so all
shells with the same angular momentum are contiguous and each individual
shell is ordered by increasing `m`.

By default orbitals index the columns of `matrix`. Set `orbital_dim=1` for a
row-indexed coefficient matrix, or `bothsides=true` for a square AO matrix.
`inverse=true` applies the inverse permutation.
"""
function apply_reorder(
    ao_labels::Union{Vector{String},Matrix{String}},
    matrix::AbstractMatrix;
    inverse::Bool=false,
    debug::Bool=false,
    bothsides::Bool=false,
    orbital_dim::Int=2,
)
    atom_ids, shells, ls, ms = unpack(ao_labels)
    permutation = sortperm(eachindex(atom_ids); by=i -> (
        atom_ids[i], ls[i], shells[i], ms[i],
    ))
    order = inverse ? invperm(permutation) : permutation
    n_orbitals = length(order)

    reordered = if bothsides
        size(matrix) == (n_orbitals, n_orbitals) || throw(DimensionMismatch(
            "both matrix dimensions must equal the number of AO labels",
        ))
        matrix[order, order]
    elseif orbital_dim == 1
        size(matrix, 1) == n_orbitals || throw(DimensionMismatch(
            "matrix row count must equal the number of AO labels",
        ))
        matrix[order, :]
    elseif orbital_dim == 2
        size(matrix, 2) == n_orbitals || throw(DimensionMismatch(
            "matrix column count must equal the number of AO labels",
        ))
        matrix[:, order]
    else
        throw(ArgumentError("orbital_dim must be 1 or 2"))
    end

    if debug
        labels = vec(ao_labels)
        for (new_index, old_index) in enumerate(order)
            println("$new_index <- $old_index: ", labels[old_index])
        end
    end

    return reordered
end


"""
    apply_reorder(ao_labels; inverse=false, full_info=false, debug=false)

Return the AO labels and their metadata in the same canonical order used by
the matrix method. The metadata arrays are permuted together with the labels.
"""
function apply_reorder(
    ao_labels::Union{Vector{String},Matrix{String}};
    inverse::Bool=false,
    full_info::Bool=false,
    debug::Bool=false,
)
    atom_ids, shells, ls, ms = unpack(ao_labels)
    permutation = sortperm(eachindex(atom_ids); by=i -> (
        atom_ids[i], ls[i], shells[i], ms[i],
    ))
    order = inverse ? invperm(permutation) : permutation
    reordered_labels = vec(ao_labels)[order]

    if debug
        labels = vec(ao_labels)
        for (new_index, old_index) in enumerate(order)
            println("$new_index <- $old_index: ", labels[old_index])
        end
    end

    if full_info
        return reordered_labels, atom_ids[order], ls[order], ms[order]
    end
    return reordered_labels, atom_ids[order]
end
