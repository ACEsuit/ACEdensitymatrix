function unpack(ao_labels::Union{Vector{String},Matrix{String}})
    atom_ids = Int64[]
    atom_symbols = String[]
    shells = Int64[]
    ls = Int64[]
    ms = Int64[]
    letter_to_l = Dict{String, Int64}("s" => 0, "p" => 1, "d" => 2, "f" => 3)
    for label in ao_labels
        atom_id, atom_symbol, orbital, m = split(label)

        atom_id = parse(Int64, atom_id)
        shell = parse(Int64, orbital[1:1])
        l = letter_to_l[orbital[2:2]]
        m = parse(Int64, m)

        push!(atom_ids, atom_id)
        push!(atom_symbols, atom_symbol)
        push!(shells, shell)
        push!(ls, l)
        push!(ms, m)
    end

    return atom_ids, atom_symbols, shells, ls, ms
end


function apply_reorder(ao_labels::Union{Vector{String},Matrix{String}}, matrix::Matrix{Float64};
        inverse=false, debug=false, bothsides=false)

    pos = typeof(ao_labels) == Vector{String} ? 1 : 2

    # keep a copy of the reordering for debug
    ref_order = Vector{Int64}(1:size(ao_labels, pos))

    # unpack the labels
    atom_ids, atom_symbols, shells, ls, ms = unpack(ao_labels)

    rotated_matrix = copy(matrix)

    # apply the transformations atom by atom
    for atom_id in 0:maximum(atom_ids)

        # here we assume that functions on the same atom are contiguous,
        # and compute the range of the atom block. If atoms are not
        # contiguous this must be changes
        atom_mask = findall(x->x==atom_id, atom_ids)
        atom_start = minimum(atom_mask)
        atom_stop = maximum(atom_mask)
        atom_range = atom_start:atom_stop

        # build and apply the unitary transformation for ordering m by value
        for shell in 1:maximum(shells[atom_range])
            shell_mask = findall(x->x==shell, shells[atom_range])
            shell_start = minimum(shell_mask)
            shell_stop = maximum(shell_mask)
            shell_range = atom_start+shell_start-1:atom_start+shell_stop-1

            # note: the range starts from 1 as we skip the s orbitals
            for l in 1:maximum(ls[shell_range])
                l_mask = findall(x->x==l, ls[shell_range])
                l_start = minimum(l_mask)
                l_stop = maximum(l_mask)
                l_range = atom_start+shell_start+l_start-2:atom_start+shell_start+l_stop-2

                Um = zeros(Float64, size(l_range, 1), size(l_range, 1))
                m = ms[l_range]
                for (i, j) in enumerate(sortperm(m))
                    Um[i, j] = 1.0
                end

                ref_order[l_range] = Um * ref_order[l_range]

                if inverse
                    Um = Um'
                end
                rotated_matrix[l_range, :] = Um * rotated_matrix[l_range, :]

                if bothsides
                    rotated_matrix[:, l_range] = rotated_matrix[:, l_range] * Um'
                end
            end
        end

        # build the unitary transformation for bringing the same values
        # of l close
        Ul = zeros(Float64, size(atom_range, 1), size(atom_range, 1))
        atom_ls = ls[atom_range]
        for (i, j) in enumerate(sortperm(atom_ls))
            Ul[i, j] = 1.0
        end

        ref_order[atom_range] = Ul * ref_order[atom_range]

        if inverse
            Ul = Ul'
        end
        rotated_matrix[atom_range, :] = Ul * rotated_matrix[atom_range, :]

        if bothsides
            rotated_matrix[:, atom_range] = rotated_matrix[:, atom_range] * Ul'
        end
    end

    # print the reordered labels for debug
    if debug
        for i in 1:size(ao_labels, 1)
            println(ao_labels[i], "  ->  ", ao_labels[ref_order[i]])
        end
    end

    return rotated_matrix
end


function apply_reorder(ao_labels::Union{Vector{String},Matrix{String}})

    pos = typeof(ao_labels) == Vector{String} ? 1 : 2
    # keep a copy of the reordering for debug
    ref_order = Vector{Int64}(1:size(ao_labels, pos))

    # unpack the labels
    atom_ids, atom_symbols, shells, ls, ms = unpack(ao_labels)

    # apply the transformations atom by atom
    for atom_id in 0:maximum(atom_ids)

        # here we assume that functions on the same atom are contiguous,
        # and compute the range of the atom block. If atoms are not
        # contiguous this must be changes
        atom_mask = findall(x->x==atom_id, atom_ids)
        atom_start = minimum(atom_mask)
        atom_stop = maximum(atom_mask)
        atom_range = atom_start:atom_stop

        # build and apply the unitary transformation for ordering m by value
        for shell in 1:maximum(shells[atom_range])
            shell_mask = findall(x->x==shell, shells[atom_range])
            shell_start = minimum(shell_mask)
            shell_stop = maximum(shell_mask)
            shell_range = atom_start+shell_start-1:atom_start+shell_stop-1

            # note: the range starts from 1 as we skip the s orbitals
            for l in 1:maximum(ls[shell_range])
                l_mask = findall(x->x==l, ls[shell_range])
                l_start = minimum(l_mask)
                l_stop = maximum(l_mask)
                l_range = atom_start+shell_start+l_start-2:atom_start+shell_start+l_stop-2

                Um = zeros(Float64, size(l_range, 1), size(l_range, 1))
                m = ms[l_range]
                for (i, j) in enumerate(sortperm(m))
                    Um[i, j] = 1.0
                end

                ref_order[l_range] = Um * ref_order[l_range]
            end
        end

        # build the unitary transformation for bringing the same values
        # of l close
        Ul = zeros(Float64, size(atom_range, 1), size(atom_range, 1))
        atom_ls = ls[atom_range]
        for (i, j) in enumerate(sortperm(atom_ls))
            Ul[i, j] = 1.0
        end

        ref_order[atom_range] = Ul * ref_order[atom_range]
    end

    return ao_labels[ref_order]
end


