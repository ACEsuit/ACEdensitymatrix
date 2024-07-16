function validate_model(MD, frames; Mode = "D")
    RMSE = 0
    RE = 0
    ME = 0
    if Mode == "H"
        global RMSE_H = 0
        global RE_H = 0
        global ME_H = 0
    end

    for frame in frames
        R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
        
        if Mode == "D"
            D_pred = eval_model(MD, R, ao_labels, retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
            RMSE += norm(D_pred - D)^2 / length(D)
            RE += norm(D_pred - D) / norm(D)
            ME = maximum( [maximum(maximum(abs.(D_pred - D))), ME] )
        elseif Mode == "H"
            H_pred = eval_model(MD, R, ao_labels)
            RMSE_H += norm(H_pred - H)^2 / length(H)
            RE_H += norm(H_pred - H) / norm(H)
            ME_H = maximum( [maximum(maximum(abs.(H_pred - H))), ME_H] )

            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred = C_pred * C_pred'

            RMSE += norm(D_pred - D)^2 / length(D)
            RE += norm(D_pred - D) / norm(D)
            ME = maximum( [maximum(maximum(abs.(D_pred - D))), ME] )
        end

        
    end

    RMSE = sqrt(RMSE/length(frames)) 
    RE /= length(frames)
    
    try 
        RMSE_H = sqrt(RMSE_H/length(frames)) 
        RE_H /= length(frames) 
        return RMSE, RE, ME, RMSE_H, RE_H, ME_H
    catch
        return RMSE, RE, ME
    end

end

function validate_model_semifull(MD, frames; Mode = "D")
    RMSE = Dict()
    n_samples = Dict()

    for key in keys(MD.Models)
        len = length(get_norbs(MD.Models[key])[1]) * length(get_norbs(MD.Models[key])[2])
        RMSE[key] = .0 # zeros(len)
        # RE[key] = .0 # zeros(len)
        # ME[key] = .0 # zeros(len)
        n_samples[key] = 0
    end

    # RMSE = identity.(RMSE)
    # RE = identity.(RE)
    # ME = identity.(ME)
    # n_samples = identity.(n_samples)

    for frame in frames
        R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
        
        if Mode == "D"
            D_pred = eval_model(MD, R, ao_labels, retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
        elseif Mode == "H"
            H_pred = eval_model(MD, R, ao_labels)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred = C_pred * C_pred'
        end
        ao_labels, atom_ids = apply_reorder(ao_labels) # with this line, we fit the reordered Density matrix in the correct order but need to map it back to the original order
        atom_ids .+= 1

        for I = 1:length(R)
            for J = I:length(R)
                pos_I = findall(x->x==I, atom_ids)
                pos_J = findall(x->x==J, atom_ids)
                if I == J
                    RMSE[R[I].Z] += norm(D_pred[pos_I,pos_J] - D[pos_I,pos_J])^2
                    n_samples[R[I].Z] +=  length(D[pos_I,pos_J])
                else
                    if R[I].Z > R[J].Z
                        RMSE[(R[J].Z,R[I].Z)] += norm(D_pred[pos_J,pos_I] - D[pos_J,pos_I])^2
                        n_samples[(R[J].Z,R[I].Z)] +=  length(D[pos_J,pos_I])
                    else
                        RMSE[(R[I].Z,R[J].Z)] += norm(D_pred[pos_I,pos_J] - D[pos_I,pos_J])^2
                        n_samples[(R[I].Z,R[J].Z)] +=  length(D[pos_I,pos_J])
                    end
                end
            end
        end
    end

    for key in keys(MD.Models)
        if n_samples[key] == 0
            RMSE[key] = 0
        else
            RMSE[key] = sqrt(RMSE[key]/n_samples[key])
        end
    end

    return RMSE
end

function validate_model_full(MD, frames; Mode = "D")
    RMSE = Dict()
    n_samples = Dict()

    for key in keys(MD.Models)
        # len = length(get_norbs(MD.Models[key])[1]) * length(get_norbs(MD.Models[key])[2])
        LLset = [(L1, L2) for L1 in 0:get_L(MD.Models[key])[1] for L2 in 0:get_L(MD.Models[key])[2]]
        RMSE[key] = Dict([LL => .0 for LL in LLset]) # zeros(len)
        # RE[key] = .0 # zeros(len)
        # ME[key] = .0 # zeros(len)
        n_samples[key] = Dict([LL => 0 for LL in LLset])
    end

    for frame in frames
        R, D, atomic_number, ao_labels, H, S, C = translate_frame(frame)["R"], translate_frame(frame)["D"], translate_frame(frame)["atomic_numbers"], translate_frame(frame)["ao_labels"], translate_frame(frame)["H"], translate_frame(frame)["S"], translate_frame(frame)["C"]
        
        if Mode == "D"
            D_pred = eval_model(MD, R, ao_labels, retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2))) # predicted density matrix with retraction
        elseif Mode == "H"
            H_pred = eval_model(MD, R, ao_labels)
            s, q = eigen(Symmetric(S))
            s_half = q * Diagonal(s.^(-1/2)) * q'
            H_pred = Symmetric(s_half * H_pred * s_half)
            C_pred = eigen(H_pred).vectors[:,1:Int(sum(atomic_number)/2)]
            D_pred = C_pred * C_pred'
        end
        ao_labels, atom_ids = apply_reorder(ao_labels) # with this line, we fit the reordered Density matrix in the correct order but need to map it back to the original order
        atom_ids .+= 1

        for I = 1:length(R)
            for J = I:length(R)
                pos_I = findall(x->x==I, atom_ids)
                pos_J = findall(x->x==J, atom_ids)
                if I == J
                    model = MD.Models[R[I].Z]
                    LLset = [(L1, L2) for L1 in 0:get_L(model)[1] for L2 in 0:get_L(model)[2]]
                    for (L1,L2) in LLset
                        RMSE[R[I].Z][L1,L2] += norm(get_block(D_pred, I, J, ao_labels, L1, L2) - get_block(D, I, J, ao_labels, L1, L2))^2
                        n_samples[R[I].Z][L1,L2] +=  length(get_block(D_pred, I, J, ao_labels, L1, L2))
                    end
                else
                    if R[I].Z > R[J].Z
                        model = MD.Models[R[J].Z, R[I].Z]
                        LLset = [(L1, L2) for L1 in 0:get_L(model)[1] for L2 in 0:get_L(model)[2]]
                        for (L1,L2) in LLset
                            RMSE[(R[J].Z,R[I].Z)][L1,L2] += norm(get_block(D_pred, J, I, ao_labels, L1, L2) - get_block(D, J, I, ao_labels, L1, L2))^2
                            n_samples[(R[J].Z,R[I].Z)][L1,L2] +=  length(get_block(D_pred, J, I, ao_labels, L1, L2))
                        end
                    else
                        model = MD.Models[R[I].Z, R[J].Z]
                        LLset = [(L1, L2) for L1 in 0:get_L(model)[1] for L2 in 0:get_L(model)[2]]
                        for (L1,L2) in LLset
                            RMSE[(R[I].Z,R[J].Z)][L1,L2] += norm(get_block(D_pred, I, J, ao_labels, L1, L2) - get_block(D, I, J, ao_labels, L1, L2))^2
                            n_samples[(R[I].Z,R[J].Z)][L1,L2] +=  length(get_block(D_pred, I, J, ao_labels, L1, L2))
                        end
                    end
                end
            end
        end
    end

    for key in keys(MD.Models)
        # len = length(get_norbs(MD.Models[key])[1]) * length(get_norbs(MD.Models[key])[2])
        LLset = [(L1, L2) for L2 in 0:get_L(MD.Models[key])[2] for L1 in 0:get_L(MD.Models[key])[1]]
        for LL in LLset
            if n_samples[key][LL] == 0
                RMSE[key][LL] = 0
            else
                RMSE[key][LL] = sqrt(RMSE[key][LL]/n_samples[key][LL])
            end
        end
    end

    return RMSE
end
