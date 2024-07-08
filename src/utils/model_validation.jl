function validate_model(MD, frames; Mode = "D")
    RMSE = 0
    RE = 0
    ME = 0
    if Mode == "H"
        global RMSE_H = 0
        global RE_H = 0
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
        return RMSE, RE, ME, RMSE_H, RE_H
    catch
        return RMSE, RE, ME
    end

end