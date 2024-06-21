using Setfield, LinearAlgebra, ACEfit

# ```
# fit function for onsite model:
#     model: a On_Model object, fitted or not
#     Rs: a vector of State objects <==> {R_II}_I
#     Ys: a vector of matrices <==> D_II
# ```
# TODO : think about what is the best way to incorporate the regularization term (in solver? in line? in variables?)
Dict_Int2Spec = Dict(1 => "H", 6 => "C", 7 => "N", 8 => "O")
Dict_Int2Orbs = Dict(0 => "S", 1 => "P", 2 => "D", 3 => "F")
Dict_Spec2Int = Dict("H" => 1, "C" => 6, "N" => 7, "O" => 8)
Dict_Orbs2Int = Dict("S" => 0, "P" => 1, "D" => 2, "F" => 3)

function fit!(model::AbstractModel, Rs::Union{Vector{PState{T}},Vector{Vector{PState{T}}}}, Ys::Vector{Matrix{TY}}; solver = ACEfit.SKLEARN_BRR(), λ = 1e-12, Γ = I) where {T, TY}
    TP = typeof(model)
    LLset = [(l1,l2) for l2 in 0:get_L(model)[2], l1 in 0:get_L(model)[1]]
    n_orbs1, n_orbs2 = get_norbs(model)
    @assert(length(LLset) == length(model.ps.dot))

    layer_set = ["layer_$i" for i in 1:length(LLset)]
    layer_set = Symbol.(layer_set)

    # evaluate the EQM - just once !
    partial_md = Chain([model.model.layers[i] for i = 1:5]...)
    ps, st = Lux.setup(MersenneTwister(1234), partial_md)
    valset = [ partial_md(R,ps,st)[1] for R in Rs ]
    
    for (i, (l1,l2)) in enumerate(LLset)
        println("Fitting the $(Dict_Int2Orbs[l1])$(Dict_Int2Orbs[l2]) blocks ...")
        println()
        
        RMSE = 0
        # construct A
        A = zeros((2l1+1)*(2l2+1)*length(Rs), size(model.ps.dot[i].W,2))
        
        for j = 1:length(Rs)
            A[(2l1+1)*(2l2+1)*(j-1)+1:(2l1+1)*(2l2+1)*j,:] = flat(valset[j][i])
        end
        # regularization should not be done in this way, which slows down the calculations (and also underestimate the RMSE!!)!
        num = size(A)[2] # number of basis
        A = [A; λ*Γ]
        
        for kk = 1 : size(model.ps.dot[i].W,1)
            ii, jj = k2ij(kk, n_orbs1[l1+1], n_orbs2[l2+1])
            # println("Fitting the ($ii,$jj)-th $(Dict_Int2Orbs[l1])$(Dict_Int2Orbs[l2]) block ...")
            
            Yij = [ get_Y(Ys[t], n_orbs1, n_orbs2, l1, l2, ii, jj) for t = 1:length(Ys) ]

            # construct Y 
            Y = zeros(Float64, length(Ys)*length(Yij[1]))
            for k in 1:length(Ys)
                Y[(k-1)*length(Yij[1])+1:k*length(Yij[1])] = Yij[k]
            end
            Y = [Y; zeros(num)]

            # solve for C[kk]
            C = ACEfit.solve(solver, A, Y)["C"];
            @set! model.ps.dot.$(layer_set[i]).W[kk,:] = C
            # list of potential solvers: ACEfit: QR, LSQR, RRQR, SKLEARN_BRR, SKLEARN_ARD, BLR, TruncatedSVD...
            
            RMSE += norm((A*C-Y)[1:end-num])^2/(length(Y)-num)
        end
        println("RMSE = $(sqrt(RMSE/length(LLset)))")
        println()
    end
    
    model = (TP <: On_Model) ? TP(model.model, model.ps, model.st, model.n_orbs, true) : TP(model.model, model.ps, model.st, model.n_orbs1, model.n_orbs2, true)
    # @show model.ps.dot[1].W
    # @show model.fitted
    # return model
end

function split_data(frames::Vector{Dict{String, Array}}, keys::Base.KeySet{Union{T,Tuple{T,T}}}; Mode = "D") where T
    Rs = Dict(key => [] for key in keys)
    Ys = Dict(key => [] for key in keys)
    
    for frame in frames
        f = translate_frame(frame)
        for key in keys
            if !(typeof(key) <: Tuple)
                for i in findall(x->x==key, f["atomic_numbers"])
                    push!(Rs[key], get_state(f["R"], i, i))
                    push!(Ys[key], get_block(f[Mode], i, i, f["ao_labels"]))
                end
            else
                i, j = key
                for ii in findall(x->x==i, f["atomic_numbers"])
                    for jj in setdiff(findall(x->x==j, f["atomic_numbers"]),ii)
                        push!(Rs[key], get_state(f["R"], ii, jj))
                        push!(Ys[key], get_block(f[Mode], ii, jj, f["ao_labels"]))
                    end
                end
            end
        end
    end

    return Rs, Ys
end

# Fit a whole Density_Model
# Here, frames can be non_franslated frame (directly read from data) which will be transfer to a readable format (i.e. translate_frame) in split_data function
# The function should return a fitted Density_Model
function fit!(model::Density_Model,frames::Union{Dict{String, Array}, Vector{Dict{String, Array}}}; solver = ACEfit.SKLEARN_BRR(), Mode = "D")
    Rs, Ys = split_data(frames, keys(model.Models); Mode = Mode)
    for key in keys(model.Models)
        typeof(key) <: Tuple ? println("=== Fitting for $(Dict_Int2Spec[key[1]])-$(Dict_Int2Spec[key[2]]) offsite model ===") : println("==== Fitting for $(Dict_Int2Spec[key]) onsite model ====")
        println()
        if length(Rs[key]) == 0 || length(Ys[key]) == 0
            continue
        end
        model.Models[key] = fit!(model.Models[key], identity.(pop!(Rs,key)), identity.(pop!(Ys,key)); solver = solver)
    end
    if !isfitted(model)
        @warn("Some models are not fitted because there is a lack of corresponding data...")
    end
    # return model
end