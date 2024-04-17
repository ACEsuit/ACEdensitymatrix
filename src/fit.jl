using Setfield, LinearAlgebra, ACEfit

# ```
# fit function for onsite model:
#     model: a On_Model object, fitted or not
#     Rs: a vector of State objects <==> {R_II}_I
#     Ys: a vector of matrices <==> D_II
# ```
function fit!(model::On_Model, Rs::Union{Vector{State{T}},Vector{Vector{State{T}}}}, Ys::Vector{Matrix{TY}}; Γ = I, λ = 1e-12, solver = ACEfit.LSQR(damp = 0, atol = 1e-6)) where {T, TY}
    LLset = [(l1,l2) for l2 in 0:get_L(model), l1 in 0:get_L(model)]
    n_orbs = get_norbs(model)
    @assert(length(LLset) == length(model.ps.dot))

    layer_set = ["layer_$i" for i in 1:length(LLset)]
    layer_set = Symbol.(layer_set)

    for (i, (l1,l2)) in enumerate(LLset)
        println("Fitting for l1 = $l1, l2 = $l2")
        println()
        
        # C can simply be a vector - saving memory
        C = zeros(eltype(model.ps.dot[i].W),size(model.ps.dot[i].W)...)
        # construct A
        A = zeros((2l1+1)*(2l2+1)*length(Rs), size(C,2))
        
        partial_md = Chain([model.model_on.layers[i] for i = 1:6]...)
        ps, st = Lux.setup(MersenneTwister(1234), partial_md)
        for (j, R) in enumerate(Rs)
            A[(2l1+1)*(2l2+1)*(j-1)+1:(2l1+1)*(2l2+1)*j,:] = flat(partial_md(R,ps,st)[1][i])
        end
        num = size(A)[2] # number of basis
        A = [A; λ*Γ]
        
        for kk = 1 : size(C, 1)
            ii, jj = k2ij(kk, n_orbs[l1+1], n_orbs[l2+1])
            println("Fitting the ($ii,$jj)-th ($l1,$l2) block")
            println()
            
            Yij = [ get_Y(Ys[t], n_orbs, n_orbs, l1, l2, ii, jj) for t = 1:length(Ys) ]

            # construct Y 
            Y = zeros(Float64, length(Ys)*length(Yij[1]))
            for k in 1:length(Ys)
                Y[(k-1)*length(Yij[1])+1:k*length(Yij[1])] = Yij[k]
            end
            Y = [Y; zeros(num)]

            # solve for C[kk]
            C[kk,:] = ACEfit.solve(solver, A, Y)["C"]
            # list of potential solvers: ACEfit: QR, LSQR, RRQR, SKLEARN_BRR, SKLEARN_ARD, BLR, TruncatedSVD...

            @show norm(A * C[kk,:] - Y)
            # @show C[kk,:]
        end
        @set! model.ps.dot.$(layer_set[i]).W = C
    end
    
    model = On_Model(model.model_on, model.ps, model.st, true)
    # @show model.ps.dot[1].W
    # @show model.fitted
    return model
end