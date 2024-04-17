using Setfield, LinearAlgebra, ACEfit, Distributed, DistributedArrays
# using ACEfit: linear_solve, SKLEARN_ARD, SKLEARN_BRR
# using PyCall
# using LowRankApprox: pqrfact

# ```
# fit function for onsite model:
#     model: a On_Model object, fitted or not
#     Rs: a vector of State objects <==> {R_II}_I
#     Ys: a vector of matrices <==> D_II
# ```
function fit!(model::On_Model, Rs::Union{Vector{State{T}},Vector{Vector{State{T}}}}, Ys::Vector{Matrix{TY}}; Γ = I, λ = 1e-12, Solver = "LSQR") where {T, TY}
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
            if Solver == "QR"
                C[kk,:] = real(qr(A) \ Y)
            elseif Solver == "NaiveSolver"
                C[kk,:] = real((A'*A) \ (A'*Y))
            elseif Solver == "LSQR"
                Ad, Yd = distribute(A), distribute(Y)
                C[kk,:] = real(IterativeSolvers.lsqr(Ad, Yd; atol = 1e-6, btol = 1e-6))
                close(Ad), close(Yd)
            # elseif Solver == "ARD"
            #     C[kk,:] =  linear_solve(SKLEARN_ARD(;n_iter = niter, tol = ardtol), A, Y)["C"]
            # elseif Solver == "BRR"
            #     C[kk,:] =  linear_solve(SKLEARN_BRR(;n_iter = niter, tol = ardtol), A, Y)["C"]
            # elseif Solver == "RRQR"
            #     AP = A / I
            #     θP = pqrfact(A, rtol = ardtol) \ Y
            #     C[kk,:] =  I \ θP
            # elseif Solver == "ARD_false"
            #     ARD = pyimport("sklearn.linear_model")."ARDRegression"
            #     clf = ARD(n_iter=niter, threshold_lambda=10000, tol=ardtol,
            #             fit_intercept=false, compute_score=true)
            #     # BRR = pyimport("sklearn.linear_model")."BayesianRidge"
            #     # clf = BRR(n_iter=niter, tol=ardtol,
            #     #           fit_intercept=false, compute_score=true)
            #     clf.fit(A, Y)
            #     if length(clf.scores_) < niter
            #        @info "BRR converged to tol=$ardtol after $(length(clf.scores_)) iterations."
            #     else
            #        @warn "\n\nBRR did not converge to tol=$ardtol after n_iter=$niter iterations.\n\n"
            #     end
            #     C[kk,:] =  clf.coef_
            end

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