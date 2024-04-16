using Statistics, Plots, Setfield, LinearAlgebra

include("../src/data_manupulation.jl")
include("../src/model_construction.jl")

# model construction
# parameters 
maxdeg = 10
ord = 2
rcut = 10.0
Zi = 6
Zs = [6,1,8]
Lmax = 2
n_orbs = [3,2,1]
# construct the basis
onsite_model = On_Model(maxdeg, ord, rcut, Zi, Zs, Lmax, n_orbs)

# read data
molecule = TrajectoryHDF5("data/propanol.h5")

frame = read_frame(molecule,2)
R, D = translate_frame(frame)
R11 = get_state(R,1,1)
@time eval_model(onsite_model, R11)

function flat(a)
    tmp = ones(length(a[1]),length(a))
    for i in 1:length(a)
        tmp[:,i] = vec(a[i])
    end
    return tmp
end

function k2ij(k, n, m)
    i = div(k-1, m) + 1
    j = k - (i-1)*m
    return i, j
end
# ```
# fit function for onsite model:
#     model: a On_Model object, fitted or not
#     Rs: a vector of State objects <==> {R_II}_I
#     Ys: a vector of matrices <==> D_II
# ```
function fit!(model::On_Model, Rs::Union{Vector{State{T}},Vector{Vector{State{T}}}}, Ys::Vector{Matrix{TY}}) where {T, TY}
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
            # if l1 == 1 && l2 == 2
            #     @show size(partial_md(R,ps,st)[1][i][1])
            # end
            A[(2l1+1)*(2l2+1)*(j-1)+1:(2l1+1)*(2l2+1)*j,:] = flat(partial_md(R,ps,st)[1][i])
        end
        
        for kk = 1 : size(C, 1)
            ii, jj = k2ij(kk, n_orbs[l1+1], n_orbs[l2+1])
            println("Fitting the ($ii,$jj)-th ($l1,$l2) block")
            println()
            
            Yij = [ get_Y(Ys[t], n_orbs, n_orbs, l1, l2, ii, jj) for t = 1:length(Ys) ]

            # # debug
            # if l1 == 1 && l2 == 2 && ii == 1 && jj == 1
            #     @show Yij[1] == Ys[1][4:6,10:14]
            # end

            # @show typeof(Yij)
            # construct Y 
            Y = zeros(Float64, length(Ys)*length(Yij[1]))
            for k in 1:length(Ys)
                Y[(k-1)*length(Yij[1])+1:k*length(Yij[1])] = Yij[k]
            end
            # solve for C[kk]
            # TODO: enable more solvers and regularizers
            Γ = I
            λ = 1e-12
            # C[kk,:] = (A'*A + λ*Γ) \ (A'*Y) # naive solver - just for illustration
            C[kk,:] = qr([A; λ*Γ]) \ [Y; zeros(Float64,size(A,2))] # another naive solver

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

# function set_own!(ps,kk,C)
#     ps.dot[kk].W = C
#     return ps
# end

# Constructed Rs and Ys and it is all set
Rs = []
Ys = []
N_data = 1000
i = 1

molecule = TrajectoryHDF5("data/propanol.h5")
while length(Rs) < N_data
    frame = read_frame(molecule,i)
    R, D = translate_frame(frame)
    kk = rand([1,5,8])
    @assert frame["Atomic numbers"][kk] == 6.0
    Rkk = get_state(R,kk,kk)
    Dkk = get_block(D,kk,kk,frame["Basis set labels"])
    push!(Rs, Rkk)
    push!(Ys, Dkk)
    i += 5
end
Rs = identity.(Rs)
Ys = identity.(Ys)

onsite_model = fit!(onsite_model, Rs, Ys)

onsite_model.fitted

# eval_model(onsite_model, Rs[1])[4:9,10:14]
# Ys[1][4:9,10:14]


E = abs.(eval_model(onsite_model, Rs[1]) - Ys[1])

contourf(1:14,1:14,E)

# pos = [1,2,3,10,11,12,13,14]
# contourf(1:8,1:8,E[pos,pos])

Yss = vec(Ys[1])
for i = 2:length(Ys)
    push!(Yss, vec(Ys[i])...)
end 

Yrss = vec(eval_model(onsite_model, Rs[1]))
for i = 2:length(Rs)
    @show i
    push!(Yrss, vec(eval_model(onsite_model, Rs[i]))...)
end

Yss
Yrss

# plot(Yss[1:1000], Yss[1:1000])
# scatter!(Yss[1:5000], Yrss[1:5000])

# testing - interpolation

Rin = []
Yin = []
i = 1

while length(Rin) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    kk = rand([1,5,8])
    Rkk = get_state(R,kk,kk)
    Dkk = get_block(D,kk,kk,frame["Basis set labels"])
    push!(Rin, Rkk)
    push!(Yin, Dkk)
    i += 1
end

Rin = identity.(Rin)
Yin = identity.(Yin)

@time eval_model(onsite_model, Rin[1])-Yin[1]

# plot and other comparisons
Yins = vec(Yin[1])
for i = 2:length(Yin)
    push!(Yins, vec(Yin[i])...)
end 

Yinss = vec(eval_model(onsite_model, Rin[1]))
for i = 2:length(Rin)
    @show i
    push!(Yinss, vec(eval_model(onsite_model, Rin[i]))...)
end

# testing - extropolation

Rext = []
Yext = []
i = 5001

while length(Rext) < N_data
    frame = read_frame(molecule,i)
    @assert frame["Atomic numbers"][1] == 6.0
    R, D = translate_frame(frame)
    kk = rand([1,5,8])
    Rkk = get_state(R,kk,kk)
    Dkk = get_block(D,kk,kk,frame["Basis set labels"])
    push!(Rext, Rkk)
    push!(Yext, Dkk)
    i += 1
end

Rext = identity.(Rext)
Yext = identity.(Yext)

@time eval_model(onsite_model, Rext[1])-Yext[1]

# plot and other comparisons
Yexts = vec(Yext[1])
for i = 2:length(Yext)
    push!(Yexts, vec(Yext[i])...)
end

Yextss = vec(eval_model(onsite_model, Rext[1]))
for i = 2:length(Rext)
    @show i
    push!(Yextss, vec(eval_model(onsite_model, Rext[i]))...)
end


# plot(Yt, Yt)

# smallest = sortperm(abs.(Yss))
smallest = 1:196000

posi = 1:5000
plot(Yss[smallest][posi], Yss[smallest][posi])
scatter!(Yss[smallest][posi], Yrss[smallest][posi], label = "Training")
scatter!(Yins[smallest][posi], Yinss[smallest][posi], label = "Interpolation Testing")
scatter!(Yexts[smallest][posi], Yextss[smallest][posi], label = "Extrapolation Testing")