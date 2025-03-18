This repository archives the codes used for learning the self-consistent density matrices $D_R$ for a given configuration $R$. The basic idea is to use the linear equivariant Atomic Cluster Expansion models, which have the same symmetry as the density matrix, to represent the density matrix. In order to enforce the predicted density matrix to be a valid density for the given system, a retraction operator is applied afterwards. 

More details about this approach are available at: 

https://arxiv.org/abs/2503.08400

An example of its use can be found on `../test/main.jl`, which illustrates the ways of reading data, constructing models, fitting parameters, saving the fitted model and predicting the density matrix.

One should run 

```julia
] registry add https://github.com/ACEsuit/ACEregistry.git
```

when setting up the Julia environment. 

This repo also uses a developing version of `ACEOperators.jl` (yet to be released), which is written by one of the authors. 

TODO: 

- [ ] (Outside this repo) Release `ACELuxOperators.jl`
- [ ] Modularization
- [ ] ...?
