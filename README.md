This repo aims to fit self-consistent density matrices $D_R$ for given configuration $R$. The basic idea is to use equivariant Atomic Cluster Expansion (ACE) to generate linear models that have the same symmetry as the density matrix. 

An example of its use can be found on `../test/main.jl`, which illustrates the ways of reading data, constructing models, fitting parameters, saving the fitted model and predicting the density matrix.

One should run 

```julia
] registry add https://github.com/ACEsuit/ACEregistry.git
```

when set up the Julia environment. 

This repo also uses a developing version of `ACELuxOperators.jl`, which is not yet released and hence some code in this repo is directly "hacked" from ACELuxOperators.jl, which is also written by the author. 

This should be fixed once `ACELuxOperators.jl` is released.

TODO: 

- [ ] (Outside this repo) Release `ACELuxOperators.jl`
- [x] Optimize output print out
- [x] Modify Onsite radial basis (something not very elegant now)
- [x] Implement other offsite-state constructions (including cutoff)
- [x] Figure out a way to save a constructed lux chain (basically A2Bmap)
- [ ] Efficiency optimization (partly done)
- [ ] Modularization
- [ ] ...?
