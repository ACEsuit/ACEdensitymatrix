This repo aims to fit self-consistent density matrices $D_R$ for given configuration $R$. The basic idea is to use equivariant Atomic Cluster Expansion (ACE) to generate linear models that have the same symmetry as the density matrix. 

An example of its use can be found on `../test/main.jl`, which illustrates the ways of reading data, constructing models, fitting parameters, and predicting the density matrix.

A note before using it is that this repo relies on a developing version of `Polynomials4ML.jl`.

One should run 

```julia
]activate .

add `https://github.com/ACEsuit/Polynomials4ML.jl#generic_linear
```

to set up the Julia environment. 

This repo also uses a developing version of `ACELuxOperators.jl`, which is not yet released and hence some code in this repo is directly "hacked" from ACELuxOperators.jl, which is also written by the author. 

This should be fixed once `ACELuxOperators.jl` is released.

TODO: 

- [ ] (Outside this repo) Release `ACELuxOperators.jl`
- [x] Optimize output print out
- [x] Modify Onsite radial basis (something not very elegant now)
- [ ] Implement other offsite-state constructions (including cutoff)
- [ ] Figure out a way to save a constructed lux chain (basically A2Bmap)
- [ ] Efficiency optimization
- [ ] Modularization
- [ ] ...?
