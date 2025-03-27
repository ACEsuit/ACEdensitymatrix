This package contains the codes used for learning the density matrices $D_{\mathrm{R}}$ for a given molecular configuration $\mathrm{R}$, using the equivariant Atomic Cluster Expansion (ACE) descriptors, which have the same symmetry as the density matrix and form a linearly complete set to represent the density matrix. In order to enforce the predicted density matrix to be a valid density for some systems, a retraction operator is applied afterwards to ensure the Grassmannianness of the density matrix. 

This package is a `Julia` package that performs model construction, model training, and efficient density matrix prediction. An example of the use of this package can be found on `../test/MWE_arXiv.jl`, which illustrates how to read data, construct models, fit parameters, save the fitted model and predict the density matrix.

A part of the package is built on some packages under `ACEsuit` (e.g. `Polynomials4ML.jl`, `EquivariantModels` and `DecoratedParticles.jl`). As a result, the following commands

```julia
] registry add https://github.com/ACEsuit/ACEregistry.git
```

need to be run in Julia REPL when setting up the Julia environment. 

More details about the above approach are available at: 

https://arxiv.org/abs/2503.08400
