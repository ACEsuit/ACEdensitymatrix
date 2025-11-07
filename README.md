This package is a `Julia` package that contains the codes used for learning the density matrices $D_{\mathrm{R}}$ for a given molecular configuration $\mathrm{R}$, using the equivariant Atomic Cluster Expansion (ACE) descriptors, which have the same symmetry as the density matrix and form a linearly complete set to represent the density matrix. In order to enforce the predicted density matrix to be a valid density for some systems, a retraction operator is applied afterwards to ensure the Grassmannianity of the density matrix. One of the goals of this package is to provide the reproducibility of the results in the following manuscript:

[https://arxiv.org/abs/2503.08400](https://arxiv.org/abs/2503.08400).

Notably, this package can also be used to learn any other equivariant operators, for instance the Fock matrices (Hamiltonians) or the overlap matrices, cf. the end of this README file. 

# Get started 

Since this is a Julia package, it is a prerequisite to have Julia installed on your device, for which one may refer to the [Julialang site](https://julialang.org/downloads/). 

NOTE: Currently, some dependencies of this package only allow Julia 1.9, so please use Julia 1.9 to install and use this package, for the time being.

The package is registered under `ACEsuit`. To install packages under `ACEsuit`, the following command needs to be run in Julia REPL when setting up the Julia environment.
```julia
] registry add https://github.com/ACEsuit/ACEregistry.git
```
Then, the package is ready to be installed by running
```julia
] add ACEdensitymatrix
```
After that, the ACEdensitymatrix package can be used via
```julia
using ACEdensitymatrix
```

Hereafter, we provide a step by step tutorial showing how the package is to be used, from data reading, model construction/fitting/IO to prediction and model validation. We also provide a more comprehensive end-to-end script to run the whole tutorial at [test/MWE_arXiv.jl](https://github.com/ACEsuit/ACEdensitymatrix/blob/main/test/MWE_arXiv.jl).

# Data

The data we used in this project is the DFT results of some typical molecules with the level of theory being DFT ωB97X-D/6-31G(d). We archive the data for reproducibility at [doi.org/10.18419/DARUS-4902](https://doi.org/10.18419/DARUS-4902), which can be read into Julia as

```julia
routine = "YOUR_ROUTINE"
training_molecule = ["propanol"] # list of molecules with which the model will be trained
filenames = ["$routine/data/$(training_molecule[i])_new.h5" for i in 1:length(training_molecule)] # the filenames of the chosen molecules

frames = [] # empty training set
frames_test = [] # empty test set
train_set = 0:2999 # frames picked from this molecule for training - this can be chosen differently for different molecules
test_set = 9000:9999 # frames picked from this molecule for test - this can be chosen differently for different molecules

for (i,fname) in enumerate(filenames) # loop for each training molecule
    molecule = TrajectoryHDF5(fname)
    push!(frames,[ read_frame(molecule,Int(j)) for j in train_set]...) # constructing a training set with the chosen frames
    push!(frames_test,[ read_frame(molecule,Int(j)) for j in test_set]...) # constructing a test set with the chosen frames
end

frames[1]
```

The last line allows one to view the structure of our data - the coordinates, the KS and overlap matrix, the eigenvectors(Coefficients), from which the density matrix can be deduced, as well as the metadata.
```julia
Dict{String, Array} with 7 entries:
  "Atomic numbers"   => [6, 1, 1, 1, 6, 1, 1, 6, 1, 1, 8, 1]
  "Basis set labels" => ["0 C 1s 0" "0 C 2s 0" … "11 H 1s 0" "11 H 2s 0"]
  "Coefficients"     => [4.42029e-5 0.000387458 … 0.000711201 0.00221085; -0.000411482 -0.000822196 … 9.42007e-5 0.00110346; … ; 0.0118902 -0.0217341 … 0.110175 0.0764495; 0.00158634 0.000962375 … 0.000681027 -0.0023718]
  "Coordinates"      => [8.20934 8.06043 … 8.21433 8.59295; 2.30922 1.99941 … 5.41169 5.67868; -21.8139 -22.8485 … -22.2938 -21.4142]
  "Overlap"          => [1.0 0.219059 … 4.70417e-11 0.00027372; 0.219059 1.0 … 4.43876e-6 0.00479534; … ; 4.70417e-11 4.43876e-6 … 1.0 0.658292; 0.00027372 0.00479534 … 0.658292 1.0]
  "Kohn-Sham matrix" => [-10.2747 -2.45905 … -3.86501e-6 -0.0031578; -2.45905 -1.0391 … 0.00022585 -0.0114218; … ; -3.86501e-6 0.00022585 … -0.180854 -0.54048; -0.0031578 -0.0114218 … -0.54048 -0.507038]
```
This is also to showcase what format of data is required to use this package. 

# Model construction
To construct the model, the model parameters should first be given
```julia
degree = 4        # Polynomial degree - resolution of the one-particle basis
order = 2         # Correlation order = Body order + const
rcut = 4.0        # environment cutoff radial
zcut = 10.0       # bond cutoff radial

ao_dict = Dict( 1 => Dict("n_orbs" => [2], "maxdeg" => degree, "ord" => order, "rcut_on" => rcut, "rcut_off" => rcut, "zcut" => zcut),
                6 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut_on" => rcut, "rcut_off" => rcut, "zcut" => zcut),
                8 => Dict("n_orbs" => [3,2,1], "maxdeg" => degree, "ord" => order, "rcut_on" => rcut, "rcut_off" => rcut, "zcut" => zcut) );
                # This is a CHO model - For H, there are 2 s orbitals, and for C and O, there are 3 s orbitals, 2 p orbitals and 1 d orbital.
                # By adjusting the above input, it is possible to construct models for different basis sets and different elements.

Model = Density_Model(ao_dict::Dict); # Construct the model
```

# Training
We use the QR solver in ACEfit.jl to perform an example fitting. Note that one may need to `add ACEfit` before running the following
```julia
using ACEfit # To use the solver inside
fit!(Model, frames; solver = ACEfit.QR(), λ = 1e-4, reg = :smooth)
# Here λ is the regularization parameter and we use the smooth prior as the regularizer.
```

# Prediction
Once the parameters in the model are fitted, a prediction of the density matrix for a given molecule is ready to be given.
```julia
fm = frames_test[1] # take one test frame
# Converting the frame to the format that is directly needed
R, D, atomic_number, ao_labels, H, S, C = convert_frame(fm)["R"], convert_frame(fm)["D"], convert_frame(fm)["atomic_numbers"], convert_frame(fm)["ao_labels"], convert_frame(fm)["H"], convert_frame(fm)["S"], convert_frame(fm)["C"];
# Predicting the density matrix of this system with a proper retraction
D_pred = eval_model(Model, R, ao_labels, retraction =  D -> eigen_retraction(D, Int(sum(atomic_number)/2)))
D - D_pred # View the difference
```

# Model validation (error)
The model is to be tested on the chosen test set, and to return some errors
```julia
# construct the test set
RMSE, RMSE_MIN, RMSE_MAX, RE, ME, MAE = validate_model(Model_load, frames_test) # Element-wise RMSE, minimum/maximum RMSE over the test frames, relative error ||D-D_pred|| / ||D||, maximum element-wise error, and the MAE of the element-wise error
```
Successfully running up to this line will give an RMSE of 0.0012828423636450006 (~1E-2.9), which corresponds to the first orange point in Figure 4(c) of the manuscript. 

# Save and load the models
The trained model oftentimes needs to be stored and reused. Here we used a JLD2 format, but in principle, any format that can store a dictionary with numbers and strings can be used
```julia
using JLD2

save("YOUR_ROUTINE/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2", write_dict(Model))

# And load
Model_load = load("YOUR_ROUTINE/model_maxdeg$(degree)_ord$(order)_rcut$(rcut)_zcut$(zcut).jld2") |> read_dict
RMSE, RMSE_MIN, RMSE_MAX, RE, ME, MAE = validate_model(Model_load, frames_test) # A consistent result will be obtained
```
Note that one may need to `add JLD2` before running the code block above.

# Remarks
It is worth mentioning that the code in this package can not only be used to fit the density matrix, but can also be used to fit the KS matrix or any other equivariant operators of many-body systems, as long as they have the correct rotation symmetry. It also does not require a specific origination of the data. The only requirement is that the data should look like the above, having the coordinates of the particles, the necessary metadata (the orbital information, that gives rise to the correct equivariance), and of course, access to the target operator (say "operator"). Then to fit the operator of interest, simply make sure that the converted frame, as a dictionary shown above, has the field "operator" (this will mean that you will need to import and overwrite the `convert_frame` function) and then run the following when the fitting is performed:
```julia
fit!(Model, frames; solver = ACEfit.QR(), λ = 1e-4, reg = :smooth, Mode = "operator")
```

If you use this package in your research or project, please cite it as follows: 

```bibtex
@misc{Zhang2025,
  title         = {A symmetry-preserving and transferable representation for learning the Kohn-Sham density matrix},
  author        = {Zhang, L. and Mazzeo, P. and Nottoli, M. and Cignoni, E. and Cupellini, L. and Stamm, B.},
  year          = {2025},
  eprint        = {2503.08400},
  archiveprefix = {arXiv},
  primaryclass  = {physics.chem-ph}
}

@misc{ACEdensitymatrix,
  title      = {ACEdensitymatrix.jl},
  author     = {Zhang, L et al.},
  year       = {2025},
  url        = {https://github.com/ACEsuit/ACEdensitymatrix}
}
```
