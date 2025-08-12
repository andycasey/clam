import pickle
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


input_path_prefix = "../korg_grid_subset_6d"


with open(f"{input_path_prefix}_meta.pkl", "rb") as f:
    meta = pickle.load(f)
    absorption_memmap_kwds = meta["absorption_memmap_kwds"]
    parameter_memmap_kwds = meta["parameter_memmap_kwds"]

print(dict(zip(meta["parameter_names"], meta["n_grid_points"])))

absorption_memmap = np.memmap(f"{input_path_prefix}_absorption.memmap", mode="r", **absorption_memmap_kwds)
x_memmap = np.memmap(f"{input_path_prefix}_parameters.memmap", mode="r", **parameter_memmap_kwds)

parameter_names = ["C_m", "alpha_m", "vmic", "m_H", "logg", "Teff"]

# Let's do PCA on the difference wrt C_m
# First we need to match all C_m != 0 spectra to their C_m = 0 counterparts

parameter_name = "C_m"
slice_dict = dict(alpha_m=0)

parameter_index = parameter_names.index(parameter_name)
subset_mask = np.ones(x_memmap.shape[0], dtype=bool)
for key, value in slice_dict.items():
    subset_mask *= (x_memmap[:, parameter_names.index(key)] == value)

# Get the C_m = 0 spectra
subset_has_zero_mask = subset_mask * (x_memmap[:, parameter_index] == 0.0)
subset_has_non_zero_mask = subset_mask * (x_memmap[:, parameter_index] != 0.0)

other_indices = tuple([i for i in range(len(parameter_names)) if i != parameter_index])
indices = []
for i in tqdm(np.where(subset_has_non_zero_mask)[0]):
        
    diff = x_memmap[i][[other_indices]] - x_memmap[:, other_indices]
    diff_match = (np.sum(np.abs(diff), axis=1) < 0.01)
    diff_match *= subset_has_zero_mask
    if not np.any(diff_match):
        continue

    assert np.sum(diff_match) == 1, "More than one match found for index {}".format(i)
    j = np.where(diff_match)[0][0]
    indices.append((i, j))

# setup delta_absorption arrays
slicer = (0, 2, 3, 4, 5)
iis = np.array([i for i, j in indices])
delta_x = x_memmap[iis].copy()[:, slicer]
delta_absorption = np.zeros((len(indices), absorption_memmap.shape[1]), dtype=absorption_memmap.dtype)
for k, (i, j) in tqdm(enumerate(indices)):
    delta_absorption[k] = absorption_memmap[i] - absorption_memmap[j]


delta_x_transformed = (
    (delta_x - min_parameters[[slicer]]) / (max_parameters[[slicer]] - min_parameters[[slicer]])
) * np.pi

n_modes = (5, 3, 7, 11, 15)
A = nufft.fourier_design_matrix(delta_x_transformed.T, n_modes)


model = PLSRegression(n_components=5)
model.fit(A, delta_absorption)

with open("c_m.pkl", "wb") as fp:
    pickle.dump((n_modes, min_parameters, max_parameters, parameter_names, model._x_mean, model.coef_, model.intercept_), fp)

# Only center X but do not scale it since the coefficients are already scaled
#X -= self._x_mean
#y_pred = X @ self.coef_.T + self.intercept_
