#!/usr/bin/env python3

input_path = "korg_grid.h5"
output_path_prefix = "korg_grid_subset_6d"
transform_type = "one_minus_flux"

import os
import numpy as np
import h5py as h5
import pickle
from utils import parse_marcs_photosphere_path

try:
    flux_transform = {
        "neg_ln_flux": lambda x: -np.log(x),
        "one_minus_flux": lambda x: 1 - x,
    }[transform_type]
except KeyError:
    raise ValueError(f"Unknown transform type: {transform_type}")

with h5.File(input_path, "r") as fp:
    parameter_names = ("C_m", "alpha_m", "vmic", "m_H", "logg", "Teff")
    slices = (
        slice(1, 2), # N_m
        slice(0, None), # C_m
        slice(0, None), # alpha_m
        slice(0, None), # vmic
        slice(0, None), # m_H
        slice(0, None), # logg
        slice(0, 27),   # Teff (keep above 3000 K)
    )

    flip_axes = (parameter_names.index("Teff"), )  # Flip the Teff axis

    min_parameters = np.array([np.min(fp[k][s]) for k, s in zip(parameter_names, slices[1:])])
    max_parameters = np.array([np.max(fp[k][s]) for k, s in zip(parameter_names, slices[1:])])
    grid_parameters = [fp[k][s] for k, s in zip(parameter_names, slices[1:])]    
    flux = fp["spectra"][slices]

    mask = (
        (flux > 0).all(axis=-1)
    *   (fp["converged_flag"][slices] == 1)
    )

    flux = flux.reshape(flux.shape[1:])
    mask = mask.reshape(mask.shape[1:])
    
    teff_index = parameter_names.index("Teff")
    n_removed_by_holes, skipped_out_of_bounds, skipped_not_vmic = (0, 0, 0)

    with open("marcs_filled_atmosphere_paths.txt", "r") as marcs_fp:
        for line in marcs_fp:
            r = parse_marcs_photosphere_path(line)
            if not (max_parameters[teff_index] >= r["Teff"] >= min_parameters[teff_index]):
                skipped_out_of_bounds += 1
                continue

            is_plane_parallel = (r["logg"] >= 3.5)
            is_spherical = not is_plane_parallel
            if (
                (is_plane_parallel and r["vmic"] == 1.0)
            |   (is_spherical and r["vmic"] == 2.0)
            ):
                # all things in that v_micro slice
                indices = [slice(None)] * len(parameter_names)
                for i, pn in enumerate(parameter_names):
                    if pn in ("vmic", "N_m"): continue
                    indices[i] = np.where(fp[pn][:] == r[pn])[0]
                    
                indices = tuple(indices)
                mask[indices] = False
                n_removed_by_holes += mask[indices].size
            else:
                skipped_not_vmic += 1

    print(f"Removed {n_removed_by_holes:,} grid points due to holes in the MARCS grid.")
    print(f"Skipped {skipped_out_of_bounds:,} grid points out of bounds.")
    print(f"Skipped {skipped_not_vmic:,} grid points not in the vmic slice.")

    if flip_axes is not None:
        for axes in flip_axes:
            grid_parameters[axes] = np.flip(grid_parameters[axes])

        mask = np.flip(mask, axis=flip_axes)
        flux = np.flip(flux, axis=flip_axes)

    n_grid_points = flux.shape[:-1]
    flux = np.clip(flux_transform(flux[mask]), 0, None)

    x = np.array(np.meshgrid(*grid_parameters, indexing='ij')).reshape(len(parameter_names), -1).T[mask.flatten()]

    print(f"flux shape: {flux.shape}")
    print(f"x shape: {x.shape}")

    print(f"Writing metadata with prefix {output_path_prefix}...")
    with open(f"{output_path_prefix}_meta.pkl", "wb") as fp:
        pickle.dump(
            dict(
                parameter_names=parameter_names,
                min_parameters=min_parameters,
                max_parameters=max_parameters,
                grid_parameters=grid_parameters,
                n_grid_points=n_grid_points,
                mask=mask,
                flip_axes=flip_axes,
                transform_type=transform_type,
                input_path=input_path,
                prepare_subset_path=os.path.abspath(__file__),
                absorption_memmap_kwds=dict(
                    dtype=flux.dtype,
                    shape=flux.shape
                ),
                parameter_memmap_kwds=dict(
                    dtype=x.dtype,
                    shape=x.shape,
                ),
            ), 
            fp
        )

    print(f"Writing absorption with prefix {output_path_prefix}...")    
    absorption_memmap = np.memmap(f"{output_path_prefix}_absorption.memmap", dtype=flux.dtype, mode="w+", shape=(np.sum(mask), flux.shape[-1]))
    absorption_memmap[:] = flux
    absorption_memmap.flush()
    del absorption_memmap

    print(f"Writing parameters with prefix {output_path_prefix}...")
    x_memmap = np.memmap(f"{output_path_prefix}_parameters.memmap", dtype=x.dtype, mode="w+", shape=x.shape)
    x_memmap[:] = x
    x_memmap.flush()
    del x_memmap

