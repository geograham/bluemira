# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Attempt at recreating the EU-DEMO 2017 reference equilibria from a known coilset.
"""

# %%
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_flux_surfs
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.optimisation.problem import (
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.solve import (
    DudsonConvergence,
    HailMaryConvergence,
    PicardIterator,
)
from bluemira.geometry.coordinates import Coordinates

# %%
# SuperX Equilbria
file_path = Path(
    get_bluemira_path("equilibria/data", subfolder="examples"),
    "2017_SuperX_SOF_eqdsk_2MVM98_v1_0.eqdsk",
)
super_eq = Equilibrium.from_eqdsk(
    file_path, from_cocos=7, qpsi_positive=False, force_symmetry=True
)
_, ax = plt.subplots()
super_eq.plot(ax=ax)
super_eq.coilset.plot(ax=ax)
plt.show()


# %%
def jii_plot(csv_name):

    iter_df = pd.read_csv(csv_name)
    iter_df = iter_df.reset_index(names="iteration_num")

    # Get vector values
    y_columns = [col for col in iter_df.columns if "x_" in col]

    # Create stacked subplots
    fig, axes = plt.subplots(
        len(y_columns), 1, figsize=(8, 4 * len(y_columns)), sharex=True
    )
    # Ensure axes is iterable even if only one subplot
    if len(y_columns) == 1:
        axes = [axes]

    i = 0
    for ax, col in zip(axes, y_columns):
        sns.scatterplot(
            data=iter_df, x="iteration_num", y=col, ax=ax, hue="f_x", size="convergence"
        )
        ax.set_title(f"Scatter plot of {col} vs iter_n")
        ax.grid(True)
        if i > 0:
            ax.legend_.remove()
        i += 1

    # plt.tight_layout()
    plt.show()


# %%
def th_approx_and_info(eq, setup):

    if setup["r0"] is None:
        R_0, Z_0 = eq.effective_centre()
    else:
        R_0, Z_0 = setup["r0"], setup["z0"]

    th_params_ref = toroidal_harmonic_grid_and_coil_setup(
        eq=eq,
        R_0=R_0,
        Z_0=Z_0,
        tau_limit=setup["tau_limit"],
        min_tau_value=setup["min_tau_value"],
    )

    result = toroidal_harmonic_approximation(
        eq=eq,
        th_params=th_params_ref,
        psi_norm=setup["psi_norm"],
        n_degrees_of_freedom=setup["n_degrees_of_freedom"],
        max_harmonic_mode=setup["max_harmonic_mode"],
        plasma_mask=setup["plasma_mask"],
    )

    f, ax = plot_toroidal_harmonic_approximation(
        eq=eq, th_params=th_params_ref, result=result, psi_norm=setup["psi_norm"]
    )
    eq.coilset.plot(ax=ax)
    ax.set_title("Comparison of bluemira coilset psi to TH approx.")
    plt.show()

    print("Modes")
    print("------")
    print(f"sin m = {result.sin_m}")
    print(f"cos m = {result.cos_m}")
    print(" ")
    print("Mode Amplitudes")
    print("------")
    print(f"sin amps  = {result.sin_amplitudes}")
    print(f"cos amps = {result.cos_amplitudes}")

    return result


# %%
# TH Approximation Set Up
setup = {
    "tau_limit": TauLimit.MANUAL,
    "min_tau_value": 1.0,  # 1.15,
    "psi_norm": 1.0,
    "n_degrees_of_freedom": 4,
    "max_harmonic_mode": 5,
    "plasma_mask": False,
    "r0": 7.0,  # 7.25, #7.0,
    "z0": -0.1,  # -0.4, #-0.1,
}

# %%
# Look at TH from input eq
# th_result = th_approx_and_info(super_eq, setup)

# %%
# New eq
coilset = deepcopy(super_eq.coilset)
passive = coilset._get_coiltype("NONE")
for coil in passive:
    coilset.remove_coil(coil.name)
    coils = []
for name in coilset.name:
    new_coil = Coil(
        coilset[name].x,
        coilset[name].z,
        current=0,
        dx=coilset[name].dx / 2.0,
        dz=coilset[name].dx / 2.0,
        ctype=coilset[name].ctype,
        name=name,
    )
    coils.append(new_coil)
new_coilset = CoilSet(*coils)
new_coilset.dx
# grid = deepcopy(super_eq.grid)
grid = Grid(
    np.min(super_eq.grid.x),
    np.max(super_eq.grid.x),
    -15.0,  # np.min(super_eq.grid.z),
    np.max(super_eq.grid.z),
    150,
    300,
)
profiles = deepcopy(super_eq.profiles)
eq = Equilibrium(new_coilset, grid, profiles, psi=None)

# %%
# Isoflux Points
fs_lst = find_flux_surfs(
    super_eq.grid.x,
    super_eq.grid.z,
    super_eq.psi(),
    setup["psi_norm"],
)
coords = [Coordinates({"x": fs.T[0], "z": fs.T[1]}) for fs in fs_lst]
fs = coords[0]
x_legs = fs.x[fs.z < super_eq._x_points[0].z][0::50]
z_legs = fs.z[fs.z < super_eq._x_points[0].z][0::50]
x_sep = fs.x[fs.z > super_eq._x_points[0].z][0::10]
z_sep = fs.z[fs.z > super_eq._x_points[0].z][0::10]

arg_inner = np.argmin(x_sep)
ref_x, ref_z = x_sep[arg_inner], z_sep[arg_inner]

leg_constraint = IsofluxConstraint(
    x_legs,
    z_legs,
    ref_x,
    ref_z,
    tolerance=1e-3,
)

core_constraint = IsofluxConstraint(
    x_sep,
    z_sep,
    ref_x,
    ref_z,
    tolerance=1e-3,
)

new_x_legs = [
    3.6205098,
    4.2,
    5.2,
    6.28272981,
    7.57476903,
    8.57632312,
    9.7729805,
    11.31866295,
    13.26935116,
]
new_z_legs = [
    -10.0,
    -8.39598997,
    -6.34085213,
    -5.43863433,
    -5.73934837,
    -7.23030816,
    -8.53195282,
    -9.46588959,
    -10.0,
]

modified_leg_constraint = IsofluxConstraint(
    new_x_legs,
    new_z_legs,
    ref_x,
    ref_z,
    tolerance=1e-3,
)

f, ax = plt.subplots()
ax.plot(fs.x, fs.z)
ax.scatter(x_legs, z_legs)
ax.scatter(x_sep, z_sep)
plt.show()

# %%
opts, xpts = super_eq.get_OX_points()
x_point_constraint = FieldNullConstraint(
    xpts[0].x,
    xpts[0].z,
    tolerance=1e-3,
)
o_point_constraint = FieldNullConstraint(opts[0].x, opts[0].z, tolerance=1e-3)

# %%
_, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
ax.plot(fs.x, fs.z, color="red")
core_constraint.plot(ax=ax)
leg_constraint.plot(ax=ax)
modified_leg_constraint.plot(ax=ax)
plt.show()

# %%
opt_eq = deepcopy(eq)
current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
    opt_eq,
    MagneticConstraintSet([
        core_constraint,
        # leg_constraint,
        # x_point_constraint,
        # o_point_constraint,
    ]),
    gamma=1e-7,
)
# Remove data before new run
# Path.unlink("iterations.csv")
program = PicardIterator(
    opt_eq,
    current_opt_problem,
    convergence=DudsonConvergence(1e-6),
    fixed_coils=True,
    relaxation=0.0,
)
program()
_, ax = plt.subplots()
opt_eq.plot(ax=ax)
opt_eq.coilset.plot(ax=ax)
plt.show()

# %%
# Rename csv
Path("data.csv").rename("initial_opt_iter_data.csv")
# Pick up the csv and plot
# jii_plot("initial_opt_iter_data.csv")

# %%
# Meshing
# for name, current in zip(
#     opt_eq.coilset.name,
#     opt_eq.coilset.current,
#     strict=False
# ):
#     opt_eq.coilset[name].resize(current)
#     opt_eq.coilset[name].fix_size()
#     opt_eq.coilset[name].discretisation = 0.3
# _, ax = plt.subplots()
# opt_eq.plot(ax=ax)
# opt_eq.coilset.plot(ax=ax)
# plt.show()

# %%
# Contol Coils - only PF
# pf_coils = opt_eq.coilset.get_coiltype("PF")
# opt_eq.coilset.control = pf_coils.name

# %%
# TH Approximation
# Use same setup as above
th_result = th_approx_and_info(super_eq, setup)

# %%
from bluemira.equilibria.find import in_zone

c = (
    np.min(th_result.th_params.R)
    + (np.max(th_result.th_params.R) - np.min(th_result.th_params.R)) / 2
)
r = (np.max(th_result.th_params.R) - np.min(th_result.th_params.R)) / 2
c2 = (
    np.min(th_result.th_params.Z)
    + (np.max(th_result.th_params.Z) - np.min(th_result.th_params.Z)) / 2
)
theta = (np.arange(1000) / 1000) * 2 * np.pi
x = c + r * np.sin(theta)
y = c2 + r * np.cos(theta)
plt.plot(x, y)

mask = in_zone(grid.x, grid.z, np.array([x, y]).T)
mask -= 1
mask *= -1

# %%
th_opt_eq = deepcopy(opt_eq)
th_opt_eq.coilset.control = th_result.th_params.th_coil_names

# %%
th_constraint = ToroidalHarmonicConstraint(
    th_result=th_result,
    constraint_type="equality",
    relative_tolerance_sin=1e-3,  # ??
    relative_tolerance_cos=1e-3,  # ??
    weights=1.0,
)

# %%
current_opt_problem = TikhonovCurrentCOP(
    th_opt_eq,
    targets=MagneticConstraintSet([
        th_constraint,
        # core_constraint,
        modified_leg_constraint,
        # leg_constraint,
    ]),
    gamma=1e-12,  # ??
    opt_algorithm="COBYLA",
    max_currents=3e10,
    constraints=[
        # th_constraint,
        # x_point_constraint,
        # o_point_constraint,
        x_point_constraint,
    ],
)

# %%
# Remove data before new run
# Path.unlink("iterations.csv")
program = PicardIterator(
    th_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=HailMaryConvergence(mask=mask, limit=1e-3),
    # convergence=DudsonConvergence(limit=1e-4),#1e-6),
    relaxation=0.0,
    maxiter=100,
)
program()
_, ax = plt.subplots()
th_opt_eq.plot(ax=ax)
th_opt_eq.coilset.plot(ax=ax)
ax.plot(fs.x, fs.z, color="red")
modified_leg_constraint.plot(ax=ax)
plt.show()

# %%
# Rename csv
# os.rename("data.csv", "th_opt_iter_data.csv")
Path("data.csv").rename("th_opt_iter_data_zero_current_start.csv")

# Pick up the csv and plot
# jii_plot("th_opt_iter_data.csv")
