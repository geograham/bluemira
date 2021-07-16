# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Simple HelmholzCage example with different current sources.
"""
import numpy as np
import matplotlib.pyplot as plt
from BLUEPRINT.utilities.plottools import Plot3D
from BLUEPRINT.geometry.parameterisations import princetonD
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.magnetostatics.biot_savart import BiotSavartFilament
from BLUEPRINT.magnetostatics.current_source import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)

# Set up some geometry and key parameters
n_TF = 6
current = 20e6
breadth = 0.5
depth = 1.0
x, z = princetonD(4, 16, 0, npoints=50)

# Make a Biot-Savart filament (which needs to be properly discretised)
n_filaments_x = 2
n_filaments_y = 3
radius = 0.5 * (breadth + depth) / (n_filaments_x * n_filaments_y)

loop = Loop(x=x, z=z)
loop.close()
loop.interpolate(50)

loops = []
filaments = []
dx_offsets = np.linspace(-breadth / 2, breadth / 2, n_filaments_x)
dy_offsets = np.linspace(-depth / 2, depth / 2, n_filaments_y)

for dx in dx_offsets:
    for dy in dy_offsets:
        new_loop = loop.offset(dx)
        new_loop.translate(vector=[0, dy, 0])
        loops.append(new_loop)

biotsavart_circuit = BiotSavartFilament(
    loops, radius=radius, current=current / (n_filaments_x * n_filaments_y)
)

# Make an analytical circuit with a rectangular cross-section
loop = Loop(x=x, z=z)
loop.close()
analytical_circuit = ArbitraryPlanarRectangularXSCircuit(
    loop, breadth=breadth, depth=depth, current=current
)

# Pattern the two circuits
biotsavart_tf_cage = HelmholtzCage(biotsavart_circuit, n_TF=n_TF)
analytical_tf_cage = HelmholtzCage(analytical_circuit, n_TF=n_TF)

# Calculate the fields in the x-y plane
nx, ny = 50, 50
x = np.linspace(0, 18, nx)
y = np.linspace(-18, 0, ny)
xx1, yy = np.meshgrid(x, y, indexing="ij")

biotsavart_xy_fields = np.zeros((nx, ny))
analytical_xy_fields = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        biotsavart_xy_fields[i, j] = np.sqrt(
            np.sum(biotsavart_tf_cage.field([x[i], y[j], 0]) ** 2)
        )
        analytical_xy_fields[i, j] = np.sqrt(
            np.sum(analytical_tf_cage.field([x[i], y[j], 0]) ** 2)
        )

# Calculate the fields in the x-z plane
nx, nz = 50, 50
x = np.linspace(0, 18, nx)
z = np.linspace(0, 14, nz)
xx, zz = np.meshgrid(x, z, indexing="ij")

biotsavart_xz_fields = np.zeros((nx, nz))
analytical_xz_fields = np.zeros((nx, nz))
for i in range(nx):
    for j in range(nz):
        biotsavart_xz_fields[i, j] = np.sqrt(
            np.sum(biotsavart_tf_cage.field([x[i], 0, z[j]]) ** 2)
        )
        analytical_xz_fields[i, j] = np.sqrt(
            np.sum(analytical_tf_cage.field([x[i], 0, z[j]]) ** 2)
        )


def plot_cage_results(cage, xz_fields, xy_fields):
    """
    Plot utility for contours in 3-D projections in matplotlib.
    """
    b_minmax = min(np.amax(xz_fields), np.amax(xy_fields))
    b_max = max(np.amax(xz_fields), np.amax(xy_fields))
    levels = np.linspace(0, b_minmax, 20)
    levels2 = np.linspace(b_minmax, b_max, 10)

    ax = Plot3D()
    # This will make the plot look better once matplotlib PRs are accepted
    ax.computed_zorder = False
    cm = ax.contourf(
        xx1, yy, xy_fields, zdir="z", levels=levels, offset=0, alpha=0.8, zorder=-1
    )
    cm2 = ax.contourf(
        xx1,
        yy,
        xy_fields,
        zdir="z",
        levels=levels2,
        offset=0,
        alpha=0.8,
        zorder=-1,
        cmap="plasma_r",
    )
    ax.contourf(
        xx, xz_fields, zz, zdir="y", levels=levels, offset=0, alpha=0.8, zorder=-1
    )

    #  We need to use 2 colorbars because of 3-D projection shenanigans in matplotlib
    f = plt.gcf()
    f.colorbar(cm, shrink=0.46)
    f.colorbar(cm2, shrink=0.46)
    cage.plot(ax=ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_ylabel("z [m]")


# Plot the two cages and the results in the two planes
plot_cage_results(analytical_tf_cage, analytical_xz_fields, analytical_xy_fields)
plot_cage_results(biotsavart_tf_cage, biotsavart_xz_fields, biotsavart_xy_fields)