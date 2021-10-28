# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
A simplified 2-D solver for calculating charged particle heat loads.
"""

import numpy as np
import matplotlib.pyplot as plt

from bluemira.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry._deprecated_base import Plane
from bluemira.geometry._deprecated_tools import (
    loop_plane_intersect,
)
from bluemira.geometry._deprecated_loop import Loop
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.flux_surfaces import OpenFluxSurface
from bluemira.radiation_transport.error import AdvectionTransportError


__all__ = ["ChargedParticleSolver"]


class ChargedParticleSolver:
    """
    A simplified charged particle transport model along open field lines.
    """

    # fmt: off
    default_params = [
        ["fw_p_sol_near", "near scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL at the outboard", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL at the outboard", 0.05, "m", None, "Input"],
        ["fw_lambda_q_near_ib", "Lambda q near SOL at the inboard", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_ib", "Lambda q far SOL at the inboard", 0.05, "m", None, "Input"],
        ["f_outer_target", "Fraction of SOL power deposited on the outer target(s)", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Fraction of SOL power deposited on the inner target(s)", 0.25, "N/A", None, "Input"],
        ["f_upper_target", "Fraction of SOL power deposited on the upper targets. DN only", 0.5, "N/A", None, "Input"],
        ["f_lower_target", "Fraction of SOL power deposited on the lower target, DN only", 0.5, "N/A", None, "Input"],
    ]
    # fmt: on

    def __init__(self, config, equilibrium, **kwargs):
        self.params = ParameterFrame(self.default_params)
        self.params.update_kw_parameters(config)
        self._check_params()

        # Midplane spatial resolution between flux surfaces
        self.dx_mp = kwargs.get("dx_mp", 0.001)

        self.eq = equilibrium

        # Constructors
        self.first_wall = None
        self.flux_surfaces_ob_lfs = None
        self.flux_surfaces_ob_hfs = None
        self.flux_surfaces_ib_lfs = None
        self.flux_surfaces_ib_hfs = None
        self.x_sep_omp = None
        self.x_sep_imp = None
        self.result = None

        # Pre-processing
        o_points, _ = self.eq.get_OX_points()
        self._o_point = o_points[0]
        z = self._o_point.z
        self._yz_plane = Plane([0, 0, z], [1, 0, z], [1, 1, z])

    @property
    def flux_surfaces(self):
        """
        All flux surfaces in the ChargedParticleSolver.

        Returns
        -------
        flux_surfaces: List[PartialOpenFluxSurface]
        """
        flux_surfaces = []
        for group in [
            self.flux_surfaces_ob_lfs,
            self.flux_surfaces_ob_hfs,
            self.flux_surfaces_ib_lfs,
            self.flux_surfaces_ib_hfs,
        ]:
            if group:
                flux_surfaces.extend(group)
        return flux_surfaces

    def _check_params(self):
        """
        Check input fractions for validity.
        """
        if self.params.f_outer_target + self.params.f_inner_target != 1.0:
            raise AdvectionTransportError(
                "Inner / outer fractions should sum to 1.0:\n"
                f"{self.params.f_outer_target} + {self.params.f_inner_target} != 1.0:"
            )
        if self.params.f_upper_target + self.params.f_lower_target != 1.0:
            raise AdvectionTransportError(
                "Upper / lower fractions should sum to 1.0:\n"
                f"{self.params.f_upper_target} + {self.params.f_lower_target} != 1.0:"
            )

    @staticmethod
    def _process_first_wall(first_wall):
        """
        Force working first wall geometry to be closed and counter-clockwise.
        """
        first_wall = first_wall.copy()

        if not first_wall.ccw:
            bluemira_warn(
                "First wall should be oriented counter-clockwise. Reversing it."
            )
            first_wall.reverse()

        if not first_wall.closed:
            bluemira_warn("First wall should be a closed geometry. Closing it.")
            first_wall.close()
        return first_wall

    @staticmethod
    def _get_arrays(flux_surfaces):
        """
        Get arrays of flux surface values.
        """
        x_mp = np.array([fs.x_start for fs in flux_surfaces])
        z_mp = np.array([fs.z_start for fs in flux_surfaces])
        x_fw = np.array([fs.x_end for fs in flux_surfaces])
        z_fw = np.array([fs.z_end for fs in flux_surfaces])
        alpha = np.array([fs.alpha for fs in flux_surfaces])
        return x_mp, z_mp, x_fw, z_fw, alpha

    def _get_sep_out_intersection(self, outboard=True):
        """
        Find the middle and maximum outboard mid-plane psi norm values
        """
        yz_plane = self._yz_plane
        o_point = self._o_point
        separatrix = self.eq.get_separatrix()

        if not isinstance(separatrix, Loop):
            sep1_intersections = loop_plane_intersect(separatrix[0], yz_plane)
            sep2_intersections = loop_plane_intersect(separatrix[1], yz_plane)
            sep1_arg = np.argmin(np.abs(sep1_intersections.T[0] - o_point.x))
            sep2_arg = np.argmin(np.abs(sep2_intersections.T[0] - o_point.x))
            x_sep1_mp = sep1_intersections.T[0][sep1_arg]
            x_sep2_mp = sep2_intersections.T[0][sep2_arg]
            if outboard:
                x_sep_mp = x_sep1_mp if x_sep1_mp > x_sep2_mp else x_sep2_mp
            else:
                x_sep_mp = x_sep1_mp if x_sep1_mp < x_sep2_mp else x_sep2_mp
        else:
            sep_intersections = loop_plane_intersect(separatrix, yz_plane)
            sep_arg = np.argmin(np.abs(sep_intersections.T[0] - o_point.x))
            x_sep_mp = sep_intersections.T[0][sep_arg]

        out_intersections = loop_plane_intersect(self.first_wall, yz_plane)
        if outboard:
            x_out_mp = np.max(out_intersections.T[0])
        else:
            x_out_mp = np.min(out_intersections.T[0])

        return x_sep_mp, x_out_mp

    def _make_flux_surfaces(self, x, z):
        """
        Make individual PartialOpenFluxSurfaces through a point.
        """
        loop = find_flux_surface_through_point(
            self.eq.x, self.eq.z, self.eq.psi(), x, z, self.eq.psi(x, z)
        )
        loop = Loop(loop[0], z=loop[1])
        f_s = OpenFluxSurface(loop)
        lfs, hfs = f_s.split(self._o_point, plane=self._yz_plane)
        return lfs, hfs

    def _make_flux_surfaces_ob(self):
        """
        Make the flux surfaces on the outboard.
        """
        self.x_sep_omp, x_out_omp = self._get_sep_out_intersection(outboard=True)

        self.flux_surfaces_ob_lfs = []
        self.flux_surfaces_ob_hfs = []

        x = self.x_sep_omp + self.dx_mp
        while x < x_out_omp:
            lfs, hfs = self._make_flux_surfaces(x, self._o_point.z)
            self.flux_surfaces_ob_lfs.append(lfs)
            self.flux_surfaces_ob_hfs.append(hfs)
            x += self.dx_mp

    def _make_flux_surfaces_ib(self):
        """
        Make the flux surfaces on the inboard.
        """
        self.x_sep_imp, x_out_imp = self._get_sep_out_intersection(outboard=False)

        self.flux_surfaces_ib_lfs = []
        self.flux_surfaces_ib_hfs = []
        x = self.x_sep_imp - self.dx_mp
        while x > x_out_imp:
            lfs, hfs = self._make_flux_surfaces(x, self._o_point.z)
            self.flux_surfaces_ib_lfs.append(lfs)
            self.flux_surfaces_ib_hfs.append(hfs)
            x -= self.dx_mp

    def analyse(self, first_wall):
        """
        Perform the calculation to obtain charged particle heat fluxes on the
        the specified first_wall

        Parameters
        ----------
        first_wall: Loop
            The closed first wall geometry on which to calculate the heat flux

        Returns
        -------
        x: np.array
            The x coordinates of the flux surface intersections
        z: np.array
            The z coordinates of the flux surface intersections
        heat_flux: np.array
            The perpendicular heat fluxes at the intersection points [MW/m^2]
        """
        self.first_wall = self._process_first_wall(first_wall)

        if self.eq.is_double_null:
            x, z, hf = self._analyse_DN(first_wall)
        else:
            x, z, hf = self._analyse_SN(first_wall)

        self.result = x, z, hf
        return x, z, hf

    def _analyse_SN(self, first_wall):
        """
        Calculation for the case of single nulls.
        """
        self._make_flux_surfaces_ob()

        # Find the intersections of the flux surfaces with the first wall
        for flux_surface in self.flux_surfaces:
            flux_surface.clip(first_wall)

        x_omp, z_omp, x_lfs_inter, z_lfs_inter, alpha_lfs = self._get_arrays(
            self.flux_surfaces_ob_lfs
        )
        _, _, x_hfs_inter, z_hfs_inter, alpha_hfs = self._get_arrays(
            self.flux_surfaces_ob_hfs
        )

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        Bp_omp = self.eq.Bp(x_omp, z_omp)
        Bt_omp = self.eq.Bt(x_omp)
        B_omp = np.hypot(Bp_omp, Bt_omp)

        # Parallel heat flux at the outboard midplane
        q_par_omp = self._q_par(x_omp, dx_omp, B_omp, Bp_omp)

        # Calculate values at intersections
        Bp_lfs = self.eq.Bp(x_lfs_inter, z_lfs_inter)
        Bp_hfs = self.eq.Bp(x_hfs_inter, z_hfs_inter)

        # Calculate parallel heat fluxes at the intersections
        # Note that flux expansion terms cancelate down to this
        q_par_lfs = q_par_omp * Bp_lfs / B_omp
        q_par_hfs = q_par_omp * Bp_hfs / B_omp

        # Calculate perpendicular heat fluxes
        heat_flux_lfs = self.params.f_outer_target * q_par_lfs * np.sin(alpha_lfs)
        heat_flux_hfs = self.params.f_inner_target * q_par_hfs * np.sin(alpha_hfs)

        # Correct power (energy conservation)
        q_omp_int = 2 * np.pi * np.sum(q_par_omp / (B_omp / Bp_omp) * self.dx_mp * x_omp)
        f_correct_power = (
            self.params.fw_p_sol_near + self.params.fw_p_sol_far
        ) / q_omp_int
        return (
            np.append(x_lfs_inter, x_hfs_inter),
            np.append(z_lfs_inter, z_hfs_inter),
            f_correct_power * np.append(heat_flux_lfs, heat_flux_hfs),
        )

    def _analyse_DN(self, first_wall):
        """
        Calculation for the case of double nulls.
        """
        self._make_flux_surfaces_ob()
        self._make_flux_surfaces_ib()

        # Find the intersections of the flux surfaces with the first wall
        for flux_surface in self.flux_surfaces:
            flux_surface.clip(first_wall)

        (
            x_omp,
            z_omp,
            x_lfs_down_inter,
            z_lfs_down_inter,
            alpha_lfs_down,
        ) = self._get_arrays(self.flux_surfaces_ob_lfs)
        _, _, x_lfs_up_inter, z_lfs_up_inter, alpha_lfs_up = self._get_arrays(
            self.flux_surfaces_ob_hfs
        )
        (
            x_imp,
            z_imp,
            x_hfs_down_inter,
            z_hfs_down_inter,
            alpha_hfs_down,
        ) = self._get_arrays(self.flux_surfaces_ib_lfs)
        _, _, x_hfs_up_inter, z_hfs_up_inter, alpha_hfs_up = self._get_arrays(
            self.flux_surfaces_ib_hfs
        )

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        Bp_omp = self.eq.Bp(x_omp, z_omp)
        Bt_omp = self.eq.Bt(x_omp)
        B_omp = np.hypot(Bp_omp, Bt_omp)

        # Calculate values at IMP
        dx_imp = abs(x_imp - self.x_sep_imp)
        Bp_imp = self.eq.Bp(x_imp, z_imp)
        Bt_imp = self.eq.Bt(x_imp)
        B_imp = np.hypot(Bp_imp, Bt_imp)

        # Parallel heat flux at the outboard and inboard midplane
        q_par_omp = self.params.f_outer_target * self._q_par(
            x_omp, dx_omp, B_omp, Bp_omp
        )
        q_par_imp = self.params.f_inner_target * self._q_par(
            x_imp, dx_imp, B_imp, Bp_imp, outboard=False
        )

        # Calculate poloidal field at intersections
        Bp_lfs_down = self.eq.Bp(x_lfs_down_inter, z_lfs_down_inter)
        Bp_lfs_up = self.eq.Bp(x_lfs_up_inter, z_lfs_up_inter)
        Bp_hfs_down = self.eq.Bp(x_hfs_down_inter, z_hfs_down_inter)
        Bp_hfs_up = self.eq.Bp(x_hfs_up_inter, z_hfs_up_inter)

        # Calculate parallel heat fluxes at the intersections
        # Note that flux expansion terms cancelate down to this
        q_par_lfs_down = q_par_omp * Bp_lfs_down / B_omp
        q_par_lfs_up = q_par_omp * Bp_lfs_up / B_omp
        q_par_hfs_down = q_par_imp * Bp_hfs_down / B_imp
        q_par_hfs_up = q_par_imp * Bp_hfs_up / B_imp

        # Calculate perpendicular heat fluxes
        heat_flux_lfs_down = (
            self.params.f_lower_target * q_par_lfs_down * np.sin(alpha_lfs_down)
        )
        heat_flux_lfs_up = (
            self.params.f_upper_target * q_par_lfs_up * np.sin(alpha_lfs_up)
        )
        heat_flux_hfs_down = (
            self.params.f_lower_target * q_par_hfs_down * np.sin(alpha_hfs_down)
        )
        heat_flux_hfs_up = (
            self.params.f_upper_target * q_par_hfs_up * np.sin(alpha_hfs_up)
        )

        # Correct power (energy conservation)
        q_omp_int = 2 * np.pi * np.sum(q_par_omp / (B_omp / Bp_omp) * self.dx_mp * x_omp)
        q_imp_int = 2 * np.pi * np.sum(q_par_imp / (B_imp / Bp_imp) * self.dx_mp * x_imp)

        total_power = self.params.fw_p_sol_near + self.params.fw_p_sol_far
        f_correct_power_ob = (self.params.f_outer_target * total_power) / q_omp_int
        f_correct_power_ib = (self.params.f_inner_target * total_power) / q_imp_int

        return (
            np.concatenate(
                [x_lfs_down_inter, x_lfs_up_inter, x_hfs_down_inter, x_hfs_up_inter]
            ),
            np.concatenate(
                [z_lfs_down_inter, z_lfs_up_inter, z_hfs_down_inter, z_hfs_up_inter]
            ),
            np.concatenate(
                [
                    f_correct_power_ob * self.params.f_lower_target * heat_flux_lfs_down,
                    f_correct_power_ob * self.params.f_upper_target * heat_flux_lfs_up,
                    f_correct_power_ib * self.params.f_lower_target * heat_flux_hfs_down,
                    f_correct_power_ib * self.params.f_upper_target * heat_flux_hfs_up,
                ]
            ),
        )

    def _q_par(self, x, dx, B, Bp, outboard=True):
        """
        Calculate the parallel heat flux at the midplane.
        """
        p_sol_near = self.params.fw_p_sol_near
        p_sol_far = self.params.fw_p_sol_far
        if outboard:
            lq_near = self.params.fw_lambda_q_near
            lq_far = self.params.fw_lambda_q_far
        else:
            lq_near = self.params.fw_lambda_q_near_ib
            lq_far = self.params.fw_lambda_q_far_ib
        return (
            (
                p_sol_near * np.exp(-dx / lq_near) / lq_near
                + p_sol_far * np.exp(-dx / lq_far) / lq_far
            )
            * B
            / (Bp * 2 * np.pi * x)
        )

    def plot(self, ax=None):
        """
        Plot the ChargedParticleSolver results.
        """
        if ax is None:
            ax = plt.gca()

        self.first_wall.plot(ax, linewidth=0.1, fill=False)
        separatrix = self.eq.get_separatrix()

        if isinstance(separatrix, Loop):
            separatrix = [separatrix]

        for sep in separatrix:
            sep.plot(ax, linewidth=0.12)

        for f_s in self.flux_surfaces:
            f_s.plot(ax, linewidth=0.01)

        cm = ax.scatter(
            self.result[0],
            self.result[1],
            c=self.result[2],
            s=2,
            zorder=40,
            cmap="plasma",
        )
        f = plt.gcf()
        f.colorbar(cm, label="MW/m^2")