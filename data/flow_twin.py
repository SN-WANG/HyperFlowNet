# 3D axisymmetric flow twin rendering with PyVista
# Author: Shengning Wang

import os
import subprocess
from pathlib import Path
from typing import Sequence, Tuple

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap
from scipy.spatial import cKDTree
from torch import Tensor
from tqdm.auto import tqdm

from data.flow_vis import _CMAP, _FLUENT_SEQ, _channel_role
from utils.hue_logger import hue, logger


class FlowTwin:
    """
    Render a 3D axisymmetric cutaway flow twin from HyperFlowNet prediction.
    """

    def __init__(self, output_dir: str | Path, channel_names: Sequence[str]) -> None:
        """
        Initialize the 3D flow twin renderer.

        Args:
            output_dir (str | Path): Directory for rendered MP4 files.
            channel_names (Sequence[str]): Ordered field names.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ch_names = list(channel_names)

        pv.set_plot_theme("document")

    # ============================================================
    # Geometry
    # ============================================================

    def _section_points(self, coords: Tensor) -> np.ndarray:
        """
        Convert axisymmetric coordinates to an x-r section.

        Args:
            coords (Tensor): Axisymmetric coordinates. (N, 2).

        Returns:
            np.ndarray: Section points. (N, 3).
        """
        pts = coords.detach().cpu().numpy().astype(np.float32)
        radius = np.maximum(pts[:, 1], 0.0) * 1.35
        return np.stack([pts[:, 0], radius, np.zeros_like(radius)], axis=1)

    def _section_mesh(self, points: np.ndarray) -> pv.PolyData:
        """
        Build the triangulated 2D section mesh used for the internal cut plane.

        Args:
            points (np.ndarray): Section points. (N, 3).

        Returns:
            pv.PolyData: Triangulated section mesh.
        """
        cloud = pv.PolyData(points)
        cloud.point_data["node_id"] = np.arange(points.shape[0], dtype=np.int64)

        tree = cKDTree(points[:, :2])
        dd, _ = tree.query(points[:, :2], k=2)
        alpha = float(np.mean(dd[:, 1])) * 2.5
        return cloud.delaunay_2d(alpha=alpha).triangulate()

    def _rotate_section(self, mesh: pv.PolyData, angle_deg: float) -> pv.PolyData:
        """
        Rotate the internal 2D section into the 3D pipe volume.

        Args:
            mesh (pv.PolyData): Axisymmetric section mesh.
            angle_deg (float): Azimuthal angle in degrees.

        Returns:
            pv.PolyData: Rotated internal section.
        """
        section = mesh.copy()
        theta = np.deg2rad(angle_deg)
        points = section.points.copy()
        radius = points[:, 1].copy()
        points[:, 1] = radius * np.cos(theta)
        points[:, 2] = radius * np.sin(theta)
        section.points = points
        return section

    def _boundary_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        """
        Extract the exposed section boundary.

        Args:
            mesh (pv.PolyData): Axisymmetric section mesh.

        Returns:
            pv.PolyData: Boundary line mesh.
        """
        return mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )

    def _outer_boundary_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        """
        Extract non-axis boundary edges for the metallic outer shell.

        Args:
            mesh (pv.PolyData): Axisymmetric section mesh.

        Returns:
            pv.PolyData: Outer boundary line mesh.
        """
        boundary = self._boundary_mesh(mesh)
        lines = boundary.lines.reshape(-1, 3)
        radius = np.abs(boundary.points[:, 1])
        keep = np.max(radius[lines[:, 1:]], axis=1) > 1.0e-4
        return pv.PolyData(boundary.points, lines[keep].ravel()).clean()

    def _pipe_shell(self, mesh: pv.PolyData) -> pv.PolyData:
        """
        Revolve the outer boundary by 270 degrees to build the quarter-cut shell.

        Args:
            mesh (pv.PolyData): Axisymmetric section mesh.

        Returns:
            pv.PolyData: Quarter-cut pipe shell.
        """
        boundary = self._rotate_section(self._outer_boundary_mesh(mesh), 90.0)
        return boundary.extrude_rotate(
            resolution=96,
            angle=270.0,
            rotation_axis=(1, 0, 0),
            capping=False,
        )

    def _camera(self, plotter: pv.Plotter, mesh: pv.PolyData) -> None:
        """
        Set a perspective camera looking into the quarter-cut opening.

        Args:
            plotter (pv.Plotter): Active plotter.
            mesh (pv.PolyData): Visible 3D pipe mesh.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        cx, cy, cz = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)
        length = (x_max - x_min) or 1.0
        diameter = max(y_max - y_min, z_max - z_min, 1.0)

        plotter.camera.focal_point = (cx, cy, cz - 0.18 * diameter)
        plotter.camera.position = (
            cx + 1.08 * length,
            cy + 2.05 * diameter,
            cz + 1.20 * diameter,
        )
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.view_angle = 28.0
        plotter.camera.parallel_projection = False
        plotter.camera.zoom(0.94)
        plotter.reset_camera_clipping_range()

    # ============================================================
    # Scalars and rendering
    # ============================================================

    def _clim(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute robust scalar limits for one temporal field.

        Args:
            data (np.ndarray): Scalar field sequence. (T, N).

        Returns:
            Tuple[float, float]: Scalar limits.
        """
        lo = float(np.percentile(data.ravel(), 2))
        hi = float(np.percentile(data.ravel(), 98))
        if abs(hi - lo) < 1e-9:
            center = 0.5 * (lo + hi)
            lo, hi = center - 1e-6, center + 1e-6
        return lo, hi

    def _signed_clim(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute symmetric scalar limits for signed derived fields.

        Args:
            data (np.ndarray): Scalar field sequence. (T, N).

        Returns:
            Tuple[float, float]: Symmetric scalar limits.
        """
        lo, hi = self._clim(data)
        vmax = max(abs(lo), abs(hi))
        return -vmax, vmax

    def _value_cmap(self, ch_idx: int, clim: Tuple[float, float]) -> Colormap:
        """
        Pick the FlowVis-style scalar colormap for one channel.

        Args:
            ch_idx (int): Channel index.
            clim (Tuple[float, float]): Scalar limits.

        Returns:
            Colormap: Matplotlib colormap.
        """
        role = _channel_role(ch_idx, 2)
        if role == "velocity" and clim[0] >= 0.0:
            return _FLUENT_SEQ
        return _CMAP[role]

    def _channel_clim(self, data: np.ndarray, ch_idx: int) -> Tuple[float, float]:
        """
        Compute channel limits and keep signed velocity fields symmetric.

        Args:
            data (np.ndarray): Scalar field sequence. (T, N).
            ch_idx (int): Channel index.

        Returns:
            Tuple[float, float]: Scalar limits.
        """
        lo, hi = self._clim(data)
        if _channel_role(ch_idx, 2) == "velocity" and lo < 0.0 < hi:
            lo, hi = self._signed_clim(data)
        return lo, hi

    def _gradient_stencil(self, coords: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build local least-squares gradient weights on the irregular 2D mesh.

        Args:
            coords (Tensor): Axisymmetric coordinates. (N, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Neighbor indices (N, K) and derivative weights. (N, K, 2).
        """
        pts = coords.detach().cpu().numpy().astype(np.float32)
        _, idx = cKDTree(pts).query(pts, k=25)
        idx = idx[:, 1:]
        weights = np.empty((idx.shape[0], idx.shape[1], 2), dtype=np.float32)

        for node_idx in range(idx.shape[0]):
            delta = pts[idx[node_idx]] - pts[node_idx]
            dist = np.linalg.norm(delta, axis=1) + 1.0e-12
            weights[node_idx] = (np.linalg.pinv(delta / dist[:, None]) / dist[None, :]).T

        return idx.astype(np.int64), weights

    def _derivative(self, data: np.ndarray, idx: np.ndarray, weights: np.ndarray, axis_idx: int) -> np.ndarray:
        """
        Apply a precomputed least-squares derivative stencil.

        Args:
            data (np.ndarray): Scalar field sequence. (T, N).
            idx (np.ndarray): Neighbor indices. (N, K).
            weights (np.ndarray): Derivative weights. (N, K, 2).
            axis_idx (int): Coordinate derivative axis.

        Returns:
            np.ndarray: Spatial derivative sequence. (T, N).
        """
        deriv = np.empty_like(data)
        for start in range(0, data.shape[1], 512):
            end = min(start + 512, data.shape[1])
            local_weights = weights[start:end, :, axis_idx]
            deriv[:, start:end] = (
                np.sum(data[:, idx[start:end]] * local_weights[None, :, :], axis=2)
                - data[:, start:end] * np.sum(local_weights, axis=1)[None, :]
            )
        return deriv

    def _vorticity(self, pred: Tensor, coords: Tensor) -> np.ndarray:
        """
        Compute the axisymmetric no-swirl circumferential vorticity.

        Args:
            pred (Tensor): Predicted flow sequence. (T, N, C).
            coords (Tensor): Axisymmetric coordinates. (N, 2).

        Returns:
            np.ndarray: Vorticity sequence. (T, N).
        """
        data = pred.detach().cpu().numpy().astype(np.float32)
        idx, weights = self._gradient_stencil(coords)
        vx = data[:, :, self.ch_names.index("Vx")]
        vy = data[:, :, self.ch_names.index("Vy")]
        return self._derivative(vy, idx, weights, 0) - self._derivative(vx, idx, weights, 1)

    def _large_bulb_mask(self, coords: Tensor) -> np.ndarray:
        """
        Select the visible core of the larger bulb for vorticity color scaling.

        Args:
            coords (Tensor): Axisymmetric coordinates. (N, 2).

        Returns:
            np.ndarray: Boolean node mask. (N,).
        """
        pts = coords.detach().cpu().numpy().astype(np.float32)
        x = pts[:, 0]
        radius = np.maximum(pts[:, 1], 0.0)
        r_max = float(radius.max())
        center_x = float(np.mean(x[radius > 0.96 * r_max]))
        return (np.abs(x - center_x) < 0.92 * r_max) & (radius > 0.04 * r_max)

    def _vorticity_clim(self, data: np.ndarray, coords: Tensor) -> Tuple[float, float]:
        """
        Compute a clipped vorticity color range focused on the larger bulb.

        Args:
            data (np.ndarray): Vorticity sequence. (T, N).
            coords (Tensor): Axisymmetric coordinates. (N, 2).

        Returns:
            Tuple[float, float]: Symmetric vorticity limits.
        """
        vmax = float(np.percentile(np.abs(data[:, self._large_bulb_mask(coords)]).ravel(), 95))
        return -vmax, vmax

    def _render_field(
        self,
        pred: Tensor,
        coords: Tensor,
        field_name: str,
    ) -> Tuple[np.ndarray, str, str, Tuple[float, float], Colormap]:
        """
        Resolve one requested flow-twin scalar field.

        Args:
            pred (Tensor): Predicted flow sequence. (T, N, C).
            coords (Tensor): Axisymmetric coordinates. (N, 2).
            field_name (str): Field name, one of Vx, Vy, P, T, or Vorticity.

        Returns:
            Tuple[np.ndarray, str, str, Tuple[float, float], Colormap]: Field, label, file tag, limits, and colormap.
        """
        if field_name.lower() in {"vorticity", "omega"}:
            field = self._vorticity(pred, coords)
            return field, "Vorticity", "vorticity", self._vorticity_clim(field, coords), _CMAP["velocity"]

        channel_lookup = {name.lower(): idx for idx, name in enumerate(self.ch_names)}
        ch_idx = channel_lookup[field_name.lower()]
        field = pred.detach().cpu().numpy().astype(np.float32)[:, :, ch_idx]
        channel_name = self.ch_names[ch_idx]
        clim = self._channel_clim(field, ch_idx)
        return field, channel_name, channel_name.lower(), clim, self._value_cmap(ch_idx, clim)

    def _sbar_args(self, channel_name: str) -> dict:
        """
        Return FlowVis-like scalar-bar arguments.

        Args:
            channel_name (str): Scalar-bar channel name.

        Returns:
            dict: PyVista scalar-bar layout arguments.
        """
        return {
            "title": channel_name,
            "height": 0.07,
            "width": 0.54,
            "position_x": 0.23,
            "position_y": 0.055,
            "vertical": False,
            "fmt": "%.2e",
            "color": "black",
            "font_family": "arial",
            "title_font_size": 14,
            "label_font_size": 12,
        }

    def _mp4(self, plotter: pv.Plotter, update_fn, seq_len: int, out_path: Path, desc: str) -> None:
        """
        Encode a rendered sequence as MP4 through ffmpeg.

        Args:
            plotter (pv.Plotter): Configured off-screen plotter.
            update_fn: Per-frame update callback.
            seq_len (int): Number of frames.
            out_path (Path): Output path.
            desc (str): Progress-bar label.
        """
        first_frame = plotter.screenshot(return_img=True)
        H, W = first_frame.shape[:2]
        W_enc = W + (W % 2)
        H_enc = H + (H % 2)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-i",
            "pipe:0",
            "-vf",
            f"pad={W_enc}:{H_enc}:0:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(out_path),
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        proc.stdin.write(first_frame[:, :, :3].astype(np.uint8).tobytes())
        for step_idx in tqdm(range(1, seq_len), desc=desc, leave=False):
            update_fn(step_idx)
            plotter.render()
            frame = plotter.screenshot(return_img=True)
            proc.stdin.write(frame[:, :, :3].astype(np.uint8).tobytes())

        proc.stdin.close()
        proc.wait()
        plotter.close()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}. Ensure ffmpeg is on PATH.")

    # ============================================================
    # Public interface
    # ============================================================

    def render(
        self,
        pred: Tensor,
        coords: Tensor,
        label: str,
        num_nodes: int,
        num_params: int,
        field_name: str = "Vy",
    ) -> Path:
        """
        Render one 3D quarter-cut MP4 for a predicted scalar field.

        Args:
            pred (Tensor): Predicted flow sequence. (T, N, C).
            coords (Tensor): Axisymmetric coordinates. (N, 2).
            label (str): Operating-condition label.
            num_nodes (int): Total node count.
            num_params (int): Model parameter count.
            field_name (str): Field name, one of Vx, Vy, P, T, or Vorticity.

        Returns:
            Path: Rendered MP4 path.
        """
        field, field_label, file_tag, clim, cmap = self._render_field(pred, coords, field_name)

        points = self._section_points(coords)
        section_2d = self._section_mesh(points)
        section_ids = section_2d.point_data["node_id"].astype(np.int64)
        section_y = self._rotate_section(section_2d, 0.0)
        section_z = self._rotate_section(section_2d, 90.0)
        rim_y = self._rotate_section(self._boundary_mesh(section_2d), 0.0)
        rim_z = self._rotate_section(self._boundary_mesh(section_2d), 90.0)
        shell = self._pipe_shell(section_2d)

        section_y.point_data["scalar"] = field[0, section_ids]
        section_z.point_data["scalar"] = field[0, section_ids]

        plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        plotter.set_background("white")
        plotter.enable_anti_aliasing("msaa", multi_samples=8)
        plotter.add_mesh(
            shell,
            color=(0.92, 0.93, 0.92),
            opacity=0.96,
            smooth_shading=True,
            show_scalar_bar=False,
            ambient=0.46,
            diffuse=0.58,
            specular=0.72,
            specular_power=72,
        )
        plotter.add_mesh(
            section_y,
            scalars="scalar",
            cmap=cmap,
            clim=clim,
            lighting=False,
            smooth_shading=True,
            scalar_bar_args=self._sbar_args(field_label),
        )
        plotter.add_mesh(
            section_z,
            scalars="scalar",
            cmap=cmap,
            clim=clim,
            lighting=False,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        for rim in (rim_y, rim_z):
            plotter.add_mesh(
                rim,
                color=(0.58, 0.60, 0.60),
                line_width=2.0,
                render_lines_as_tubes=True,
                show_scalar_bar=False,
                specular=0.75,
                specular_power=72,
            )

        title = f"HyperFlowNet (nodes: {num_nodes:,}, params: {num_params:,})"
        plotter.add_text(title, position="upper_edge", font_size=15, color="black", font="arial")
        plotter.add_text(
            f"{field_label} (label {label})",
            position="upper_left",
            font_size=18,
            color="black",
            font="arial",
        )

        for light in (
            pv.Light(position=(2.5, 4.0, 3.2), focal_point=(1.8, 0.0, 0.0), color="white", intensity=0.78),
            pv.Light(position=(-1.6, -2.5, 1.8), focal_point=(1.8, 0.0, 0.0), color="white", intensity=0.48),
        ):
            plotter.add_light(light)
        self._camera(plotter, shell)

        def _update(step_idx: int) -> None:
            section_y.point_data["scalar"] = field[step_idx, section_ids]
            section_z.point_data["scalar"] = field[step_idx, section_ids]

        out_path = self.output_dir / f"{label}_twin_{file_tag}.mp4"
        self._mp4(plotter, _update, field.shape[0], out_path, desc=f"Rendering {label} 3D twin")
        logger.info(f"3D flow twin saved to {hue.g}{out_path}{hue.q}")
        return out_path
