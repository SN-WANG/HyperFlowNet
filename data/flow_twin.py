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
    Render a 3D axisymmetric Vy-field digital twin from HyperFlowNet prediction.
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

    def _rotate_section(self, mesh: pv.PolyData) -> pv.PolyData:
        """
        Rotate the internal 2D section into the 3D pipe volume.

        Args:
            mesh (pv.PolyData): Axisymmetric section mesh.

        Returns:
            pv.PolyData: Rotated internal section.
        """
        section = mesh.copy()
        theta = np.deg2rad(32.0)
        points = section.points.copy()
        radius = points[:, 1].copy()
        points[:, 1] = radius * np.cos(theta)
        points[:, 2] = radius * np.sin(theta)
        section.points = points
        return section

    def _pipe_shell(self, mesh: pv.PolyData) -> pv.PolyData:
        """
        Revolve the section boundary by 360 degrees to build the pipe shell.

        Args:
            mesh (pv.PolyData): Axisymmetric section mesh.

        Returns:
            pv.PolyData: Rotated pipe shell.
        """
        boundary = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        return boundary.extrude_rotate(
            resolution=72,
            angle=360.0,
            rotation_axis=(1, 0, 0),
            capping=True,
        )

    def _camera(self, plotter: pv.Plotter, mesh: pv.PolyData) -> None:
        """
        Set a perspective camera with visible depth.

        Args:
            plotter (pv.Plotter): Active plotter.
            mesh (pv.PolyData): Visible 3D pipe mesh.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        cx, cy, cz = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)
        length = (x_max - x_min) or 1.0
        diameter = max(y_max - y_min, z_max - z_min, 1.0)

        plotter.camera.focal_point = (cx, cy, cz)
        plotter.camera.position = (
            cx + 0.48 * length,
            cy - 2.30 * diameter,
            cz + 1.25 * diameter,
        )
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.view_angle = 24.0
        plotter.camera.parallel_projection = False
        plotter.camera.zoom(1.18)
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
            vmax = max(abs(lo), abs(hi))
            lo, hi = -vmax, vmax
        return lo, hi

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
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
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

    def render(self, pred: Tensor, coords: Tensor, label: str, num_nodes: int, num_params: int) -> Path:
        """
        Render one 3D MP4 for the predicted Vy field.

        Args:
            pred (Tensor): Predicted flow sequence. (T, N, C).
            coords (Tensor): Axisymmetric coordinates. (N, 2).
            label (str): Operating-condition label.
            num_nodes (int): Total node count.
            num_params (int): Model parameter count.

        Returns:
            Path: Rendered MP4 path.
        """
        ch_idx = self.ch_names.index("Vy")
        channel_name = self.ch_names[ch_idx]
        field = pred.detach().cpu().numpy().astype(np.float32)[:, :, ch_idx]
        clim = self._channel_clim(field, ch_idx)
        cmap = self._value_cmap(ch_idx, clim)

        points = self._section_points(coords)
        section_2d = self._section_mesh(points)
        section_ids = section_2d.point_data["node_id"].astype(np.int64)
        section = self._rotate_section(section_2d)
        shell = self._pipe_shell(section_2d)
        shell_ids = shell.point_data["node_id"].astype(np.int64)

        section.point_data["scalar"] = field[0, section_ids]
        shell.point_data["scalar"] = field[0, shell_ids]

        plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        plotter.set_background("white")
        plotter.enable_anti_aliasing("ssaa")
        plotter.add_mesh(
            shell,
            scalars="scalar",
            cmap=cmap,
            clim=clim,
            opacity=0.42,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        plotter.add_mesh(
            section,
            scalars="scalar",
            cmap=cmap,
            clim=clim,
            smooth_shading=True,
            scalar_bar_args=self._sbar_args(channel_name),
        )

        title = f"HyperFlowNet (nodes: {num_nodes:,}, params: {num_params:,})"
        plotter.add_text(title, position="upper_edge", font_size=15, color="black")
        plotter.add_text(channel_name, position="upper_left", font_size=18, color="black")

        light = pv.Light(position=(2.0, -3.0, 3.0), focal_point=(0.0, 0.0, 0.0), color="white", intensity=0.8)
        plotter.add_light(light)
        self._camera(plotter, shell)

        def _update(step_idx: int) -> None:
            section.point_data["scalar"] = field[step_idx, section_ids]
            shell.point_data["scalar"] = field[step_idx, shell_ids]

        out_path = self.output_dir / f"{label}_twin_{channel_name.lower()}.mp4"
        self._mp4(plotter, _update, field.shape[0], out_path, desc=f"Rendering {label} 3D twin")
        logger.info(f"3D flow twin saved to {hue.g}{out_path}{hue.q}")
        return out_path
