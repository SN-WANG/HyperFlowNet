# Flow sequence visualization with PyVista / EGL / MP4
# Author: Shengning Wang

import os
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap, LinearSegmentedColormap
from torch import Tensor
from tqdm.auto import tqdm

from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor


def _channel_role(ch_idx: int, spatial_dim: int) -> str:
    """Return the semantic role of one channel."""
    if ch_idx < spatial_dim:
        return "velocity"
    if ch_idx == spatial_dim:
        return "pressure"
    return "temperature"


_FLUENT_SEQ = LinearSegmentedColormap.from_list(
    "fluent_seq",
    ["#0a2a88", "#005bff", "#00a7ff", "#00d68f", "#b8ef00", "#ffd100", "#ff7a00", "#c50000"],
    N=256,
)
_FLUENT_DIV = LinearSegmentedColormap.from_list(
    "fluent_div",
    ["#0a2a88", "#005bff", "#00a7ff", "#00cf9a", "#b8ef00", "#ffd100", "#ff7a00", "#c50000"],
    N=256,
)
_FLUENT_ERR = LinearSegmentedColormap.from_list(
    "fluent_err",
    ["#0c3b9e", "#0094ff", "#33d17a", "#d8ef00", "#ffd100", "#ff7a00", "#c50000"],
    N=256,
)
_CMAP = {
    "velocity": _FLUENT_DIV,
    "pressure": _FLUENT_SEQ,
    "temperature": _FLUENT_SEQ,
    "error": _FLUENT_ERR,
}


class FlowVis:
    """
    GPU-first flow visualization backend for HyperFlowNet.

    All videos are rendered off-screen with PyVista and encoded to MP4 by
    piping raw RGB frames into system ffmpeg.
    """

    FFMPEG_EXE: str = os.environ.get("FFMPEG_EXE", "ffmpeg")

    def __init__(
        self,
        output_dir: str | Path,
        spatial_dim: int = 2,
        channel_names: Sequence[str] | None = None,
        fps: int = 30,
        theme: str = "document",
        window_width: int = 3600,
        subplot_height: int = 380,
        relative_eps: float = 1e-6,
    ) -> None:
        """
        Initialize the visualization backend.

        Args:
            output_dir (str | Path): Directory for rendered MP4 files.
            spatial_dim (int): Spatial dimension. 2 or 3.
            channel_names (Sequence[str] | None): Ordered field names.
            fps (int): Output frames per second.
            theme (str): PyVista theme name.
            window_width (int): Base figure width in pixels.
            subplot_height (int): Minimum subplot height in pixels.
            relative_eps (float): Epsilon for relative-error denominator.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.spatial_dim = spatial_dim
        self.fps = fps
        self.window_width = window_width
        self.subplot_height = subplot_height
        self.focus_width = max(1200, int(window_width * 0.35))
        self.focus_height = max(420, int(subplot_height * 1.20))
        self.relative_eps = relative_eps
        self.p_idx = spatial_dim

        pv.set_plot_theme(theme)

        if channel_names is not None:
            self.ch_names = list(channel_names)
        elif spatial_dim == 2:
            self.ch_names = ["Vx", "Vy", "P", "T"]
        elif spatial_dim == 3:
            self.ch_names = ["Vx", "Vy", "Vz", "P", "T"]
        else:
            self.ch_names = [f"Field_{i}" for i in range(spatial_dim + 2)]

    # ============================================================
    # Geometry
    # ============================================================

    def _points(self, coords: Tensor) -> np.ndarray:
        """
        Convert coordinates to a VTK-friendly point array.

        Args:
            coords (Tensor): Node coordinates. (N, D).

        Returns:
            np.ndarray: Point coordinates. (N, 3).
        """
        pts = coords.detach().cpu().numpy().astype(np.float32)
        if self.spatial_dim == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
        return pts

    def _mesh(self, points: np.ndarray) -> pv.PolyData:
        """
        Build the rendering mesh for the current point set.

        Args:
            points (np.ndarray): Point coordinates. (N, 3).

        Returns:
            pv.PolyData: Surface mesh in 2D or point cloud fallback.
        """
        cloud = pv.PolyData(points)
        if self.spatial_dim != 2:
            return cloud

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(points[:, :2])
            dd, _ = tree.query(points[:, :2], k=2)
            alpha = float(np.mean(dd[:, 1])) * 2.5
            mesh = cloud.delaunay_2d(alpha=alpha)
            if mesh.n_cells == 0:
                return cloud
            return mesh
        except Exception as exc:
            logger.warning(f"delaunay_2d failed ({exc}), falling back to point cloud")
            return cloud

    def _window_size(self, points: np.ndarray, num_cols: int, num_rows: int, focus: bool = False) -> Tuple[int, int]:
        """
        Compute the figure size from geometry aspect ratio.

        Args:
            points (np.ndarray): Point coordinates. (N, 3).
            num_cols (int): Number of subplot columns.
            num_rows (int): Number of subplot rows.
            focus (bool): Whether this is the focused local layout.

        Returns:
            Tuple[int, int]: Window size. (W, H).
        """
        width = self.focus_width if focus else self.window_width
        x_range = float(points[:, 0].max() - points[:, 0].min()) or 1.0
        y_range = float(points[:, 1].max() - points[:, 1].min()) or 1.0
        aspect = x_range / y_range

        col_width = width / max(num_cols, 1)
        subplot_h = max(int(col_width / aspect), self.subplot_height)
        if focus:
            subplot_h = max(subplot_h, self.focus_height)

        return width, subplot_h * num_rows

    def _focus_bounds(self, points: np.ndarray, focus_bbox_rel: Sequence[float]) -> np.ndarray:
        """
        Convert a relative focus box to absolute coordinates.

        Args:
            points (np.ndarray): Point coordinates. (N, 3).
            focus_bbox_rel (Sequence[float]): Relative bbox, length 2 * D.

        Returns:
            np.ndarray: Absolute bounds. (2, D).
        """
        bbox = np.asarray(focus_bbox_rel, dtype=np.float32).reshape(self.spatial_dim, 2)
        mins = points[:, :self.spatial_dim].min(axis=0)
        maxs = points[:, :self.spatial_dim].max(axis=0)
        span = maxs - mins
        lo = mins + bbox[:, 0] * span
        hi = mins + bbox[:, 1] * span
        return np.stack([lo, hi], axis=0)

    def _focus_mask(self, points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """
        Build the node mask of the focus region.

        Args:
            points (np.ndarray): Point coordinates. (N, 3).
            bounds (np.ndarray): Absolute focus bounds. (2, D).

        Returns:
            np.ndarray: Boolean node mask. (N,).
        """
        eps = 1e-6
        mask = np.ones(points.shape[0], dtype=bool)
        for dim in range(self.spatial_dim):
            mask &= points[:, dim] >= bounds[0, dim] - eps
            mask &= points[:, dim] <= bounds[1, dim] + eps
        return mask

    def _camera(self, plotter: pv.Plotter, points: np.ndarray) -> None:
        """
        Set a stable view for one renderer.

        Args:
            plotter (pv.Plotter): Active plotter.
            points (np.ndarray): Visible point coordinates. (N, 3).
        """
        if self.spatial_dim == 2:
            plotter.view_xy()
            x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
            y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            dx = (x_max - x_min) or 1.0
            dy = (y_max - y_min) or 1.0

            renderer = plotter.renderer
            vp = renderer.GetViewport()
            win_w, win_h = plotter.window_size
            vp_w = (vp[2] - vp[0]) * win_w
            vp_h = (vp[3] - vp[1]) * win_h
            vp_aspect = vp_w / max(vp_h, 1.0)

            pad_frac = 0.04
            scale_y = dy * (1.0 + pad_frac) * 0.5
            scale_x = dx * (1.0 + pad_frac) / (2.0 * vp_aspect)
            scale = max(scale_y, scale_x)

            plotter.camera.focal_point = (cx, cy, 0.0)
            plotter.camera.position = (cx, cy, 1.0)
            plotter.camera.parallel_scale = scale
            plotter.camera.parallel_projection = True
            return

        plotter.view_isometric()
        plotter.reset_camera()

    # ============================================================
    # Scalar transforms
    # ============================================================

    def _value_cmap(self, ch_idx: int, clim: Tuple[float, float]) -> Colormap:
        """
        Pick the scalar colormap of one physical channel.

        Args:
            ch_idx (int): Channel index.
            clim (Tuple[float, float]): Value range.

        Returns:
            Colormap: Colormap object.
        """
        role = _channel_role(ch_idx, self.spatial_dim)
        if role == "velocity" and clim[0] >= 0.0:
            return _FLUENT_SEQ
        return _CMAP[role]

    def _clim(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute robust color limits from a temporal stack.

        Args:
            data (np.ndarray): Scalar field stack.

        Returns:
            Tuple[float, float]: Color limits.
        """
        flat = data.ravel()
        lo = float(np.percentile(flat, 2))
        hi = float(np.percentile(flat, 98))
        if abs(hi - lo) < 1e-9:
            center = 0.5 * (lo + hi)
            lo, hi = center - 1e-6, center + 1e-6
        return lo, hi

    def _value_clims(self, gt_np: np.ndarray, pred_np: np.ndarray) -> List[Tuple[float, float]]:
        """
        Compute shared GT / prediction color ranges channel-wise.

        Args:
            gt_np (np.ndarray): Ground truth. (T, N, C).
            pred_np (np.ndarray): Prediction. (T, N, C).

        Returns:
            List[Tuple[float, float]]: Channel-wise color limits.
        """
        num_channels = gt_np.shape[-1]
        clims: List[Tuple[float, float]] = []
        for ch_idx in range(num_channels):
            merged = np.concatenate([gt_np[:, :, ch_idx].ravel(), pred_np[:, :, ch_idx].ravel()])
            lo, hi = self._clim(merged)
            if _channel_role(ch_idx, self.spatial_dim) == "velocity" and lo < 0.0 < hi:
                vmax = max(abs(lo), abs(hi))
                lo, hi = -vmax, vmax
            clims.append((lo, hi))
        return clims

    def _relative(self, gt: Tensor, pred: Tensor) -> Tensor:
        """
        Compute relative-error fields from separately normalized GT / prediction.

        Args:
            gt (Tensor): Ground-truth field. (T, N, C).
            pred (Tensor): Prediction field. (T, N, C).

        Returns:
            Tensor: Relative-error field in percent. (T, N, C).
        """
        gt_scaler = MinMaxScalerTensor(norm_range="unit").fit(gt, channel_dim=-1)
        pred_scaler = MinMaxScalerTensor(norm_range="unit").fit(pred, channel_dim=-1)
        gt_unit = gt_scaler.transform(gt)
        pred_unit = pred_scaler.transform(pred)
        denom = gt_unit.abs().clamp_min(self.relative_eps)
        return (gt_unit - pred_unit).abs() / denom * 100.0

    def _accuracy(self, gt: Tensor, pred: Tensor) -> Tensor:
        """
        Compute per-channel global accuracy from raw physical values.

        Args:
            gt (Tensor): Ground truth. (T, N, C).
            pred (Tensor): Prediction. (T, N, C).

        Returns:
            Tensor: Per-channel accuracy in percent. (C,).
        """
        abs_diff = (gt - pred).abs().sum(dim=(0, 1))
        abs_gt = gt.abs().sum(dim=(0, 1)).clamp_min(self.relative_eps)
        return (1.0 - abs_diff / abs_gt) * 100.0

    def _sbar_args(self, focus: bool = False) -> dict:
        """Return scalar-bar layout arguments."""
        if focus:
            return {
                "height": 0.08,
                "width": 0.72,
                "position_x": 0.14,
                "position_y": 0.10,
                "vertical": False,
                "fmt": "%.2e",
                "title_font_size": 14,
                "label_font_size": 12,
            }
        return {
            "height": 0.06,
            "width": 0.60,
            "position_x": 0.20,
            "position_y": 0.05,
            "vertical": False,
            "fmt": "%.2e",
            "title_font_size": 12,
            "label_font_size": 11,
        }

    # ============================================================
    # Rendering
    # ============================================================

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
            self.FFMPEG_EXE,
            "-y",
            "-framerate",
            str(self.fps),
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

        proc.stdin.write(first_frame.tobytes())
        for step_idx in tqdm(range(1, seq_len), desc=desc, leave=False):
            update_fn(step_idx)
            plotter.render()
            proc.stdin.write(plotter.screenshot(return_img=True).tobytes())

        proc.stdin.close()
        proc.wait()
        plotter.close()

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {proc.returncode}. "
                f"Ensure ffmpeg is on PATH or set FFMPEG_EXE."
            )

    def _draw(
        self,
        plotter: pv.Plotter,
        mesh: pv.PolyData,
        scalars: np.ndarray,
        ch_idx: int,
        clim: Tuple[float, float],
        sbar_title: str,
        title: str | None,
        corner: str | None,
        footer: str | None,
        points: np.ndarray,
        focus: bool = False,
        cmap: str | Colormap | None = None,
    ) -> pv.PolyData:
        """
        Attach one field renderer to the active subplot.

        Args:
            plotter (pv.Plotter): Active plotter.
            mesh (pv.PolyData): Geometry for this pane.
            scalars (np.ndarray): Initial scalar values. (N,).
            ch_idx (int): Channel index.
            clim (Tuple[float, float]): Scalar limits.
            sbar_title (str): Scalar-bar title.
            title (str | None): Top centered title.
            corner (str | None): Upper-left label.
            footer (str | None): Bottom centered label.
            points (np.ndarray): Visible points. (N, 3).
            focus (bool): Whether this pane belongs to the focus layout.
            cmap (str | Colormap | None): Colormap override.

        Returns:
            pv.PolyData: The per-pane mesh handle.
        """
        pane = mesh.copy()
        pane.point_data["scalar"] = scalars.astype(np.float32)

        add_kw = dict(
            scalars="scalar",
            cmap=self._value_cmap(ch_idx, clim) if cmap is None else cmap,
            clim=clim,
            scalar_bar_args={**self._sbar_args(focus), "title": sbar_title},
        )
        if pane.n_cells > 0:
            plotter.add_mesh(pane, **add_kw)
        else:
            plotter.add_mesh(pane, **add_kw, point_size=5, render_points_as_spheres=True)

        if title is not None:
            plotter.add_text(title, position="upper_edge", font_size=15 if focus else 11)
        if corner is not None:
            plotter.add_text(corner, position="upper_left", font_size=18 if focus else 13)
        if footer is not None:
            plotter.add_text(footer, position="lower_edge", font_size=18 if focus else 12)

        self._camera(plotter, points)
        return pane

    # ============================================================
    # Public interface
    # ============================================================

    def render_seq(self, sequence: Tensor, coords: Tensor, case_name: str) -> None:
        """
        Render one rollout with all channels stacked vertically.

        Args:
            sequence (Tensor): Flow sequence. (T, N, C).
            coords (Tensor): Node coordinates. (N, D).
            case_name (str): Output case name.
        """
        seq_np = sequence.detach().cpu().numpy()
        seq_len, _, num_channels = seq_np.shape
        points = self._points(coords)
        mesh = self._mesh(points)
        win_w, win_h = self._window_size(points, num_cols=1, num_rows=num_channels)

        clims = [self._clim(seq_np[:, :, ch_idx]) for ch_idx in range(num_channels)]
        for ch_idx in range(num_channels):
            if _channel_role(ch_idx, self.spatial_dim) == "velocity":
                lo, hi = clims[ch_idx]
                if lo < 0.0 < hi:
                    vmax = max(abs(lo), abs(hi))
                    clims[ch_idx] = (-vmax, vmax)

        plotter = pv.Plotter(shape=(num_channels, 1), off_screen=True, window_size=(win_w // 3, win_h))

        panes: List[pv.PolyData] = []
        for ch_idx in range(num_channels):
            plotter.subplot(ch_idx, 0)
            panes.append(
                self._draw(
                    plotter=plotter,
                    mesh=mesh,
                    scalars=seq_np[0, :, ch_idx],
                    ch_idx=ch_idx,
                    clim=clims[ch_idx],
                    sbar_title=self.ch_names[ch_idx],
                    title=f"Field: {self.ch_names[ch_idx]}",
                    corner=None,
                    footer=None,
                    points=points,
                )
            )

        def _update(step_idx: int) -> None:
            for ch_idx, pane in enumerate(panes):
                pane.point_data["scalar"] = seq_np[step_idx, :, ch_idx].astype(np.float32)

        out_path = self.output_dir / f"{case_name}_seq.mp4"
        self._mp4(plotter, _update, seq_len, out_path, desc=f"Rendering {case_name} sequence")
        logger.info(f"sequence video saved to {hue.g}{out_path}{hue.q}")

    def render_full(
        self,
        gt: Tensor,
        pred: Tensor,
        coords: Tensor,
        case_name: str,
        num_nodes: int,
        num_params: int,
    ) -> None:
        """
        Render the full-channel GT / prediction / relative-error video.

        Args:
            gt (Tensor): Ground truth. (T, N, C).
            pred (Tensor): Prediction. (T, N, C).
            coords (Tensor): Node coordinates. (N, D).
            case_name (str): Output case name.
            num_nodes (int): Total node count.
            num_params (int): Model parameter count.
        """
        seq_len, _, num_channels = gt.shape
        rel = self._relative(gt, pred)
        acc = self._accuracy(gt, pred).detach().cpu().numpy()

        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        rel_np = rel.detach().cpu().numpy()
        points = self._points(coords)
        mesh = self._mesh(points)
        win_w, win_h = self._window_size(points, num_cols=3, num_rows=num_channels)

        value_clims = self._value_clims(gt_np, pred_np)
        rel_clims = [self._clim(rel_np[:, :, ch_idx]) for ch_idx in range(num_channels)]

        plotter = pv.Plotter(shape=(num_channels, 3), off_screen=True, window_size=(win_w, win_h))
        panes: List[List[pv.PolyData]] = []

        node_text = f"{num_nodes:,}"
        param_text = f"{num_params:,}"

        for ch_idx in range(num_channels):
            row: List[pv.PolyData] = []
            col_titles = [
                f"Ground Truth (nodes: {node_text})",
                f"Prediction (params: {param_text})",
                f"Relative Error (accuracy: {acc[ch_idx]:.2f}%)",
            ]
            sbar_titles = [
                self.ch_names[ch_idx] if ch_idx != self.p_idx else "P (Pa)",
                self.ch_names[ch_idx] if ch_idx != self.p_idx else "P (Pa)",
                "Relative Error (%)",
            ]
            cmaps = [
                self._value_cmap(ch_idx, value_clims[ch_idx]),
                self._value_cmap(ch_idx, value_clims[ch_idx]),
                _CMAP["error"],
            ]
            arrays = [gt_np[:, :, ch_idx], pred_np[:, :, ch_idx], rel_np[:, :, ch_idx]]
            clims = [value_clims[ch_idx], value_clims[ch_idx], rel_clims[ch_idx]]

            for col_idx in range(3):
                plotter.subplot(ch_idx, col_idx)
                row.append(
                    self._draw(
                        plotter=plotter,
                        mesh=mesh,
                        scalars=arrays[col_idx][0],
                        ch_idx=ch_idx,
                        clim=clims[col_idx],
                        sbar_title=sbar_titles[col_idx],
                        title=col_titles[col_idx],
                        corner=self.ch_names[ch_idx] if col_idx == 0 else None,
                        footer=None,
                        points=points,
                        cmap=cmaps[col_idx],
                    )
                )
            panes.append(row)

        def _update(step_idx: int) -> None:
            for ch_idx in range(num_channels):
                panes[ch_idx][0].point_data["scalar"] = gt_np[step_idx, :, ch_idx].astype(np.float32)
                panes[ch_idx][1].point_data["scalar"] = pred_np[step_idx, :, ch_idx].astype(np.float32)
                panes[ch_idx][2].point_data["scalar"] = rel_np[step_idx, :, ch_idx].astype(np.float32)

        out_path = self.output_dir / f"{case_name}_full.mp4"
        self._mp4(plotter, _update, seq_len, out_path, desc=f"Rendering {case_name} full")
        logger.info(f"full video saved to {hue.g}{out_path}{hue.q}")

    def render_focus(
        self,
        gt: Tensor,
        pred: Tensor,
        coords: Tensor,
        case_name: str,
        num_nodes: int,
        num_params: int,
        focus_channel_idx: int = 1,
        focus_bbox_rel: Sequence[float] | None = None,
    ) -> None:
        """
        Render the focused local triptych for one selected channel.

        Args:
            gt (Tensor): Ground truth. (T, N, C).
            pred (Tensor): Prediction. (T, N, C).
            coords (Tensor): Node coordinates. (N, D).
            case_name (str): Output case name.
            num_nodes (int): Total node count.
            num_params (int): Model parameter count.
            focus_channel_idx (int): Visualized channel index.
            focus_bbox_rel (Sequence[float] | None): Relative bbox, length 2 * D.
        """
        if focus_bbox_rel is None:
            if self.spatial_dim == 2:
                focus_bbox_rel = (0.74, 1.00, 0.00, 1.00)
            else:
                focus_bbox_rel = (0.74, 1.00, 0.00, 1.00, 0.00, 1.00)

        points = self._points(coords)
        bounds = self._focus_bounds(points, focus_bbox_rel)
        mask = self._focus_mask(points, bounds)
        focus_points = points[mask]
        focus_gt = gt[:, mask, focus_channel_idx:focus_channel_idx + 1]
        focus_pred = pred[:, mask, focus_channel_idx:focus_channel_idx + 1]

        seq_len = focus_gt.shape[0]
        focus_mesh = self._mesh(focus_points)
        rel = self._relative(focus_gt, focus_pred)
        acc = float(
            self._accuracy(
                gt[:, :, focus_channel_idx:focus_channel_idx + 1],
                pred[:, :, focus_channel_idx:focus_channel_idx + 1],
            )[0]
        )

        gt_np = focus_gt.detach().cpu().numpy()
        pred_np = focus_pred.detach().cpu().numpy()
        rel_np = rel.detach().cpu().numpy()

        value_clim = self._value_clims(gt_np, pred_np)[0]
        rel_clim = self._clim(rel_np[:, :, 0])
        channel_name = self.ch_names[focus_channel_idx]

        win_w, win_h = self._window_size(focus_points, num_cols=1, num_rows=3, focus=True)
        plotter = pv.Plotter(shape=(3, 1), off_screen=True, window_size=(win_w, win_h))

        col_titles = ["CFD Simulation", "HyperFlowNet", "Relative Error"]
        footers = [f"nodes: {num_nodes:,}", f"params: {num_params:,}", f"accuracy: {acc:.2f}%"]
        sbar_titles = [channel_name, channel_name, "Relative Error (%)"]
        cmaps = [
            self._value_cmap(focus_channel_idx, value_clim),
            self._value_cmap(focus_channel_idx, value_clim),
            _CMAP["error"],
        ]
        arrays = [gt_np[:, :, 0], pred_np[:, :, 0], rel_np[:, :, 0]]
        clims = [value_clim, value_clim, rel_clim]

        panes: List[pv.PolyData] = []
        for row_idx in range(3):
            plotter.subplot(row_idx, 0)
            panes.append(
                self._draw(
                    plotter=plotter,
                    mesh=focus_mesh,
                    scalars=arrays[row_idx][0],
                    ch_idx=focus_channel_idx,
                    clim=clims[row_idx],
                    sbar_title=sbar_titles[row_idx],
                    title=col_titles[row_idx],
                    corner=channel_name if row_idx == 0 else None,
                    footer=footers[row_idx],
                    points=focus_points,
                    focus=True,
                    cmap=cmaps[row_idx],
                )
            )

        def _update(step_idx: int) -> None:
            for row_idx, pane in enumerate(panes):
                pane.point_data["scalar"] = arrays[row_idx][step_idx].astype(np.float32)

        out_path = self.output_dir / f"{case_name}_focus_{channel_name.lower()}.mp4"
        self._mp4(plotter, _update, seq_len, out_path, desc=f"Rendering {case_name} focus")
        logger.info(f"focus video saved to {hue.g}{out_path}{hue.q}")
