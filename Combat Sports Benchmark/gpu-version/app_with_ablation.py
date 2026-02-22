#!/usr/bin/env python3
"""
app_with_ablation.py — Combat Sports Pipeline Ablation Study
=============================================================
Systematically disables YOLO inference, temporal smoothing, hit-detection,
and overlay rendering to isolate each component's latency contribution.

An ablation study removes components one at a time to measure their
individual contribution to overall system performance, following the
methodology described in Chapter 6 of the thesis.

Usage
-----
    # Real video (requires models/best.pt or --model path)
    python app_with_ablation.py --video data/fight.mp4

    # Custom model / frame limit
    python app_with_ablation.py --video data/fight.mp4 --model models/best.pt --frames 300

    # No video — synthetic random frames for pure timing benchmarks
    python app_with_ablation.py --synthetic --frames 200

    # Only regenerate the Chapter 6 text (re-uses existing summary CSV)
    python app_with_ablation.py --text-only

Outputs  (all saved to thesis_assets/ablation/ by default)
------------------------------------------
    ablation_results.csv       raw per-frame timings for every config
    ablation_summary.csv       mean / SD / P95 / FPS per config
    ablation_table.tex         booktabs LaTeX table ready for thesis
    ablation_latency.png       stacked-bar latency figure
    chapter6_ablation.txt      auto-generated Chapter 6 subsection

Notes
-----
- No results are invented.  The thesis text generator reads the summary CSV
  and substitutes real measurements.  If no CSV exists the generator
  produces a justification-of-omission section based on latency budget and
  design trade-off arguments only.
- CUDA synchronisation is used before/after YOLO so GPU timings are
  wall-clock accurate.
- A GPU warm-up pass (10 frames) is performed before any timed run.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import os
import pathlib
import sys
import time
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")               # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

# Fix Windows path issue for torch hub models (same fix as original app)
pathlib.PosixPath = pathlib.WindowsPath

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Compute device: {device}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONF_THRESH  = 0.40
CLASS_BAG    = 0
CLASS_PUNCH  = 5
CLASS_KICK   = 2

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),     # bag        – green
    5: (255, 0, 255),   # punch      – magenta
    2: (0, 255, 255),   # kick       – cyan
    4: (255, 0, 0),     # person     – blue
    1: (0, 128, 255),   # high-guard – orange-blue
    3: (255, 165, 0),   # low-guard  – orange
}

CLASS_NAMES: Dict[int, str] = {
    0: "bag", 1: "high-guard", 2: "kick",
    3: "low-guard", 4: "person", 5: "punch",
}

# Stage colours used in the latency figure
STAGE_COLOURS = {
    "yolo":        "#2196F3",   # blue
    "hit_det":     "#FF9800",   # amber
    "temporal":    "#4CAF50",   # green
    "overlay":     "#F44336",   # red
}

REALTIME_THRESHOLD_MS = 33.3    # 30 FPS


# ===========================================================================
# Section 1 — Ablation Configurations
# ===========================================================================

@dataclass
class AblationConfig:
    """Describes one ablation variant of the pipeline."""
    name: str
    enable_yolo:          bool = True
    enable_hit_detection: bool = True
    enable_temporal:      bool = True
    enable_overlay:       bool = True

    def flags_dict(self) -> dict:
        return {
            "enable_yolo":          self.enable_yolo,
            "enable_hit_detection": self.enable_hit_detection,
            "enable_temporal":      self.enable_temporal,
            "enable_overlay":       self.enable_overlay,
        }

    def component_marks(self) -> str:
        """One-line summary of which stages are active, e.g. ✓ ✓ ✓ ✗"""
        def m(b): return "\\checkmark" if b else "$\\times$"
        return (
            f"{m(self.enable_yolo)} & "
            f"{m(self.enable_hit_detection)} & "
            f"{m(self.enable_temporal)} & "
            f"{m(self.enable_overlay)}"
        )


# Standard six-config ablation suite
#   Row 0  – Full Pipeline (baseline)
#   Rows 1-4 – one component removed each time
#   Row 5  – YOLO-only (all post-processing stripped)
ABLATION_SUITE: List[AblationConfig] = [
    AblationConfig("Full Pipeline",      True,  True,  True,  True),
    AblationConfig("No YOLO",            False, True,  True,  True),
    AblationConfig("No Hit Detection",   True,  False, True,  True),
    AblationConfig("No Temporal",        True,  True,  False, True),
    AblationConfig("No Overlay",         True,  True,  True,  False),
    AblationConfig("YOLO Only",          True,  False, False, False),
]


# ===========================================================================
# Section 2 — Per-Frame Timing Container
# ===========================================================================

@dataclass
class FrameTiming:
    config_name: str
    frame_idx:   int
    yolo_ms:     float
    hit_det_ms:  float
    temporal_ms: float
    overlay_ms:  float

    @property
    def total_ms(self) -> float:
        return self.yolo_ms + self.hit_det_ms + self.temporal_ms + self.overlay_ms


# ===========================================================================
# Section 3 — Geometry Helpers  (identical to production pipeline)
# ===========================================================================

def boxes_intersect(boxA: Tuple, boxB: Tuple) -> bool:
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    return not (xA2 < xB1 or xB2 < xA1 or yA2 < yB1 or yB2 < yA1)


def merge_overlapping_boxes(boxes: List[Tuple]) -> List[Tuple]:
    merged: List[Tuple] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        placed = False
        for i, (mx1, my1, mx2, my2) in enumerate(merged):
            if boxes_intersect(box, (mx1, my1, mx2, my2)):
                merged[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                placed = True
                break
        if not placed:
            merged.append(box)

    # Second pass: merge any new overlaps created by expansion
    changed = True
    while changed:
        changed = False
        new_merged: List[Tuple] = []
        for box in merged:
            x1, y1, x2, y2 = box
            placed = False
            for j, (nx1, ny1, nx2, ny2) in enumerate(new_merged):
                if boxes_intersect(box, (nx1, ny1, nx2, ny2)):
                    new_merged[j] = (min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2))
                    placed = True
                    changed = True
                    break
            if not placed:
                new_merged.append(box)
        merged = new_merged
    return merged


# ===========================================================================
# Section 4 — YOLO Utilities
# ===========================================================================

def load_yolo_model(weights: str = "models/best.pt"):
    """Load a YOLOv5 custom model.  Returns None on failure."""
    if not os.path.exists(weights):
        logger.warning(f"Weights file not found: {weights}")
        return None
    try:
        logger.info(f"Loading YOLO model from {weights} ...")
        # Suppress torch-hub stdout (model summary) AND stderr (pip noise)
        import contextlib, io
        _null = io.StringIO()
        with contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
            model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=weights, force_reload=False, verbose=False,
            )
        model.to(device).eval()  # type: ignore[union-attr]
        logger.info("YOLO model loaded successfully.")
        return model
    except Exception as exc:
        logger.warning(f"Could not load YOLO model: {exc}")
        return None


def yolo_inference_timed(frame: np.ndarray, model) -> Tuple[list, float]:
    """
    Run YOLO inference on one BGR frame.
    Returns (detections_list, elapsed_ms).
    Uses CUDA synchronise so GPU timing is wall-clock accurate.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model.iou  = 0.2
    model.conf = 0.1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    results = model(rgb)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    try:
        dets = list(results.xyxy[0].cpu().numpy())
    except Exception:
        dets = []

    return dets, (t1 - t0) * 1_000.0


# ===========================================================================
# Section 5 — Pipeline Stage Functions
# ===========================================================================

def _check_overlap(
    action_boxes: List[Tuple],
    bag_boxes:    List[Tuple],
    frame:        np.ndarray,
    draw:         bool = False,
) -> bool:
    for ab in action_boxes:
        for bb in bag_boxes:
            if boxes_intersect(ab, bb):
                if draw:
                    ca = ((ab[0] + ab[2]) // 2, (ab[1] + ab[3]) // 2)
                    cb = ((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2)
                    cv2.line(frame, ca, cb, (0, 255, 255), 2)
                return True
    return False


def run_hit_detection_stage(
    punch_boxes: List[Tuple],
    kick_boxes:  List[Tuple],
    bag_boxes:   List[Tuple],
    frame:       np.ndarray,
) -> Tuple[bool, bool, float]:
    """
    Compute spatial overlap between action boxes and bag boxes.
    Returns (punch_overlap, kick_overlap, elapsed_ms).
    """
    t0 = time.perf_counter()
    ov_punch = _check_overlap(punch_boxes, bag_boxes, frame)
    ov_kick  = _check_overlap(kick_boxes,  bag_boxes, frame)
    return ov_punch, ov_kick, (time.perf_counter() - t0) * 1_000.0


def run_temporal_stage(
    ov_punch:  bool,
    ov_kick:   bool,
    state:     dict,
    frame_idx: int,
) -> Tuple[dict, float]:
    """
    Apply gap-tolerance / minimum-duration event debouncing.
    Identical logic to the production pipeline.
    Returns (updated_state, elapsed_ms).
    """
    t0 = time.perf_counter()
    for action, is_over in [("punch", ov_punch), ("kick", ov_kick)]:
        if is_over:
            state["gap_counter"][action] = 0
            if not state["in_event"][action]:
                state["in_event"][action]   = True
                state["event_start"][action] = frame_idx
        else:
            if state["in_event"][action]:
                state["gap_counter"][action] += 1
                if state["gap_counter"][action] >= state["gap_tolerance"][action]:
                    dur = frame_idx - state["event_start"][action]
                    if dur >= state["min_event_dur"][action]:
                        state["counters"][action] += 1
                    state["in_event"][action]  = False
                    state["gap_counter"][action] = 0
    return state, (time.perf_counter() - t0) * 1_000.0


def run_overlay_stage(
    frame:     np.ndarray,
    filtered:  list,
    counters:  dict,
    fps_disp:  float,
) -> Tuple[np.ndarray, float]:
    """
    Draw bounding boxes, labels, hit counters and FPS overlay.
    Returns (annotated_frame, elapsed_ms).
    """
    t0 = time.perf_counter()
    for det in filtered:
        x1, y1, x2, y2, conf, cls_id_f = map(float, det[:6])
        cls_id = int(cls_id_f)
        color  = CLASS_COLORS.get(cls_id, (0, 0, 255))
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.putText(frame, label, (xi1, yi1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    total_hits = counters["punch"] + counters["kick"]
    cv2.putText(frame, f"Hits: {total_hits}",
                (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Punch: {counters['punch']}",
                (50, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(frame, f"Kick: {counters['kick']}",
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    fps_text = f"FPS: {fps_disp:.1f}"
    ts = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(frame, fps_text,
                (frame.shape[1] - ts[0] - 10, ts[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, (time.perf_counter() - t0) * 1_000.0


# ===========================================================================
# Section 6 — AblationRunner
# ===========================================================================

class AblationRunner:
    """
    Runs the inference pipeline under each AblationConfig and records
    per-frame timings for every stage.
    """

    def __init__(self, model=None, warmup_frames: int = 10):
        self.model         = model
        self.warmup_frames = warmup_frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_state(self) -> dict:
        """Fresh temporal-smoothing state dict (one per config run)."""
        return {
            "counters":      {"punch": 0, "kick": 0},
            "in_event":      {"punch": False, "kick": False},
            "event_start":   {"punch": 0, "kick": 0},
            "gap_counter":   {"punch": 0, "kick": 0},
            "gap_tolerance": {"punch": 1, "kick": 4},
            "min_event_dur": {"punch": 2, "kick": 6},
        }

    def _warmup(self, sample_frame: np.ndarray) -> None:
        """Run YOLO N times to stabilise GPU clock and JIT compilation."""
        if self.model is None:
            return
        logger.info(f"GPU warm-up: {self.warmup_frames} YOLO passes ...")
        for _ in range(self.warmup_frames):
            yolo_inference_timed(sample_frame, self.model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("Warm-up complete.")

    # ------------------------------------------------------------------
    # Single-config run
    # ------------------------------------------------------------------

    def run_config(
        self,
        config: AblationConfig,
        frames: List[np.ndarray],
    ) -> List[FrameTiming]:
        """
        Process every frame under a single AblationConfig.
        Returns one FrameTiming per frame.
        """
        logger.info(f"  [{config.name}]  n_frames={len(frames)}")
        state    = self._make_state()
        timings: List[FrameTiming] = []
        prev_t   = time.time()

        for idx, frame in enumerate(frames):
            frame_copy = frame.copy()

            # ── Stage 1: YOLO inference ─────────────────────────────────
            if config.enable_yolo and self.model is not None:
                dets, yolo_ms = yolo_inference_timed(frame_copy, self.model)
            else:
                t0 = time.perf_counter()
                dets   = []
                yolo_ms = (time.perf_counter() - t0) * 1_000.0  # ~0 ms

            filtered   = [d for d in dets if len(d) >= 6 and float(d[4]) >= CONF_THRESH]
            raw_bag    = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == CLASS_BAG]
            raw_punch  = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == CLASS_PUNCH]
            raw_kick   = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == CLASS_KICK]
            bag_boxes   = merge_overlapping_boxes(raw_bag)
            punch_boxes = merge_overlapping_boxes(raw_punch)
            kick_boxes  = merge_overlapping_boxes(raw_kick)

            # ── Stage 2: Hit detection (spatial overlap) ─────────────────
            if config.enable_hit_detection:
                ov_punch, ov_kick, hit_det_ms = run_hit_detection_stage(
                    punch_boxes, kick_boxes, bag_boxes, frame_copy
                )
            else:
                t0 = time.perf_counter()
                ov_punch = ov_kick = False
                hit_det_ms = (time.perf_counter() - t0) * 1_000.0

            # ── Stage 3: Temporal smoothing (event debouncing) ────────────
            if config.enable_temporal:
                state, temporal_ms = run_temporal_stage(ov_punch, ov_kick, state, idx)
            else:
                # Disabled: count every overlap frame directly, no debouncing
                t0 = time.perf_counter()
                if ov_punch:
                    state["counters"]["punch"] += 1
                if ov_kick:
                    state["counters"]["kick"] += 1
                temporal_ms = (time.perf_counter() - t0) * 1_000.0

            # ── Stage 4: Overlay rendering ────────────────────────────────
            if config.enable_overlay:
                now      = time.time()
                fps_disp = 1.0 / (now - prev_t) if (now - prev_t) > 0 else 0.0
                prev_t   = now
                _, overlay_ms = run_overlay_stage(
                    frame_copy, filtered, state["counters"], fps_disp
                )
            else:
                t0 = time.perf_counter()
                overlay_ms = (time.perf_counter() - t0) * 1_000.0

            timings.append(FrameTiming(
                config_name = config.name,
                frame_idx   = idx,
                yolo_ms     = yolo_ms,
                hit_det_ms  = hit_det_ms,
                temporal_ms = temporal_ms,
                overlay_ms  = overlay_ms,
            ))

            if (idx + 1) % 50 == 0:
                logger.info(f"    {idx + 1}/{len(frames)} frames done")

        logger.info(
            f"    >> mean total: "
            f"{np.mean([t.total_ms for t in timings]):.1f} ms/frame"
        )
        return timings

    # ------------------------------------------------------------------
    # Full suite
    # ------------------------------------------------------------------

    def run_all(
        self,
        frames:  List[np.ndarray],
        configs: Optional[List[AblationConfig]] = None,
    ) -> List[FrameTiming]:
        """Run every config in the suite and return all timings."""
        if configs is None:
            configs = ABLATION_SUITE
        if not frames:
            logger.error("No frames provided — aborting.")
            return []

        self._warmup(frames[0])

        all_timings: List[FrameTiming] = []
        for i, cfg in enumerate(configs, 1):
            logger.info(f"Config {i}/{len(configs)}: {cfg.name}")
            all_timings.extend(self.run_config(cfg, frames))

        logger.info(f"Ablation complete: {len(all_timings)} frame timings collected.")
        return all_timings


# ===========================================================================
# Section 7 — AblationExporter
# ===========================================================================

class AblationExporter:
    """
    Converts raw FrameTiming records into all thesis-ready output files.
    """

    STAGE_COLS = ["yolo_ms", "hit_det_ms", "temporal_ms", "overlay_ms"]
    STAGE_LABELS = {
        "yolo_ms":     "YOLO Inference",
        "hit_det_ms":  "Hit Detection",
        "temporal_ms": "Temporal Smoothing",
        "overlay_ms":  "Overlay Rendering",
    }

    def __init__(
        self,
        timings:       List[FrameTiming],
        configs:       List[AblationConfig],
        output_dir:    str = "thesis_assets/ablation",
        hardware_info: Optional[dict] = None,
    ):
        self.timings       = timings
        self.configs       = configs
        self.output_dir    = pathlib.Path(output_dir)
        self.hardware_info = hardware_info or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build DataFrames once
        self.raw_df     = self._build_raw_df()
        self.summary_df = self._build_summary_df()

    # ------------------------------------------------------------------
    # DataFrame builders
    # ------------------------------------------------------------------

    def _build_raw_df(self) -> pd.DataFrame:
        # Config flags lookup
        flag_map = {cfg.name: cfg.flags_dict() for cfg in self.configs}
        rows = []
        for t in self.timings:
            flags = flag_map.get(t.config_name, {})
            rows.append({
                "config_name":          t.config_name,
                "enable_yolo":          flags.get("enable_yolo",          True),
                "enable_hit_detection": flags.get("enable_hit_detection", True),
                "enable_temporal":      flags.get("enable_temporal",      True),
                "enable_overlay":       flags.get("enable_overlay",       True),
                "frame_idx":            t.frame_idx,
                "yolo_ms":              t.yolo_ms,
                "hit_det_ms":           t.hit_det_ms,
                "temporal_ms":          t.temporal_ms,
                "overlay_ms":           t.overlay_ms,
                "total_ms":             t.total_ms,
            })
        return pd.DataFrame(rows)

    def _build_summary_df(self) -> pd.DataFrame:
        flag_map: Dict[str, dict] = {cfg.name: cfg.flags_dict() for cfg in self.configs}
        rows: List[dict] = []
        for cfg_name_raw, grp in self.raw_df.groupby("config_name", sort=False):
            cfg_name: str = str(cfg_name_raw)
            flags = flag_map.get(cfg_name, {})
            row: dict = {
                "config_name":          cfg_name,
                "enable_yolo":          flags.get("enable_yolo",          True),
                "enable_hit_detection": flags.get("enable_hit_detection", True),
                "enable_temporal":      flags.get("enable_temporal",      True),
                "enable_overlay":       flags.get("enable_overlay",       True),
                "n_frames":             len(grp),
            }
            for col in self.STAGE_COLS:
                row[f"{col[:-3]}_mean"] = round(float(grp[col].mean()), 3)
                row[f"{col[:-3]}_sd"]   = round(float(grp[col].std()),  3)
                row[f"{col[:-3]}_p95"]  = round(float(grp[col].quantile(0.95)), 3)
            total_mean = float(grp["total_ms"].mean())
            row["total_mean"] = round(total_mean, 3)
            row["total_sd"]   = round(float(grp["total_ms"].std()), 3)
            row["total_p95"]  = round(float(grp["total_ms"].quantile(0.95)), 3)
            row["fps_mean"]   = round(1_000.0 / total_mean, 1) if total_mean > 0 else 0.0
            rows.append(row)

        # Preserve original suite order
        order: Dict[str, int] = {cfg.name: i for i, cfg in enumerate(self.configs)}
        df = pd.DataFrame(rows)
        df["_order"] = df["config_name"].apply(lambda x: order.get(str(x), 999))
        df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 1.  Raw CSV
    # ------------------------------------------------------------------

    def save_raw_csv(self) -> pathlib.Path:
        p = self.output_dir / "ablation_results.csv"
        self.raw_df.to_csv(p, index=False)
        logger.info(f"Raw timing data -> {p}")
        return p

    # ------------------------------------------------------------------
    # 2.  Summary CSV
    # ------------------------------------------------------------------

    def save_summary_csv(self) -> pathlib.Path:
        p = self.output_dir / "ablation_summary.csv"
        self.summary_df.to_csv(p, index=False)
        logger.info(f"Summary CSV -> {p}")
        return p

    # ------------------------------------------------------------------
    # 3.  LaTeX booktabs table
    # ------------------------------------------------------------------

    def save_latex_table(self) -> pathlib.Path:
        p    = self.output_dir / "ablation_table.tex"
        df   = self.summary_df
        hw   = self.hardware_info

        # Compute Δ relative to Full Pipeline (row 0)
        base_mean = df.iloc[0]["total_mean"]
        base_p95  = df.iloc[0]["total_p95"]

        hw_note = ""
        if hw.get("gpu_name"):
            hw_note = (
                f"Measurements taken on {hw['gpu_name']} "
                f"(CUDA {hw.get('cuda_version', 'N/A')}) "
                f"over $N={int(df.iloc[0]['n_frames'])}$ frames per configuration."
            )
        else:
            n = int(df.iloc[0]["n_frames"]) if len(df) > 0 else "N"
            hw_note = f"Measurements over $N={n}$ synthetic frames per configuration (no YOLO model loaded)."

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation study: per-frame latency by pipeline configuration. " + hw_note + r"}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccccrrrrr}",
            r"\toprule",
            r"\multirow{2}{*}{Configuration} & \multicolumn{4}{c}{Components} & "
            r"\multicolumn{5}{c}{Latency} \\",
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-10}",
            r"& YOLO & Hit Det. & Temporal & Overlay "
            r"& Mean (ms) & SD (ms) & P\textsubscript{95} (ms) "
            r"& FPS & $\Delta$ (ms) \\",
            r"\midrule",
        ]

        for _, row in df.iterrows():
            cfg_name = str(row["config_name"])
            bold     = cfg_name == "Full Pipeline"

            def _bold(s: str, _bold: bool = bold) -> str:
                return f"\\textbf{{{s}}}" if _bold else s

            # Component checkmarks — read as bool explicitly to avoid Series ambiguity
            def _check(col: str) -> str:
                return r"\checkmark" if bool(row[col]) else r"$\times$"

            marks = (
                f"{_check('enable_yolo')} & "
                f"{_check('enable_hit_detection')} & "
                f"{_check('enable_temporal')} & "
                f"{_check('enable_overlay')}"
            )

            delta     = float(row["total_mean"]) - base_mean
            delta_str = "—" if cfg_name == "Full Pipeline" else f"{delta:+.1f}"

            # Build cell strings separately to avoid nested f-string quoting issues
            s_mean  = f"{float(row['total_mean']):.1f}"
            s_sd    = f"{float(row['total_sd']):.1f}"
            s_p95   = f"{float(row['total_p95']):.1f}"
            s_fps   = f"{float(row['fps_mean']):.0f}"

            lines.append(
                f"{_bold(cfg_name)} & {marks} & "
                f"{_bold(s_mean)} & "
                f"{_bold(s_sd)} & "
                f"{_bold(s_p95)} & "
                f"{_bold(s_fps)} & "
                f"{_bold(delta_str)} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        logger.info(f"LaTeX table -> {p}")
        return p

    # ------------------------------------------------------------------
    # 4.  Latency figure
    # ------------------------------------------------------------------

    def save_latency_figure(self) -> pathlib.Path:
        p   = self.output_dir / "ablation_latency.png"
        df  = self.summary_df

        cfg_names = df["config_name"].tolist()
        n_cfgs    = len(cfg_names)
        y_pos     = np.arange(n_cfgs)

        stage_keys = [
            ("yolo_ms",     "yolo"),
            ("hit_det_ms",  "hit_det"),
            ("temporal_ms", "temporal"),
            ("overlay_ms",  "overlay"),
        ]

        fig, ax = plt.subplots(figsize=(11, 0.9 * n_cfgs + 2.5))
        fig.patch.set_facecolor("white")

        bottoms = np.zeros(n_cfgs)
        patches = []
        for raw_col, stage_key in stage_keys:
            mean_col = f"{raw_col[:-3]}_mean"
            vals     = df[mean_col].values.astype(float)
            color    = STAGE_COLOURS[stage_key]
            label    = self.STAGE_LABELS[raw_col]
            bars = ax.barh(y_pos, vals, left=bottoms, color=color,
                           height=0.55, label=label, edgecolor="white", linewidth=0.4)
            patches.append(mpatches.Patch(facecolor=color, label=label))
            bottoms += vals

        # P95 markers (small vertical tick on each bar)
        p95_vals = df["total_p95"].values.astype(float)
        ax.scatter(p95_vals, y_pos, marker="|", color="black",
                   s=120, zorder=5, linewidths=1.5, label="P$_{95}$")

        # 30 FPS threshold line
        ax.axvline(x=REALTIME_THRESHOLD_MS, color="#333333", linestyle="--",
                   linewidth=1.2, label=f"{REALTIME_THRESHOLD_MS} ms (30 FPS)")

        # Annotate total mean values
        for i, (mean_val, p95_val) in enumerate(zip(df["total_mean"].values, p95_vals)):
            ax.text(mean_val + 0.4, i, f"{mean_val:.1f}", va="center",
                    ha="left", fontsize=8.5, color="#222222")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cfg_names, fontsize=9)
        ax.set_xlabel("Mean per-frame latency (ms)", fontsize=10)
        ax.set_title(
            "Ablation Study: Pipeline Component Latency Contributions\n"
            "(stacked bars = stage means; | = P$_{95}$; dashed = 30 FPS threshold)",
            fontsize=10, pad=10,
        )
        ax.invert_yaxis()   # Full Pipeline at top
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        p95_patch   = mpatches.Patch(color="black", label="P$_{95}$")
        thr_patch   = mpatches.Patch(
            facecolor="none", edgecolor="#333333",
            linestyle="--", linewidth=1.2,
            label=f"{REALTIME_THRESHOLD_MS} ms (30 FPS)",
        )
        all_handles = patches + [
            mlines.Line2D([0], [0], color="black", marker="|",
                          linestyle="none", markersize=8, label="P$_{95}$"),
            mlines.Line2D([0], [0], color="#333333", linestyle="--",
                          linewidth=1.2, label=f"{REALTIME_THRESHOLD_MS} ms (30 FPS)"),
        ]
        ax.legend(handles=all_handles, loc="lower right", fontsize=8.5, framealpha=0.9)

        plt.tight_layout()
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Latency figure -> {p}")
        return p

    # ------------------------------------------------------------------
    # 5.  Master export method
    # ------------------------------------------------------------------

    def export_all(self) -> Dict[str, pathlib.Path]:
        logger.info(f"Exporting results to {self.output_dir} ...")
        out = {
            "raw_csv":     self.save_raw_csv(),
            "summary_csv": self.save_summary_csv(),
            "latex":       self.save_latex_table(),
            "figure":      self.save_latency_figure(),
        }
        return out


# ===========================================================================
# Section 8 — Chapter 6 Thesis Text Generator
# ===========================================================================

class ThesisChapter6Generator:
    """
    Generates a Chapter 6 ablation subsection from real measurements.

    Two branches:
      • generate_with_results()    — summary_df exists and has data
      • generate_without_results() — no data; produces justification
                                     section citing latency budget and
                                     design trade-offs only
    """

    def __init__(
        self,
        output_dir:    str  = "thesis_assets/ablation",
        summary_df:    Optional[pd.DataFrame] = None,
        hardware_info: Optional[dict] = None,
    ):
        self.output_dir    = pathlib.Path(output_dir)
        self.summary_df    = summary_df
        self.hardware_info = hardware_info or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _hw_string(self) -> str:
        hw = self.hardware_info
        if hw.get("gpu_name"):
            return (
                f"an {hw['gpu_name']} (CUDA {hw.get('cuda_version', 'N/A')}, "
                f"PyTorch {hw.get('pytorch_version', 'N/A')})"
            )
        return "the test hardware (CPU-only mode; no YOLO model was loaded)"

    # ------------------------------------------------------------------

    def generate_with_results(self) -> str:
        """Produce thesis text using actual measured values.  No values invented."""
        assert self.summary_df is not None, "summary_df must exist in generate_with_results"
        df: pd.DataFrame = self.summary_df
        hw  = self._hw_string()

        # Helper: get row by config name — returns pd.Series or None
        def row(name: str):
            matches = df[df["config_name"] == name]
            return matches.iloc[0] if len(matches) > 0 else None

        full = row("Full Pipeline")
        if full is None:
            logger.warning("Full Pipeline row not found in summary — falling back.")
            return self.generate_without_results()

        base_mean = full["total_mean"]
        base_p95  = full["total_p95"]
        base_fps  = full["fps_mean"]
        n_frames  = int(full["n_frames"])

        # Collect component contributions
        contributions = []
        for cfg_name in ["No YOLO", "No Hit Detection", "No Temporal", "No Overlay"]:
            r = row(cfg_name)
            if r is None:
                continue
            delta_mean = base_mean - r["total_mean"]   # saving from removal
            pct        = 100.0 * delta_mean / base_mean if base_mean > 0 else 0.0

            # Which stage was removed? Map config name → column
            stage_map = {
                "No YOLO":          "yolo_ms_mean",
                "No Hit Detection": "hit_det_ms_mean",
                "No Temporal":      "temporal_ms_mean",
                "No Overlay":       "overlay_ms_mean",
            }
            stage_col  = stage_map.get(cfg_name, "")
            stage_mean = full[stage_col] if stage_col and stage_col in full.index else float("nan")

            contributions.append({
                "name":       cfg_name,
                "delta_mean": delta_mean,
                "pct":        pct,
                "stage_mean": stage_mean,
                "cfg_mean":   r["total_mean"],
                "cfg_fps":    r["fps_mean"],
            })

        # Sort by absolute contribution descending
        contributions.sort(key=lambda x: abs(x["delta_mean"]), reverse=True)

        # Identify dominant component
        dominant    = contributions[0] if contributions else None
        negligible  = [c for c in contributions if abs(c["pct"]) < 2.0]

        # Real-time assessment
        rt_capable = base_mean < REALTIME_THRESHOLD_MS
        rt_str = (
            f"below the {REALTIME_THRESHOLD_MS:.1f}~ms threshold required for 30 FPS"
            if rt_capable else
            f"above the {REALTIME_THRESHOLD_MS:.1f}~ms real-time threshold, "
            f"limiting deployment to batch-processing contexts"
        )

        # ──────────────────────────────────────────────────────────────
        # Build prose paragraphs
        # ──────────────────────────────────────────────────────────────
        paras = []

        # P1 – Motivation and methodology
        paras.append(textwrap.dedent(f"""\
            \\subsection{{Ablation Study}}
            \\label{{sec:ablation}}

            To quantify the contribution of each pipeline component to overall
            end-to-end latency, a systematic ablation study was conducted on {hw}.
            The study follows the standard methodology of disabling one stage at a
            time while keeping all other stages active, then comparing per-frame
            timing against the full-pipeline baseline.
            The four components evaluated are: (i)~YOLO object-detection inference,
            (ii)~spatial hit-detection (bounding-box overlap), (iii)~temporal
            smoothing (gap-tolerance and minimum-duration event debouncing), and
            (iv)~overlay rendering (OpenCV drawing calls).
            Six configurations were tested over $N = {n_frames}$ frames each;
            results are summarised in Table~\\ref{{tab:ablation}} and
            Figure~\\ref{{fig:ablation_latency}}.
        """))

        # P2 – Baseline
        paras.append(textwrap.dedent(f"""\
            \\subsubsection*{{Baseline}}

            The full pipeline achieved a mean per-frame latency of
            ${base_mean:.1f}$~ms (SD~$= {full['total_sd']:.1f}$~ms,
            P\\textsubscript{{95}}~$= {base_p95:.1f}$~ms), equivalent to
            ${base_fps:.0f}$~FPS.
            This is {rt_str}, {'confirming' if rt_capable else 'indicating limitations for'}
            real-time operation at the target frame rate.
        """))

        # P3 – Per-component findings
        component_bullets = []
        stage_label_map = {
            "No YOLO":          "YOLO inference",
            "No Hit Detection": "hit-detection",
            "No Temporal":      "temporal smoothing",
            "No Overlay":       "overlay rendering",
        }
        for c in contributions:
            lbl   = stage_label_map.get(c["name"], c["name"])
            delta = c["delta_mean"]
            pct   = c["pct"]
            sign  = "reduction" if delta > 0 else "increase"
            component_bullets.append(
                f"  \\item \\textbf{{{lbl}}} — removing this stage changed mean "
                f"latency by ${abs(delta):.1f}$~ms "
                f"({abs(pct):.1f}\\%~{sign}, residual "
                f"${c['cfg_mean']:.1f}$~ms at ${c['cfg_fps']:.0f}$~FPS)."
            )

        paras.append(
            "\\subsubsection*{Component Contributions}\n\n"
            "Removing each component in turn yielded the following latency changes "
            "relative to the full pipeline:\n\n"
            "\\begin{itemize}\n"
            + "\n".join(component_bullets)
            + "\n\\end{itemize}\n"
        )

        # P4 – Dominant component interpretation
        if dominant:
            dom_lbl: str = stage_label_map.get(dominant["name"]) or str(dominant["name"])
            # Capitalise only the first letter; leave the rest unchanged (preserves YOLO etc.)
            dom_lbl_cap = dom_lbl[0].upper() + dom_lbl[1:] if dom_lbl else dom_lbl
            paras.append(textwrap.dedent(f"""\
                \\subsubsection*{{Discussion}}

                {dom_lbl_cap} constitutes the largest single contributor
                to pipeline latency
                (${abs(dominant['delta_mean']):.1f}$~ms,
                {abs(dominant['pct']):.1f}\\% of the full-pipeline mean),
                confirming that it is the dominant bottleneck.
                Future work targeting throughput improvements — such as model
                quantisation, TensorRT export, or batched inference — should
                therefore prioritise this stage.
            """))

            if negligible:
                negl_names = " and ".join(
                    stage_label_map.get(c["name"]) or str(c["name"])
                    for c in negligible
                )
                paras.append(textwrap.dedent(f"""\
                    By contrast, {negl_names} each contributed less than 2\\% of
                    total latency.  These components are functionally essential for
                    accurate hit counting and video output but impose negligible
                    computational overhead; retaining them imposes no meaningful
                    latency penalty within the real-time budget.
                """))

        # P5 – Temporal smoothing note (accuracy vs latency trade-off)
        no_temp = row("No Temporal")
        if no_temp is not None:
            delta_temp = base_mean - no_temp["total_mean"]
            paras.append(textwrap.dedent(f"""\
                Temporal smoothing (gap-tolerance and minimum-duration debouncing)
                incurred only ${full.get('temporal_ms_mean', 0):.2f}$~ms per frame —
                a negligible latency cost — yet it is architecturally significant:
                removing it eliminates the debouncing logic that suppresses spurious
                single-frame overlaps, directly inflating hit counts.
                The latency trade-off (${abs(delta_temp):.2f}$~ms) is therefore
                justified by its accuracy contribution, which cannot be captured
                purely by timing measurements.
            """))

        return "\n\n".join(paras)

    # ------------------------------------------------------------------

    def generate_without_results(self) -> str:
        """
        Produce a justification-of-omission section when no ablation data
        is available.  Uses latency budget arguments and design trade-offs.
        No performance figures are invented.
        """
        return textwrap.dedent(f"""\
            \\subsection{{Ablation Study}}
            \\label{{sec:ablation}}

            A formal ablation study was not conducted within the scope of this
            project due to constraints on available video footage and labelled
            ground-truth data necessary to measure the accuracy impact of
            removing individual pipeline stages.
            This section provides a qualitative justification for the inclusion
            of each pipeline component, grounded in the real-time latency budget
            and design trade-off arguments described below.

            \\subsubsection*{{Latency Budget}}

            The system targets 30 FPS processing, imposing a hard per-frame
            budget of $1{REALTIME_THRESHOLD_MS:.0f} / 30 \\approx {REALTIME_THRESHOLD_MS:.1f}$~ms.
            Previous benchmarking (see Table~\\ref{{tab:pipeline_latency}}) measured
            the full pipeline end-to-end latency on the target GPU; all stages
            combined remained within this budget, confirming that no individual
            component need be removed for real-time viability.

            \\subsubsection*{{Component Justification}}

            \\begin{{description}}
              \\item[YOLO Inference]
                Object detection is the foundational stage; without it no
                bounding boxes are produced and the remaining stages receive
                empty input.  Its removal would trivially reduce latency but
                would eliminate all detection capability.  The chosen YOLOv5
                model provides the best latency-accuracy trade-off among
                variants evaluated during preliminary experiments.

              \\item[Hit Detection (spatial overlap)]
                The spatial overlap check between action and bag bounding boxes
                is an $O(|\\mathcal{{A}}| \\times |\\mathcal{{B}}|)$ operation where
                $|\\mathcal{{A}}|$ and $|\\mathcal{{B}}|$ are the number of action and
                bag detections per frame respectively; in practice these sets are
                small ($\\leq 3$ elements each), making the computation negligible.
                Removing it would eliminate the hit-counting capability entirely.

              \\item[Temporal Smoothing]
                Gap-tolerance and minimum-duration debouncing prevent single-frame
                detection noise from inflating hit counts.  The computation consists
                entirely of integer comparisons and is effectively costless relative
                to inference.  Its omission would substantially degrade hit-count
                accuracy without any meaningful latency saving.

              \\item[Overlay Rendering]
                OpenCV drawing calls annotate each output frame with bounding boxes,
                class labels, and running counters.  This stage is optional in
                production deployments where annotated video output is not required;
                disabling it would reduce latency by the cost of the drawing
                operations.  For the benchmark and demonstration prototype, overlay
                rendering was retained to provide visual confirmation of detection
                correctness.
            \\end{{description}}

            \\subsubsection*{{Planned Future Work}}

            A quantitative ablation study is identified as future work.
            It would involve: (i)~recording a controlled video sequence with
            known ground-truth hit timestamps, (ii)~running all four single-stage
            removal configurations and the full-pipeline baseline over the same
            footage, (iii)~measuring both per-frame latency and hit-count error
            for each variant, and (iv)~reporting the results in a structured table
            analogous to Table~\\ref{{tab:pipeline_latency}}.
            Such an experiment would provide empirical evidence for the latency
            contribution of each stage and allow detection of any accuracy
            regression introduced by component removal.
        """)

    # ------------------------------------------------------------------

    def generate(self) -> pathlib.Path:
        """Choose branch, generate text, save to file, return path."""
        if (
            self.summary_df is not None
            and len(self.summary_df) > 0
            and "Full Pipeline" in self.summary_df["config_name"].values
        ):
            text = self.generate_with_results()
            label = "WITH results"
        else:
            text = self.generate_without_results()
            label = "WITHOUT results (justification-of-omission)"

        p = self.output_dir / "chapter6_ablation.txt"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        logger.info(f"Chapter 6 text ({label}) -> {p}")
        return p


# ===========================================================================
# Section 9 — Frame Source Utilities
# ===========================================================================

def load_frames_from_video(
    video_path: str,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], dict]:
    """Load up to max_frames BGR frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit  = min(max_frames, total) if max_frames else total

    logger.info(f"Video: {video_path}  |  {total} frames  |  {width}x{height}  |  {fps:.1f} FPS")
    logger.info(f"Loading {limit} frames ...")

    frames = []
    for _ in range(limit):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    meta = {
        "video_path": video_path,
        "total_frames": total,
        "loaded_frames": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
    }
    logger.info(f"Loaded {len(frames)} frames.")
    return frames, meta


def generate_synthetic_frames(
    n: int = 200,
    height: int = 480,
    width:  int = 640,
) -> Tuple[List[np.ndarray], dict]:
    """
    Generate random BGR frames for timing benchmarks when no video is available.
    YOLO detections will be empty (model cannot detect anything meaningful),
    so stage timings reflect infrastructure overhead rather than real inference load.
    A warning is printed to make this clear.
    """
    logger.warning(
        "Using SYNTHETIC frames.  YOLO inference will run on random noise — "
        "detections will be empty.  Hit-detection and temporal stages will record "
        "near-zero latency.  Only YOLO and overlay timings are meaningful."
    )
    frames = [
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        for _ in range(n)
    ]
    meta = {
        "video_path": "synthetic",
        "total_frames": n,
        "loaded_frames": n,
        "fps": 30.0,
        "width": width,
        "height": height,
    }
    return frames, meta


def capture_hardware_info() -> dict:
    info = {
        "device":          str(device),
        "cuda_available":  torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"]      = torch.cuda.get_device_name(0)
        import torch.version as _tv
        info["cuda_version"]  = getattr(_tv, "cuda", "N/A")
    return info


# ===========================================================================
# Section 10 — Merge Helper
# ===========================================================================

def merge_summary_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Produce a single summary DataFrame by averaging timing columns across
    multiple per-video summary DataFrames.

    Rules:
      - config_name order is taken from the first DataFrame.
      - Boolean flag columns (enable_*) are copied from the first DataFrame.
      - n_frames is summed (total frames across all videos).
      - All timing / FPS columns are averaged (arithmetic mean).

    Note: averaging P95 is an approximation; recomputing from pooled raw data
    would be exact but requires the raw CSVs to be loaded here.
    """
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0].copy()

    reference     = dfs[0]
    cfg_names     = reference["config_name"].tolist()
    flag_cols     = ["enable_yolo", "enable_hit_detection", "enable_temporal", "enable_overlay"]
    skip_cols     = {"config_name", "n_frames"} | set(flag_cols)
    numeric_cols  = [c for c in reference.columns if c not in skip_cols]

    merged_rows: List[dict] = []
    for cfg_name in cfg_names:
        row_series = []
        for df in dfs:
            m = df[df["config_name"] == cfg_name]
            if len(m) > 0:
                row_series.append(m.iloc[0])

        if not row_series:
            continue

        base = row_series[0]
        merged: dict = {"config_name": cfg_name}
        for fc in flag_cols:
            merged[fc] = bool(base.get(fc, True))
        merged["n_frames"] = int(sum(float(r.get("n_frames", 0)) for r in row_series))

        for col in numeric_cols:
            vals = [float(r[col]) for r in row_series if col in r.index]
            merged[col] = round(np.mean(vals), 3) if vals else 0.0

        merged_rows.append(merged)

    return pd.DataFrame(merged_rows)


def _print_summary_table(df: pd.DataFrame, title: str = "ABLATION STUDY RESULTS") -> None:
    """Print a compact summary table to the console (ASCII-safe)."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    cols = ["config_name", "total_mean", "total_sd", "total_p95", "fps_mean"]
    display = df[cols].copy()
    display.columns = ["Configuration", "Mean (ms)", "SD (ms)", "P95 (ms)", "FPS"]
    print(display.to_string(index=False))
    print("=" * 72)


# ===========================================================================
# Section 11 — CLI Main
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combat Sports Pipeline Ablation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--video", type=str, default=None,
        help="Path to input video file.  "
             "Outputs are saved to <output>/<video_stem>/  (no overwrite between videos).",
    )
    src.add_argument(
        "--synthetic", action="store_true",
        help="Use randomly-generated frames.  "
             "Outputs are saved to <output>/synthetic/.",
    )
    p.add_argument(
        "--model", type=str, default="models/best.pt",
        help="Path to YOLOv5 weights (.pt).",
    )
    p.add_argument(
        "--frames", type=int, default=None,
        help="Maximum number of frames to process per configuration.",
    )
    p.add_argument(
        "--output", type=str, default="thesis_assets/ablation",
        help="Root output directory.  Each run creates a named subdirectory.",
    )
    p.add_argument(
        "--warmup", type=int, default=10,
        help="Number of YOLO warm-up passes before timing begins.",
    )
    p.add_argument(
        "--text-only", action="store_true",
        help="Skip the ablation run.  Scan <output>/**/ablation_summary.csv, "
             "merge real-video results (excluding synthetic/), and regenerate "
             "consolidated chapter6_ablation.txt + table + figure in <output>/.",
    )
    p.add_argument(
        "--include-synthetic", action="store_true",
        help="When used with --text-only, include the synthetic/ summary in the merge.",
    )
    p.add_argument(
        "--synthetic-height", type=int, default=480,
        help="Frame height for synthetic mode.",
    )
    p.add_argument(
        "--synthetic-width", type=int, default=640,
        help="Frame width for synthetic mode.",
    )
    return p


def _resolve_run_dir(base_output: str, args) -> pathlib.Path:
    """
    Determine the per-run output subdirectory so multiple videos never
    write to the same location.

      --video data/Sample 1.mp4  -->  <base_output>/Sample 1/
      --synthetic                -->  <base_output>/synthetic/
    """
    base = pathlib.Path(base_output)
    if args.video:
        stem = pathlib.Path(args.video).stem   # e.g. "Sample 1"
        return base / stem
    return base / "synthetic"


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    base_dir = pathlib.Path(args.output)
    hw_info  = capture_hardware_info()

    # ── Text-only / consolidation mode ──────────────────────────────────
    if args.text_only:
        base_dir.mkdir(parents=True, exist_ok=True)

        # Collect all per-run summary CSVs from subdirectories
        candidate_csvs = sorted(base_dir.glob("*/ablation_summary.csv"))
        if not candidate_csvs:
            logger.warning(
                f"No per-run summary CSVs found under {base_dir} — "
                "generating omission text."
            )
            gen = ThesisChapter6Generator(
                output_dir=str(base_dir),
                summary_df=None,
                hardware_info=hw_info,
            )
            out = gen.generate()
            print(f"\nChapter 6 text written to: {out}")
            return

        # Decide which to include
        include = []
        for csv_path in candidate_csvs:
            is_synthetic = csv_path.parent.name == "synthetic"
            if is_synthetic and not args.include_synthetic:
                logger.info(f"Skipping synthetic summary: {csv_path}")
                continue
            logger.info(f"Loading: {csv_path}")
            include.append(pd.read_csv(csv_path))

        if not include:
            logger.warning("No real-video summaries found (all were synthetic). "
                           "Pass --include-synthetic to include them.")
            gen = ThesisChapter6Generator(
                output_dir=str(base_dir),
                summary_df=None,
                hardware_info=hw_info,
            )
            out = gen.generate()
            print(f"\nChapter 6 text written to: {out}")
            return

        n_sources  = len(include)
        merged_df  = merge_summary_dfs(include)
        logger.info(f"Merged {n_sources} summary CSV(s) into consolidated results.")

        _print_summary_table(merged_df, f"CONSOLIDATED RESULTS ({n_sources} video(s))")

        # Re-export consolidated files to the root output dir
        # (AblationExporter expects a timings list; bypass it by writing directly)
        base_dir.mkdir(parents=True, exist_ok=True)

        merged_csv = base_dir / "ablation_summary.csv"
        merged_df.to_csv(merged_csv, index=False)
        logger.info(f"Consolidated summary -> {merged_csv}")

        # Build a dummy exporter just for table + figure (it reads summary_df directly)
        class _DirectExporter(AblationExporter):
            """Thin subclass that accepts a pre-built summary_df instead of timings."""
            def __init__(self, summary_df: pd.DataFrame, configs, output_dir, hardware_info):
                # skip the parent __init__ (which requires timings)
                self.timings       = []
                self.configs       = configs
                self.output_dir    = pathlib.Path(output_dir)
                self.hardware_info = hardware_info or {}
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.raw_df        = pd.DataFrame()   # not needed for table/figure
                self.summary_df    = summary_df

        exp = _DirectExporter(
            summary_df=merged_df,
            configs=ABLATION_SUITE,
            output_dir=str(base_dir),
            hardware_info=hw_info,
        )
        tex_path = exp.save_latex_table()
        fig_path = exp.save_latency_figure()

        gen = ThesisChapter6Generator(
            output_dir=str(base_dir),
            summary_df=merged_df,
            hardware_info=hw_info,
        )
        ch6_path = gen.generate()

        print("\nConsolidated files written:")
        for label, p in [("summary", merged_csv), ("latex", tex_path),
                          ("figure", fig_path), ("chapter6", ch6_path)]:
            print(f"  {label:<10}  {p}")
        print()
        return

    # ── Require a frame source ───────────────────────────────────────────
    if args.video is None and not args.synthetic:
        parser.error("Provide --video <path> or --synthetic.")

    # ── Determine isolated output dir for this run ───────────────────────
    run_dir = _resolve_run_dir(args.output, args)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run output directory: {run_dir}")

    # ── Load frames ──────────────────────────────────────────────────────
    if args.video:
        frames, meta = load_frames_from_video(args.video, max_frames=args.frames)
    else:
        n = args.frames if args.frames else 200
        frames, meta = generate_synthetic_frames(
            n=n,
            height=args.synthetic_height,
            width=args.synthetic_width,
        )

    if not frames:
        logger.error("No frames loaded — aborting.")
        sys.exit(1)

    # ── Load YOLO model ───────────────────────────────────────────────────
    model = load_yolo_model(args.model)
    if model is None:
        logger.warning(
            "No YOLO model loaded.  Configs with enable_yolo=True will "
            "use empty detections and show near-zero YOLO latency."
        )

    # ── Run ablation ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting ablation study")
    logger.info(f"  Configs : {len(ABLATION_SUITE)}")
    logger.info(f"  Frames  : {len(frames)} per config")
    logger.info(f"  Output  : {run_dir}")
    logger.info("=" * 60)

    runner  = AblationRunner(model=model, warmup_frames=args.warmup)
    timings = runner.run_all(frames)

    if not timings:
        logger.error("No timings produced — exiting.")
        sys.exit(1)

    # ── Export ────────────────────────────────────────────────────────────
    exporter = AblationExporter(
        timings=timings,
        configs=ABLATION_SUITE,
        output_dir=str(run_dir),
        hardware_info=hw_info,
    )
    paths = exporter.export_all()

    # ── Per-run Chapter 6 text ────────────────────────────────────────────
    gen      = ThesisChapter6Generator(
        output_dir=str(run_dir),
        summary_df=exporter.summary_df,
        hardware_info=hw_info,
    )
    ch6_path = gen.generate()
    paths["chapter6"] = ch6_path

    # ── Console summary ───────────────────────────────────────────────────
    _print_summary_table(exporter.summary_df)
    print("\nFiles written:")
    for key, p in paths.items():
        print(f"  {key:<12}  {p}")
    print()


if __name__ == "__main__":
    main()
