"""Spatially-conditioned sensor model for YOLO detector.

This implements the "creative direction ①" from 项目方案与代码分析.md: instead
of using a single global TP / FP scalar pair for the Bayes update, we measure
the detector's reliability at different (range, viewing angle) bins per object
class, and use those bin-specific rates in the belief update.

The collected JSON looks like:

    {
      "class": "remote control",
      "bins": {
        "0_30":  {"tp": 0.96, "fp": 0.02, "n_pos": 120, "n_neg": 4500},  # 0-1m, +-30deg
        "0_60":  {"tp": 0.90, "fp": 0.03, "n_pos": 80,  "n_neg": 3000},  # 0-1m, +-30..60deg
        "1_30":  {"tp": 0.75, "fp": 0.02, ...},                          # 1-2m, +-30deg
        ...
      },
      "n_views_total": 5000
    }

`bin = "<dist_bin>_<angle_bin>"` where dist_bin is the integer meter floor of
the object's distance and angle_bin is the absolute relative angle in
30-degree buckets.

Usage:
    # offline collection (long-running; uses metadata as oracle ground truth)
    python sensor_model.py collect \\
        --classes "remote control,book,mug,television" \\
        --scenes FloorPlan201,FloorPlan202,FloorPlan203 \\
        --views-per-scene 200 --out outputs/sensor_model.json

    # at inference time
    from sensor_model import SensorModel
    sm = SensorModel.load("outputs/sensor_model.json")
    tp, fp = sm.lookup(target_class="remote control",
                       distance=1.7, rel_angle_deg=22.0)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Binning
# ============================================================

def _dist_bin(d: float, max_dist: float = 5.0) -> int:
    """Integer-meter bucket, clamped to [0, max_dist)."""
    return max(0, min(int(max_dist) - 1, int(math.floor(d))))


def _angle_bin(theta_deg: float) -> int:
    """|theta| bucket in 30deg increments: 0=[0,30), 1=[30,60), 2=[60,90)."""
    a = abs(theta_deg)
    if a >= 90.0:
        return 2
    return int(a // 30.0)


def bin_key(d: float, theta_deg: float) -> str:
    return f"{_dist_bin(d)}_{_angle_bin(theta_deg) * 30}"


# ============================================================
# SensorModel data class
# ============================================================

@dataclass
class _BinStats:
    n_pos: int = 0          # frames where target is in view from this bin
    n_pos_detected: int = 0 # frames where detector also fired on target
    n_neg: int = 0          # frames where target is NOT in this bin range
    n_neg_falseposed: int = 0  # frames where detector falsely fired on this class

    @property
    def tp(self) -> float:
        return self.n_pos_detected / max(self.n_pos, 1)

    @property
    def fp(self) -> float:
        return self.n_neg_falseposed / max(self.n_neg, 1)


class SensorModel:
    """Per-class, per-bin TP/FP table with safe fallbacks."""

    DEFAULT_TP = 0.85
    DEFAULT_FP = 0.05

    def __init__(self, data: Dict):
        self.data = data
        self.classes = list(data.keys())

    @classmethod
    def load(cls, path: str) -> "SensorModel":
        with open(path) as f:
            return cls(json.load(f))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    def lookup(
        self,
        target_class: str,
        distance: float,
        rel_angle_deg: float,
        min_obs: int = 5,
    ) -> Tuple[float, float]:
        """Return (TP, FP) for this (class, bin).

        Falls back to defaults if the bin has too few observations.
        """
        cls_data = self.data.get(target_class)
        if not cls_data:
            return self.DEFAULT_TP, self.DEFAULT_FP
        bins = cls_data.get("bins", {})
        k = bin_key(distance, rel_angle_deg)
        b = bins.get(k)
        if not b:
            return self.DEFAULT_TP, self.DEFAULT_FP
        if b.get("n_pos", 0) < min_obs:
            return self.DEFAULT_TP, self.DEFAULT_FP
        return float(b["tp"]), float(b["fp"])

    @staticmethod
    def synthetic_spatial_model(
        max_distance: float = 5.0,
        tp_near: float = 0.95,
        tp_far: float = 0.30,
        fp_rate: float = 0.04,
        angle_falloff: float = 0.5,
    ):
        """Return a spatial sensor-model callable based on simple physics.

        TP decays exponentially with distance and with absolute viewing angle:
            tp(d, a) = tp_far + (tp_near - tp_far) * exp(-d / lambda)
                       * (1 - angle_falloff * |a|/half_fov)

        Use this when no YOLO-measured data is available yet — it still
        captures the qualitative "far / off-centre = unreliable" behaviour
        that motivates the spatial sensor model contribution.
        """
        lam = max_distance / 3.0  # decay length

        def _model(d_grid: np.ndarray, a_grid: np.ndarray):
            d = np.clip(d_grid, 0, max_distance)
            tp_dist = tp_far + (tp_near - tp_far) * np.exp(-d / lam)
            angle_factor = 1.0 - angle_falloff * np.minimum(
                1.0, np.abs(a_grid) / (np.pi / 3.0)
            )
            tp = tp_dist * angle_factor
            fp = np.full_like(d_grid, fp_rate, dtype=np.float32)
            return tp.astype(np.float32), fp.astype(np.float32)

        return _model

    def as_spatial_callable(self, target_class: str):
        """Return a callable compatible with BeliefGrid.update_negative_observation.

        Signature: (distance_grid_HxW, rel_angle_rad_HxW) -> (tp_grid, fp_grid)
        """
        cls_data = self.data.get(target_class, {})
        bins = cls_data.get("bins", {})
        tp_default = self.DEFAULT_TP
        fp_default = self.DEFAULT_FP

        def _model(d_grid: np.ndarray, a_grid: np.ndarray):
            # vectorised lookup: build small (dist_bin, angle_bin) maps once
            tp_out = np.full_like(d_grid, tp_default, dtype=np.float32)
            fp_out = np.full_like(d_grid, fp_default, dtype=np.float32)
            angle_deg = np.degrees(np.abs(a_grid))
            for k, b in bins.items():
                if b.get("n_pos", 0) < 5:
                    continue
                dbin_str, abin_str = k.split("_")
                dbin = int(dbin_str)
                abin_lo = int(abin_str)
                abin_hi = abin_lo + 30
                mask = (
                    (np.floor(d_grid) == dbin)
                    & (angle_deg >= abin_lo)
                    & (angle_deg < abin_hi)
                )
                tp_out[mask] = float(b["tp"])
                fp_out[mask] = float(b["fp"])
            return tp_out, fp_out

        return _model


# ============================================================
# Collection
# ============================================================

def _ai2thor_to_prompt(object_type: str) -> str:
    import re
    mapping = {
        "Television": "tv",
        "RemoteControl": "remote control",
        "TVStand": "tv stand",
        "DiningTable": "dining table",
        "CoffeeTable": "coffee table",
        "SideTable": "side table",
        "DeskLamp": "desk lamp",
        "FloorLamp": "floor lamp",
        "HousePlant": "house plant",
        "GarbageCan": "garbage can",
    }
    if object_type in mapping:
        return mapping[object_type]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", object_type).lower()


def _prompt_to_types(prompt: str) -> List[str]:
    rev = {
        "tv": "Television",
        "remote control": "RemoteControl",
        "tv stand": "TVStand",
        "dining table": "DiningTable",
        "coffee table": "CoffeeTable",
        "side table": "SideTable",
    }
    if prompt in rev:
        return [rev[prompt]]
    return ["".join(w.capitalize() for w in prompt.split())]


def collect_sensor_stats(
    target_prompts: List[str],
    scenes: List[str],
    views_per_scene: int = 100,
    seed: int = 0,
    out_path: str = "outputs/sensor_model.json",
    detector_factory=None,
) -> str:
    """Random walk through scenes, count YOLO TP/FP at each (class, bin).

    detector_factory: callable returning an object with a .detect(rgb) method
        that yields a list of dicts with keys "label", "score".  If None,
        uses the metadata detector — perfect TP/FP=1/0, useful only for
        plumbing tests.
    """
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    rng = random.Random(seed)

    # initialise stats: {class -> {bin_key -> _BinStats}}
    stats: Dict[str, Dict[str, _BinStats]] = {
        tp: {} for tp in target_prompts
    }

    # active classes lookup
    target_ai2thor_types = {tp: _prompt_to_types(tp) for tp in target_prompts}

    detector = detector_factory() if detector_factory else None

    n_views_total = 0
    t_start = time.time()

    for scene in scenes:
        print(f"[collect] scene {scene}", flush=True)
        ctrl = Controller(
            platform=CloudRendering,
            scene=scene,
            width=320, height=240,
            gridSize=0.25, rotateStepDegrees=90,
            visibilityDistance=5.0,
            fieldOfView=60,
            renderDepthImage=False,
        )
        try:
            ev = ctrl.step("GetReachablePositions")
            reachable = ev.metadata["actionReturn"] or []
            if not reachable:
                continue

            for v in range(views_per_scene):
                sp = rng.choice(reachable)
                yaw = rng.choice([0, 90, 180, 270])
                ev = ctrl.step(
                    "Teleport",
                    position=sp,
                    rotation={"x": 0, "y": yaw, "z": 0},
                    horizon=rng.choice([-30, 0, 30]),
                )
                if not ev.metadata["lastActionSuccess"]:
                    continue
                n_views_total += 1

                # which target classes are visible in this view (oracle)?
                visible_by_type = {}
                for obj in ev.metadata["objects"]:
                    if not obj.get("visible"):
                        continue
                    ot = obj.get("objectType")
                    pos = obj.get("position")
                    if pos is None:
                        continue
                    d = math.hypot(
                        pos["x"] - sp["x"], pos["z"] - sp["z"],
                    )
                    # rel angle
                    dx = pos["x"] - sp["x"]
                    dz = pos["z"] - sp["z"]
                    abs_angle = math.degrees(
                        math.atan2(dx, dz) - math.radians(yaw)
                    )
                    abs_angle = (abs_angle + 180) % 360 - 180
                    visible_by_type.setdefault(ot, []).append({
                        "distance": d, "angle_deg": abs_angle,
                    })

                # detector predictions
                if detector is not None:
                    rgb = ev.frame
                    detections = detector.detect(rgb) or []
                else:
                    # oracle: treats visible_by_type as detections
                    detections = [
                        {"label": _ai2thor_to_prompt(ot), "score": 1.0}
                        for ot in visible_by_type
                    ]

                detected_prompts = {d["label"].lower() for d in detections}

                # update stats per target class
                for tp, ai2t_types in target_ai2thor_types.items():
                    instances = []
                    for at in ai2t_types:
                        instances += visible_by_type.get(at, [])
                    detected = tp.lower() in detected_prompts

                    if instances:
                        # pick the nearest visible instance — that's "the" target
                        inst = min(instances, key=lambda x: x["distance"])
                        bk = bin_key(inst["distance"], inst["angle_deg"])
                        bs = stats[tp].setdefault(bk, _BinStats())
                        bs.n_pos += 1
                        if detected:
                            bs.n_pos_detected += 1
                    else:
                        # target absent: count toward FP for all bins equally
                        for bk in list(stats[tp].keys()) or ["0_0"]:
                            bs = stats[tp].setdefault(bk, _BinStats())
                            bs.n_neg += 1
                            if detected:
                                bs.n_neg_falseposed += 1

                if (v + 1) % 50 == 0:
                    dt = time.time() - t_start
                    print(
                        f"  [{n_views_total}] scene {scene}  v={v + 1}  "
                        f"elapsed={dt:.1f}s",
                        flush=True,
                    )

        finally:
            ctrl.stop()

    # serialize
    out = {}
    for cls, bins in stats.items():
        out[cls] = {
            "bins": {
                k: {
                    "tp": bs.tp, "fp": bs.fp,
                    "n_pos": bs.n_pos,
                    "n_pos_detected": bs.n_pos_detected,
                    "n_neg": bs.n_neg,
                    "n_neg_falseposed": bs.n_neg_falseposed,
                }
                for k, bs in bins.items()
            },
            "n_views_total": n_views_total,
        }

    sm = SensorModel(out)
    sm.save(out_path)
    print(f"saved sensor model to {out_path}")
    return out_path


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--classes", required=True,
                    help="comma-separated target prompt list")
    pc.add_argument("--scenes", required=True,
                    help="comma-separated scenes")
    pc.add_argument("--views-per-scene", type=int, default=100)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--detector", choices=["oracle", "yolo"],
                    default="oracle")
    pc.add_argument("--out", required=True)

    ps = sub.add_parser("show")
    ps.add_argument("--path", required=True)

    args = p.parse_args()

    if args.cmd == "collect":
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
        if args.detector == "yolo":
            from yolo_detector import YoloWorldDetector

            def factory():
                return YoloWorldDetector(
                    classes=classes,
                    model_name="yolov8s-world.pt",
                    conf=0.10, device="cpu",
                )
        else:
            factory = None

        collect_sensor_stats(
            target_prompts=classes,
            scenes=scenes,
            views_per_scene=args.views_per_scene,
            seed=args.seed,
            out_path=args.out,
            detector_factory=factory,
        )
    elif args.cmd == "show":
        sm = SensorModel.load(args.path)
        for cls in sm.classes:
            print(f"=== {cls} ===")
            for k, b in sorted(sm.data[cls]["bins"].items()):
                if b["n_pos"] < 5:
                    continue
                print(
                    f"  bin {k:>6s}  tp={b['tp']:.3f}  fp={b['fp']:.4f}  "
                    f"n_pos={b['n_pos']}  n_neg={b['n_neg']}"
                )


if __name__ == "__main__":
    main()
