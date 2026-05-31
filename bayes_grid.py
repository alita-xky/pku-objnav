"""Bayesian belief grid for object goal navigation.

Replaces the original per-reachable-point scalar belief with an explicit
2D probability distribution over the scene floor plane. Three things make
this version different from the baseline `bayes_search.py`:

  1. Belief is a proper distribution over (H x W) cells — naturally supports
     blurring, marginalization, and Shannon entropy.

  2. Update step uses an explicit sensor model.  When the agent looks at a
     region V and does NOT detect the target, every cell c in V gets
     multiplied by the likelihood ratio (1 - TP_c) / (1 - FP_c) instead of
     by a magic constant. TP_c and FP_c default to scalars but a spatial
     sensor model (distance x angle x class) can be passed in.

  3. Information-gain-driven next-view selection.  For each candidate
     viewpoint we compute
            IG(v) = H(b) - E_D[H(b | D, v)]
     exactly under the simplified two-state model { detect | no detect },
     and pick the viewpoint maximising IG(v) - lambda * travel_cost(v).

The grid is anchored on the AI2-THOR scene's reachable positions extent
plus a small margin, with resolution matching `gridSize` (0.25m). Non
reachable cells stay at zero mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Belief grid
# ============================================================

@dataclass
class BeliefGridConfig:
    resolution: float = 0.25        # meters per cell
    margin: float = 2.0             # meters past reachable extent
    init_concentration: float = 0.0 # >0 to put extra mass on reachable cells
    eps: float = 1e-12


class BeliefGrid:
    """2D probability grid b[i, j] = P(target in cell c_{i,j} | obs)."""

    def __init__(
        self,
        reachable_positions: List[Dict[str, float]],
        config: Optional[BeliefGridConfig] = None,
    ):
        self.cfg = config or BeliefGridConfig()
        xs = [p["x"] for p in reachable_positions]
        zs = [p["z"] for p in reachable_positions]

        if not xs:
            raise ValueError("reachable_positions is empty")

        self.x_min = float(min(xs) - self.cfg.margin)
        self.x_max = float(max(xs) + self.cfg.margin)
        self.z_min = float(min(zs) - self.cfg.margin)
        self.z_max = float(max(zs) + self.cfg.margin)

        self.W = int(math.ceil((self.x_max - self.x_min) / self.cfg.resolution))
        self.H = int(math.ceil((self.z_max - self.z_min) / self.cfg.resolution))

        # cell-centre world coordinates, precomputed for vectorised updates
        self._xs = self.x_min + (np.arange(self.W) + 0.5) * self.cfg.resolution
        self._zs = self.z_min + (np.arange(self.H) + 0.5) * self.cfg.resolution
        self._x_grid, self._z_grid = np.meshgrid(self._xs, self._zs)

        # reachable mask
        self.reachable_mask = np.zeros((self.H, self.W), dtype=bool)
        for p in reachable_positions:
            i, j = self.world_to_cell(p["x"], p["z"])
            if 0 <= i < self.H and 0 <= j < self.W:
                self.reachable_mask[i, j] = True

        # uniform prior over reachable cells
        self.belief = self.reachable_mask.astype(np.float32)
        if self.cfg.init_concentration > 0:
            self.belief += self.cfg.init_concentration
            self.belief *= self.reachable_mask
        self._normalize()

    # ---- coordinate helpers -------------------------------------------------

    def world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        j = int((x - self.x_min) / self.cfg.resolution)
        i = int((z - self.z_min) / self.cfg.resolution)
        return i, j

    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x = self.x_min + (j + 0.5) * self.cfg.resolution
        z = self.z_min + (i + 0.5) * self.cfg.resolution
        return x, z

    # ---- distribution maintenance ------------------------------------------

    def _normalize(self) -> None:
        s = float(self.belief.sum())
        if s > self.cfg.eps:
            self.belief /= s
        # if all mass evaporates (numerical issues), reset to uniform on
        # reachable cells so we don't propagate NaNs
        else:
            self.belief = self.reachable_mask.astype(np.float32)
            s = float(self.belief.sum())
            if s > 0:
                self.belief /= s

    def entropy(self) -> float:
        p = self.belief[self.belief > self.cfg.eps]
        return float(-np.sum(p * np.log(p)))

    def map_estimate(self) -> Tuple[Dict[str, float], float]:
        """Most-likely cell and its mass."""
        idx = int(np.argmax(self.belief))
        i, j = divmod(idx, self.W)
        x, z = self.cell_to_world(i, j)
        return {"x": x, "z": z}, float(self.belief[i, j])

    # ---- visibility -------------------------------------------------------

    def visible_mask(
        self,
        agent_pose: Dict,
        fov_deg: float = 60.0,
        max_distance: float = 1.5,
    ) -> np.ndarray:
        """Boolean mask of cells inside the agent's FOV cone.

        AI2-THOR convention: rotation.y = 0 means the agent faces +z.
        """
        ax = agent_pose["position"]["x"]
        az = agent_pose["position"]["z"]
        yaw_rad = math.radians(agent_pose["rotation"]["y"])

        dx = self._x_grid - ax
        dz = self._z_grid - az
        d = np.sqrt(dx * dx + dz * dz)

        angle = np.arctan2(dx, dz) - yaw_rad
        angle = (angle + math.pi) % (2 * math.pi) - math.pi

        half_fov = math.radians(fov_deg) / 2.0
        return (d <= max_distance) & (np.abs(angle) <= half_fov)

    # ---- updates ----------------------------------------------------------

    def update_negative_observation(
        self,
        agent_pose: Dict,
        fov_deg: float = 60.0,
        max_distance: float = 1.5,
        tp_rate: float = 0.85,
        fp_rate: float = 0.05,
        spatial_sensor_model: Optional[Callable] = None,
    ) -> None:
        """Bayes update assuming the agent looked at FOV V and saw no target.

        Likelihood ratio for cell c in V:
            P(D=0 | target in c)        1 - TP
            ----------------------- = ----------
            P(D=0 | target not in c)    1 - FP

        Cells outside V get likelihood 1 (no information).

        spatial_sensor_model: optional callable returning (tp, fp) per cell.
            signature: (distance_grid, rel_angle_grid) -> (tp_grid, fp_grid)
        """
        ax = agent_pose["position"]["x"]
        az = agent_pose["position"]["z"]
        yaw_rad = math.radians(agent_pose["rotation"]["y"])

        dx = self._x_grid - ax
        dz = self._z_grid - az
        d = np.sqrt(dx * dx + dz * dz)
        angle = np.arctan2(dx, dz) - yaw_rad
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        half_fov = math.radians(fov_deg) / 2.0
        visible = (d <= max_distance) & (np.abs(angle) <= half_fov)

        if spatial_sensor_model is not None:
            tp_grid, fp_grid = spatial_sensor_model(d, angle)
        else:
            tp_grid = np.full_like(self.belief, tp_rate, dtype=np.float32)
            fp_grid = np.full_like(self.belief, fp_rate, dtype=np.float32)

        miss_ratio = (1.0 - tp_grid) / np.clip(1.0 - fp_grid, self.cfg.eps, None)
        self.belief = np.where(visible, self.belief * miss_ratio, self.belief)
        self.belief *= self.reachable_mask
        self._normalize()

    def update_positive_observation(
        self,
        target_position: Dict[str, float],
        tp_rate: float = 0.85,
        fp_rate: float = 0.05,
        spread_sigma: float = 0.6,
        bump_radius: float = 1.0,
    ) -> None:
        """Bayes update when the target detector fires near `target_position`.

        We do not assume the position estimate is exact; we spread the
        positive evidence over a Gaussian ball of stdev `spread_sigma` and
        clip at `bump_radius` to stay local.
        """
        tx = target_position["x"]
        tz = target_position["z"]
        d = np.sqrt((self._x_grid - tx) ** 2 + (self._z_grid - tz) ** 2)
        bump = np.exp(-0.5 * (d / spread_sigma) ** 2).astype(np.float32)
        bump *= (d <= bump_radius)

        ratio = tp_rate / max(fp_rate, self.cfg.eps)
        likelihood = 1.0 + (ratio - 1.0) * bump
        self.belief *= likelihood
        self.belief *= self.reachable_mask
        self._normalize()

    def update_context_detection(
        self,
        context_position: Dict[str, float],
        prior_weight: float,
        detector_confidence: float = 1.0,
        spread_sigma: float = 1.2,
        bump_radius: float = 3.0,
        gain: float = 3.0,
    ) -> None:
        """Spread positive belief around a detected context object.

        Likelihood multiplier at cell c with distance d from the context is
            L(c) = exp( gain * w * conf * exp(-d^2 / (2 sigma^2)) )
        clipped by `bump_radius`.  At the bump centre and for w*conf=1 this
        gives a likelihood ratio of e^gain (≈20 for gain=3), strong enough
        for a single context detection to visibly concentrate belief while
        keeping the formula monotonic and renormalisable.
        """
        if prior_weight <= 0:
            return
        cx = context_position["x"]
        cz = context_position["z"]
        d = np.sqrt((self._x_grid - cx) ** 2 + (self._z_grid - cz) ** 2)
        bump = np.exp(-0.5 * (d / spread_sigma) ** 2).astype(np.float32)
        bump *= (d <= bump_radius)
        log_likelihood = gain * prior_weight * detector_confidence * bump
        # numerically stable: subtract max to avoid overflow before exp
        log_likelihood -= log_likelihood.max() if log_likelihood.size else 0.0
        likelihood = np.exp(log_likelihood)
        self.belief *= likelihood
        self.belief *= self.reachable_mask
        self._normalize()

    # ---- viewpoint scoring (for IG-strategy ablation) ----------------------

    def belief_at_position(self, position: Dict[str, float]) -> float:
        """Belief at the cell containing world (x, z).

        Used by the "greedy max-belief" viewpoint selection strategy: just
        head for the cell with the highest target-probability mass.
        """
        i, j = self.world_to_cell(position["x"], position["z"])
        if 0 <= i < self.H and 0 <= j < self.W:
            return float(self.belief[i, j])
        return 0.0

    def belief_in_fov(
        self,
        viewpoint_pose: Dict,
        fov_deg: float = 60.0,
        max_distance: float = 1.5,
    ) -> float:
        """Total belief mass inside the viewpoint's FOV cone.

        Used by the "IPPON-style heuristic gain" viewpoint selection
        strategy: pick the viewpoint that has the most target-probability
        mass in view (without computing entropy).
        """
        mask = self.visible_mask(
            viewpoint_pose, fov_deg=fov_deg, max_distance=max_distance,
        )
        return float(self.belief[mask].sum())

    # ---- information gain --------------------------------------------------

    def expected_information_gain(
        self,
        candidate_pose: Dict,
        fov_deg: float = 60.0,
        max_distance: float = 1.5,
        tp_rate: float = 0.85,
        fp_rate: float = 0.05,
    ) -> float:
        """E[H(b) - H(b | D, v)] under the simplified two-state model.

        D in {0, 1}: did the detector fire for the target this step?

            P(D=1 | v) = sum_c b(c) * TP * [c in V] + sum_c b(c) * FP * [c not in V]

        Posterior under D=1:
            b(c | D=1) ∝ b(c) * (TP if c in V else FP)
        Posterior under D=0:
            b(c | D=0) ∝ b(c) * ((1-TP) if c in V else (1-FP))

        IG = H(b) - P(D=1|v) * H(b|D=1) - P(D=0|v) * H(b|D=0)
        """
        visible = self.visible_mask(
            candidate_pose, fov_deg=fov_deg, max_distance=max_distance,
        )

        b = self.belief.astype(np.float64)
        if b.sum() < self.cfg.eps:
            return 0.0

        # Probability of detection
        mass_in = float(b[visible].sum())
        mass_out = 1.0 - mass_in
        p_d1 = tp_rate * mass_in + fp_rate * mass_out
        p_d0 = 1.0 - p_d1

        # Posteriors
        w1 = np.where(visible, tp_rate, fp_rate)
        b1 = b * w1
        s1 = float(b1.sum())
        if s1 > self.cfg.eps:
            b1 /= s1
        else:
            b1 = b

        w0 = np.where(visible, 1.0 - tp_rate, 1.0 - fp_rate)
        b0 = b * w0
        s0 = float(b0.sum())
        if s0 > self.cfg.eps:
            b0 /= s0
        else:
            b0 = b

        def _H(p):
            pp = p[p > self.cfg.eps]
            return float(-np.sum(pp * np.log(pp)))

        H_prior = _H(b)
        H1 = _H(b1)
        H0 = _H(b0)
        ig = H_prior - p_d1 * H1 - p_d0 * H0
        return float(ig)

    # ---- visualization -----------------------------------------------------

    def save_heatmap(
        self,
        path: str,
        agent_pose: Optional[Dict] = None,
        targets: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Render belief + agent + targets to a PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        # display heatmap aligned with world coords; origin at lower-left
        ax.imshow(
            self.belief,
            origin="lower",
            extent=(self.x_min, self.x_max, self.z_min, self.z_max),
            cmap="viridis",
            interpolation="nearest",
        )
        ax.contour(
            self._x_grid,
            self._z_grid,
            self.reachable_mask.astype(np.float32),
            levels=[0.5],
            colors="white",
            linewidths=0.5,
            alpha=0.5,
        )

        if agent_pose is not None:
            ax.plot(
                agent_pose["position"]["x"], agent_pose["position"]["z"],
                "r^", markersize=12, label="agent",
            )

        if targets:
            for t in targets:
                ax.plot(t["x"], t["z"], "yo", markersize=10,
                        markerfacecolor="none", markeredgewidth=2,
                        label="target" if t is targets[0] else None)

        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(f"belief  H={self.entropy():.3f}")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close(fig)


# ============================================================
# Smoke test
# ============================================================

def _smoke():
    """Self test on a small synthetic scene."""
    np.random.seed(0)
    reachable = [
        {"x": x * 0.25, "y": 0.0, "z": z * 0.25}
        for x in range(-10, 11) for z in range(-10, 11)
        if (x * x + z * z) < 80
    ]
    grid = BeliefGrid(reachable)
    print(f"grid size: {grid.H} x {grid.W}")
    print(f"reachable cells: {int(grid.reachable_mask.sum())}")
    print(f"H_initial: {grid.entropy():.3f}")

    # context detection at (1.5, 0.5)
    grid.update_context_detection(
        context_position={"x": 1.5, "y": 0, "z": 0.5},
        prior_weight=0.9,
        detector_confidence=0.7,
        spread_sigma=1.0,
    )
    print(f"H_after_context: {grid.entropy():.3f}")
    map_pt, map_mass = grid.map_estimate()
    print(f"MAP after context: {map_pt} mass={map_mass:.4f}")

    # negative observation at (0,0) looking +z
    grid.update_negative_observation(
        agent_pose={"position": {"x": 0, "y": 0, "z": 0},
                    "rotation": {"y": 0}},
        fov_deg=60, max_distance=1.5,
    )
    print(f"H_after_neg: {grid.entropy():.3f}")

    # IG candidates
    candidates = [
        {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"y": yaw}}
        for yaw in [0, 90, 180, 270]
    ]
    for c in candidates:
        ig = grid.expected_information_gain(c)
        print(f"IG yaw={c['rotation']['y']:>4d}: {ig:.4f}")

    print("smoke test OK")


if __name__ == "__main__":
    _smoke()
