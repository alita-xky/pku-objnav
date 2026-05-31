"""Unified prior backend for context-prompt probability scoring.

Why: the original repo had two parallel prior tables — a hardcoded 8-class
dict (CONTEXT_PRIOR in run_yolo_bayes_approach.py and bayes_search.py) and a
much richer offline PMI-based prior in outputs/prior_build/semantic_prior.json
constructed by build_ai2thor_prior.py. The richer prior was queried only to
pick YOLO classes; it never reached the Bayesian update step. This module
unifies both behind one interface so they can be swapped via env var.

Usage:
    from prior_backend import make_prior
    prior = make_prior("pmi")          # default — uses semantic_prior.json
    prior = make_prior("hardcoded")    # baseline — original 8-class dict
    ctxs = prior.get_context_prompts("remote control")
    w = prior.get_context_weight("remote control", "sofa")

Env var:
    PRIOR_BACKEND=hardcoded|pmi
"""

from typing import Dict, List, Tuple

try:
    from semantic_prior import SemanticPrior  # type: ignore
except Exception:  # pragma: no cover
    SemanticPrior = None  # type: ignore


# ---------- label normalization & alias matching ----------

def norm_label(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")


_ALIAS_GROUPS = [
    {"sofa", "couch"},
    {"tv", "television", "monitor", "screen"},
    {"remotecontrol", "remote", "remotecontroller"},
    {"coffeetable", "table"},
    {"diningtable", "table"},
    {"houseplant", "plant"},
    {"garbagecan", "trashcan", "bin"},
    {"cellphone", "phone"},
    {"armchair", "chair"},
]


def label_match(a: str, b: str) -> bool:
    na = norm_label(a)
    nb = norm_label(b)
    if na == nb:
        return True
    for group in _ALIAS_GROUPS:
        if na in group and nb in group:
            return True
    return False


# ---------- backend 1: hardcoded (baseline) ----------

HARDCODED_CONTEXT_PRIOR: Dict[str, Dict[str, float]] = {
    "pencil": {"pen": 1.0, "dining table": 0.9, "desk": 0.9,
               "book": 0.5, "newspaper": 0.5, "laptop": 0.4},
    "pen": {"pencil": 1.0, "dining table": 0.9, "desk": 0.9,
            "book": 0.5, "newspaper": 0.5},
    "remotecontrol": {"sofa": 1.0, "couch": 1.0, "tv": 0.9,
                      "television": 0.9, "coffee table": 0.8,
                      "tv stand": 0.7},
    "book": {"shelf": 1.0, "desk": 0.8, "dining table": 0.6,
             "coffee table": 0.6, "sofa": 0.4, "couch": 0.4},
    "laptop": {"desk": 1.0, "dining table": 0.8, "chair": 0.5},
    "mug": {"dining table": 1.0, "coffee table": 0.9,
            "side table": 0.7, "desk": 0.7},
    "sofa": {"tv": 0.9, "television": 0.9, "coffee table": 0.8,
             "remote control": 0.7, "pillow": 0.6},
    "tv": {"sofa": 0.9, "couch": 0.9, "coffee table": 0.7,
           "tv stand": 1.0},
}


class HardcodedPrior:
    """Original hardcoded 8-class prior — kept for ablation/baseline."""

    name = "hardcoded"

    def __init__(self):
        self.table = HARDCODED_CONTEXT_PRIOR

    def get_context_prompts(self, target_prompt: str) -> List[str]:
        return list(self.table.get(norm_label(target_prompt), {}).keys())

    def get_context_weight(self, target_prompt: str,
                           context_prompt: str) -> float:
        sub = self.table.get(norm_label(target_prompt), {})
        for k, v in sub.items():
            if label_match(k, context_prompt):
                return v
        return 0.0


# ---------- backend 2: PMI from semantic_prior.json ----------

class SemanticPriorBackend:
    """PMI-based prior loaded from outputs/prior_build/semantic_prior.json.

    Scores are normalized per target so the strongest context has weight 1.0,
    making it numerically compatible with the hardcoded prior (which also
    peaks at 1.0). This keeps downstream evidence formulas unchanged.
    """

    name = "pmi"

    def __init__(
        self,
        prior_path: str = "outputs/prior_build/semantic_prior.json",
        top_k_contexts: int = 8,
        min_score: float = 0.05,
    ):
        if SemanticPrior is None:
            raise ImportError("semantic_prior module not available")
        self._prior = SemanticPrior(prior_path)
        self._top_k = top_k_contexts
        self._min_score = min_score
        self._cache: Dict[str, List[Tuple[str, float]]] = {}

    def _get(self, target_prompt: str) -> List[Tuple[str, float]]:
        k = norm_label(target_prompt)
        if k in self._cache:
            return self._cache[k]
        raw = self._prior.get_navigation_contexts_for_query(
            query=target_prompt,
            top_k_contexts=self._top_k,
            min_score=self._min_score,
        )
        # Normalize so the strongest context has weight = 1.0.
        if raw:
            max_s = max(s for _, s in raw)
            if max_s > 0:
                raw = [(c, s / max_s) for c, s in raw]
        self._cache[k] = list(raw)
        return self._cache[k]

    def get_context_prompts(self, target_prompt: str) -> List[str]:
        return [c for c, _ in self._get(target_prompt)]

    def get_context_weight(self, target_prompt: str,
                           context_prompt: str) -> float:
        for c, w in self._get(target_prompt):
            if label_match(c, context_prompt):
                return w
        return 0.0


# ---------- factory ----------

def make_prior(backend: str = "pmi", **kwargs):
    """Return a prior backend by name.

    backend: "hardcoded" | "pmi"
    Extra kwargs are passed to the backend constructor.
    """
    if backend == "hardcoded":
        return HardcodedPrior()
    if backend == "pmi":
        return SemanticPriorBackend(**kwargs)
    raise ValueError(f"unknown prior backend: {backend!r}")


# ---------- smoke test ----------

if __name__ == "__main__":
    import os

    print("=== HardcodedPrior ===")
    hp = make_prior("hardcoded")
    for t in ["remote control", "book", "mug", "pencil"]:
        ctxs = hp.get_context_prompts(t)
        print(f"  {t!r:24s} ->", ctxs[:6])

    print()
    print("=== SemanticPriorBackend ===")
    if os.path.exists("outputs/prior_build/semantic_prior.json"):
        pp = make_prior("pmi")
        for t in ["remote control", "book", "mug", "pencil", "tv",
                 "laptop", "house plant"]:
            ctxs = pp.get_context_prompts(t)
            weights = [pp.get_context_weight(t, c) for c in ctxs[:5]]
            print(f"  {t!r:24s} ->",
                  [(c, round(w, 3)) for c, w in zip(ctxs[:5], weights)])
    else:
        print("  (prior file missing, skipping PMI backend test)")
