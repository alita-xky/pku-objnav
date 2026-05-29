import json
import re
from typing import List, Tuple, Dict


def norm_label(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")


def split_camel_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


class SemanticPrior:
    def __init__(self, prior_path: str):
        with open(prior_path, "r") as f:
            data = json.load(f)

        self.data = data
        self.objects = data["objects"]
        self.prior = data["prior"]

        self.alias = {
            "remote": "remote control",
            "tv remote": "remote control",
            "television remote": "remote control",
            "controller": "remote control",
            "television": "tv",
            "monitor": "tv",
            "screen": "tv",
            "couch": "sofa",
            "plant": "house plant",
            "trash can": "garbage can",
            "bin": "garbage can",
            "cellphone": "cell phone",
            "phone": "cell phone",
        }

    def resolve_query(self, query: str) -> List[Tuple[str, float]]:
        q = query.lower().strip()

        if q in self.alias:
            return [(self.alias[q], 1.0)]

        qn = norm_label(q)

        exact = []
        for obj in self.objects:
            if norm_label(obj) == qn:
                exact.append((obj, 1.0))

        if exact:
            return exact

        partial = []
        for obj in self.objects:
            on = norm_label(obj)

            if qn in on or on in qn:
                score = min(len(qn), len(on)) / max(len(qn), len(on))
                partial.append((obj, score))

        partial = sorted(partial, key=lambda x: x[1], reverse=True)

        if partial:
            return partial[:5]

        return []

    def get_contexts_for_query(
        self,
        query: str,
        top_k_categories: int = 3,
        top_k_contexts: int = 10,
        min_score: float = 0.05,
    ) -> List[Tuple[str, float]]:
        matched_categories = self.resolve_query(query)

        if len(matched_categories) == 0:
            print(f"[SemanticPrior] Cannot resolve query: {query}")
            return []

        matched_categories = matched_categories[:top_k_categories]

        context_scores: Dict[str, float] = {}

        for category, category_weight in matched_categories:
            if category not in self.prior:
                continue

            for item in self.prior[category]:
                ctx = item["context"]
                score = item["score"] * category_weight

                if score < min_score:
                    continue

                context_scores[ctx] = max(context_scores.get(ctx, 0.0), score)

        contexts = sorted(
            context_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return contexts[:top_k_contexts]

    def get_navigation_contexts_for_query(
        self,
        query: str,
        top_k_categories: int = 3,
        top_k_contexts: int = 8,
        min_score: float = 0.05,
    ) -> List[Tuple[str, float]]:
        raw_contexts = self.get_contexts_for_query(
            query=query,
            top_k_categories=top_k_categories,
            top_k_contexts=40,
            min_score=min_score,
        )

        allowed_contexts = {
            "sofa",
            "arm chair",
            "chair",
            "tv",
            "television",
            "tv stand",
            "coffee table",
            "dining table",
            "side table",
            "desk",
            "shelf",
            "drawer",
            "cabinet",
            "counter top",
            "bed",
            "pillow",
            "laptop",
            "book",
            "floor lamp",
            "desk lamp",
            "house plant",
            "painting",
            "window",
            "box",
            "newspaper",
            "cup",
            "plate",
            "bowl",
            "coffee machine",
        }

        allowed_norms = {norm_label(x) for x in allowed_contexts}

        filtered = []

        for ctx, score in raw_contexts:
            if norm_label(ctx) in allowed_norms:
                filtered.append((ctx, score))

        if len(filtered) == 0:
            filtered = raw_contexts[:top_k_contexts]

        return filtered[:top_k_contexts]

    def print_query_result(self, query: str, top_k_contexts: int = 10):
        print("\n========== Semantic Prior Query ==========")
        print("Query:", query)

        matched = self.resolve_query(query)
        print("Matched categories:", matched)

        contexts = self.get_contexts_for_query(
            query=query,
            top_k_contexts=top_k_contexts,
        )

        print("Raw top contexts:")
        for ctx, score in contexts:
            print(f"  {ctx:20s} {score:.3f}")

        nav_contexts = self.get_navigation_contexts_for_query(
            query=query,
            top_k_contexts=top_k_contexts,
        )

        print("Navigation contexts:")
        for ctx, score in nav_contexts:
            print(f"  {ctx:20s} {score:.3f}")

        return nav_contexts


def main():
    prior = SemanticPrior("outputs/prior_build/semantic_prior.json")

    test_queries = [
        "remote",
        "tv remote",
        "book",
        "pencil",
        "couch",
        "television",
        "coffee table",
    ]

    for q in test_queries:
        prior.print_query_result(q, top_k_contexts=10)


if __name__ == "__main__":
    main()
