"""Hyperparameter search utilities for CandidateFinding methods."""
import argparse
import dataclasses
import itertools
import json
import logging
import pathlib
import random
import sys
import time
from typing import Dict, Iterable, List, Tuple

try:
    from eve_smlm.GUI_main import FindingAnalysis
    from eve_smlm.Utils import utils, utilsHelper
    from eve_smlm.CandidateFinding import *  # noqa: F401,F403
except ImportError:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from GUI_main import FindingAnalysis
    from Utils import utils, utilsHelper
    from CandidateFinding import *  # type: ignore # noqa: F401,F403

RATIO_RANGE = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
DBSCAN_EPS_RANGE = [4, 5, 6, 7, 8, 9, 10]

DBSCAN_GRID = {
    "DBSCAN_onlyHighDensity": {
        "distance_radius_lookup": [4, 6, 8, 10, 12],
        "density_multiplier": [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
        "min_cluster_size": [10, 15, 20, 25, 30, 35, 40],
        "ratio_ms_to_px": RATIO_RANGE,
        "DBSCAN_eps": DBSCAN_EPS_RANGE,
        "min_consec": [1, 2, 3, 4, 5],
    },
    "DBSCAN_allEvents": {
        "distance_radius_lookup": [4, 6, 8, 10, 12],
        "density_multiplier": [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
        "min_cluster_size": [10, 15, 20, 25, 30, 35, 40],
        "ratio_ms_to_px": RATIO_RANGE,
        "DBSCAN_eps": DBSCAN_EPS_RANGE,
        "padding_xy": [0, 1, 2, 3, 4],
        "min_consec": [1, 2, 3, 4, 5],
    },
    "DBSCAN_allEvents_remove_outliers": {
        "distance_radius_lookup": [4, 6, 8, 10, 12],
        "density_multiplier": [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
        "min_cluster_size": [10, 15, 20, 25, 30, 35, 40],
        "ratio_ms_to_px": RATIO_RANGE,
        "DBSCAN_eps": DBSCAN_EPS_RANGE,
        "padding_xy": [0, 1, 2, 3, 4],
        "outlier_removal_radius": [2, 3, 4, 5, 6],
        "outlier_removal_nbPoints": [20, 30, 40, 50, 60],
        "min_consec": [1, 2, 3, 4, 5],
    },
}

EIGEN_GRID = {
    "eigenFeature_analysis": {
        "search_n_neighbours": [30, 45, 60, 75, 90, 105, 120],
        "max_eigenval_cutoff": [0.0, 3.0, 5.0, 7.0, 9.0],
        "linearity_cutoff": [0.5, 0.6, 0.7, 0.8, 0.85],
        "ratio_ms_to_px": [15.0, 20.0, 25.0, 30.0, 35.0],
        "DBSCAN_eps": [2, 3, 4, 5, 6],
        "DBSCAN_n_neighbours": [15, 20, 25, 30, 35],
    },
    "eigenFeature_analysis_and_bbox_finding": {
        "search_n_neighbours": [30, 45, 60, 75, 90, 105, 120],
        "max_eigenval_cutoff": [0.0, 3.0, 5.0, 7.0, 9.0],
        "linearity_cutoff": [0.5, 0.6, 0.7, 0.8, 0.85],
        "ratio_ms_to_px": [15.0, 20.0, 25.0, 30.0, 35.0],
        "DBSCAN_eps": [2, 3, 4, 5, 6],
        "DBSCAN_n_neighbours": [15, 20, 25, 30, 35],
        "bbox_padding": [0, 1, 2, 3],
    },
}

FRAME_GRID = {
    "FrameBased_finding": {
        "threshold_detection": [2.5, 3.0, 3.5, 4.0, 4.5],
        "exclusion_radius": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "min_diameter": [1.0, 1.25, 1.5, 1.75, 2.0],
        "max_diameter": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "frame_time": [50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0],
        "candidate_radius": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
}

GRID_REGISTRY = {**DBSCAN_GRID, **EIGEN_GRID, **FRAME_GRID}


@dataclasses.dataclass
class RankingResult:
    method: str
    params: Dict[str, float]
    metrics: Dict[str, float]
    score: float


class RankingMetric:
    CANDIDATE_COUNT = "candidate_count"
    CANDIDATES_PER_MS = "candidates_per_ms"
    MEDIAN_CLUSTER_SIZE = "median_cluster_size"
    RUNTIME_INVERSE = "runtime_inverse"

    @classmethod
    def all(cls) -> List[str]:
        return [
            cls.CANDIDATE_COUNT,
            cls.CANDIDATES_PER_MS,
            cls.MEDIAN_CLUSTER_SIZE,
            cls.RUNTIME_INVERSE,
        ]


def compute_metric(metric: str, stats: Dict[str, float]) -> float:
    if metric == RankingMetric.CANDIDATE_COUNT:
        return stats.get("candidate_count", 0.0)
    if metric == RankingMetric.CANDIDATES_PER_MS:
        duration = stats.get("preview_duration_ms", 1.0)
        return stats.get("candidate_count", 0.0) / max(duration, 1.0)
    if metric == RankingMetric.MEDIAN_CLUSTER_SIZE:
        return stats.get("median_cluster_size", 0.0)
    if metric == RankingMetric.RUNTIME_INVERSE:
        runtime = stats.get("runtime_seconds", 1.0)
        return 1.0 / max(runtime, 1e-3)
    raise ValueError(f"Unknown metric {metric}")


def generate_param_grid(method: str) -> Iterable[Dict[str, float]]:
    grid = GRID_REGISTRY.get(method)
    if not grid:
        raise ValueError(f"No grid defined for {method}")
    keys = sorted(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def evaluate_method(
    finding: FindingAnalysis,
    method_name: str,
    params: Dict[str, float],
    metric: str,
) -> RankingResult:
    start = time.time()
    eval_text = utils.getEvalTextFromGUIFunction(
        method_name,
        list(params.keys()),
        [str(v) for v in params.values()],
        "self.events",
    )
    finding.set_EvalText(eval_text)
    finding.runFinding()
    results = finding.get_Results()
    candidate_count = len(results.get(0, {})) if isinstance(results, dict) else 0
    stats = {
        "candidate_count": candidate_count,
        "runtime_seconds": time.time() - start,
        "preview_duration_ms": finding.timeStretchMs[1] - finding.timeStretchMs[0],
        "median_cluster_size": utilsHelper.medianClusterSize(results.get(0, {})),
    }
    score = compute_metric(metric, stats)
    return RankingResult(method=method_name, params=params, metrics=stats, score=score)


def run_search(args: argparse.Namespace) -> List[RankingResult]:
    finding = FindingAnalysis()
    finding.set_fileLocation(args.file)
    finding.set_timeStretchMs([args.preview_start_ms, args.preview_end_ms])
    finding.set_xyStretch([[args.min_x, args.max_x], [args.min_y, args.max_y]])
    finding.set_polarityAnalysis(args.polarity)
    finding.set_chunkingTime([args.chunk_ms, args.chunk_overlap_ms])
    ranking_results: List[RankingResult] = []
    for method in args.methods:
        for paramset in generate_param_grid(method):
            try:
                result = evaluate_method(finding, method, paramset, args.ranking_metric)
            except Exception as exc:
                logging.error("Method %s failed for params %s: %s", method, paramset, exc)
                continue
            ranking_results.append(result)
    ranking_results.sort(key=lambda r: r.score, reverse=True)
    return ranking_results


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finding hyperparameter search")
    parser.add_argument("file", help="Event data file path")
    parser.add_argument("--methods", nargs="+", default=list(GRID_REGISTRY.keys()))
    parser.add_argument("--preview-start-ms", type=float, default=0.0)
    parser.add_argument("--preview-end-ms", type=float, default=1000.0)
    parser.add_argument("--min-x", type=float, default=0.0)
    parser.add_argument("--max-x", type=float, default=512.0)
    parser.add_argument("--min-y", type=float, default=0.0)
    parser.add_argument("--max-y", type=float, default=512.0)
    parser.add_argument("--polarity", choices=["Pos", "Neg", "Mix", "Both"], default="Mix")
    parser.add_argument("--chunk-ms", type=float, default=1000.0)
    parser.add_argument("--chunk-overlap-ms", type=float, default=0.0)
    parser.add_argument(
        "--ranking-metric",
        choices=RankingMetric.all(),
        default=RankingMetric.CANDIDATE_COUNT,
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0, help="Random subset of param combos")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    results = run_search(args)
    top_results = results[: args.max_results]
    if args.output:
        payload = [dataclasses.asdict(r) for r in top_results]
        args.output.write_text(json.dumps(payload, indent=2))
    for r in top_results:
        logging.info("%s score=%.3f params=%s", r.method, r.score, r.params)


if __name__ == "__main__":
    main(sys.argv[1:])
