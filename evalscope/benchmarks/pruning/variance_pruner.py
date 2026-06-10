"""
Variance-based benchmark pruner.

Strategy: keep samples where models disagree the most.
- All pass or all fail = no discrimination = discard
- Mixed results = high variance = keep

This is benchmark-agnostic and works for any score field.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


def load_reviews(
    reviews_path: str,
    score_key: str = None,
    file_prefix: str = None,
) -> Dict[int, List[float]]:
    """
    Load review files and return {sample_index: [scores across models]}.

    Args:
        reviews_path: path to folder containing .jsonl review files
        score_key: score field name ('pass' or 'acc'). If None, auto-detects.
        file_prefix: only load files starting with this prefix (e.g. 'aa_lcr', 'live_code_bench')
    """
    reviews_dir = Path(reviews_path)
    model_scores: Dict[int, List[float]] = {}

    pattern = f'{file_prefix}*.jsonl' if file_prefix else '*.jsonl'

    for review_file in sorted(reviews_dir.glob(pattern)):
        with open(review_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                idx = record['index']
                score_value = record['sample_score']['score']['value']

                # auto-detect score key if not provided
                if score_key is None:
                    score_key = list(score_value.keys())[0]

                score = float(score_value.get(score_key, 0.0))

                if idx not in model_scores:
                    model_scores[idx] = []
                model_scores[idx].append(score)

    return model_scores


def compute_variance(scores: List[float]) -> float:
    """Compute variance of a list of scores."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


def select_pruned_indices(
    model_scores: Dict[int, List[float]],
    prune_ratio: float = 0.3,
) -> List[int]:
    """
    Select indices to KEEP based on score variance.

    prune_ratio: fraction of samples to keep (0.3 = keep 30%)
    """
    variances = {
        idx: compute_variance(scores)
        for idx, scores in model_scores.items()
    }

    sorted_indices = sorted(variances.keys(), key=lambda i: variances[i], reverse=True)

    n_keep = max(1, int(len(sorted_indices) * prune_ratio))
    kept = sorted_indices[:n_keep]

    return sorted(kept)


def prune_benchmark(
    reviews_path: str,
    prune_ratio: float = 0.3,
    score_key: str = None,
    file_prefix: str = None,
) -> List[int]:
    """
    Main entry point. Returns list of sample indices to keep.

    Args:
        reviews_path: path to folder with .jsonl review files
        prune_ratio: fraction of dataset to keep
        score_key: 'pass' for LCB, 'acc' for AA-LCR. Auto-detected if None.
        file_prefix: filter review files by prefix (e.g. 'aa_lcr', 'live_code_bench')
    """
    model_scores = load_reviews(reviews_path, score_key, file_prefix)
    kept_indices = select_pruned_indices(model_scores, prune_ratio)

    total = len(model_scores)
    kept = len(kept_indices)
    print(f'Pruning: keeping {kept}/{total} samples ({kept/total:.1%}) with prune_ratio={prune_ratio}')

    return kept_indices