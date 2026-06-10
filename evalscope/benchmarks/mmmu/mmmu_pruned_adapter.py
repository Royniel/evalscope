"""
MMMU Pruned Adapter - Image encoder stress probe.

Strategy: select samples where the image is LOAD-BEARING for the answer.
We stress the image encoder by prioritizing:
1. Subjects where visual content is essential (Pathology, Engineering diagrams, etc.)
2. Image types that require fine-grained visual processing (Charts, Diagrams, Tables)
3. Samples where the model got it WRONG (encoder likely struggled)

This is designed to surface image encoder degradation specifically,
not general capability gaps.
"""

from typing import List, Dict, Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter, SUBSET_LIST, MULT_CHOICE_PROMPT, OPEN_PROMPT

# Subjects where image is MOST load-bearing (text alone cannot answer)
HIGH_VISUAL_SUBJECTS = {
    'Diagnostics_and_Laboratory_Medicine',  # microscopy, lab images
    'Pathology',                            # tissue slides
    'Electronics',                          # circuit diagrams
    'Architecture_and_Engineering',         # technical drawings
    'Materials',                            # microscopy, crystal structures
    'Art',                                  # artwork analysis
    'Design',                               # visual design
    'Geography',                            # maps, satellite images
    'Math',                                 # geometry diagrams
    'Mechanical_Engineering',               # engineering drawings
    'Physics',                              # experimental setups
    'Chemistry',                            # molecular diagrams
}

# Image types that stress the encoder most
HIGH_STRESS_IMG_TYPES = {
    'Figures',
    'Diagrams',
    'Charts',
    'Tables',
    'Medical Images',
    'Circuit Diagrams',
    'Maps',
    'Chemical Structures',
}


def compute_visual_stress_score(metadata: Dict[str, Any], acc: float) -> float:
    """
    Score each sample by how much it stresses the image encoder.

    Higher score = better probe sample.
    """
    score = 0.0

    # Boost for high-visual subjects
    subfield = metadata.get('subfield', '')
    subject = metadata.get('id', '').split('_')[1] if metadata.get('id') else ''
    for vs in HIGH_VISUAL_SUBJECTS:
        if vs.lower() in subfield.lower() or vs.lower() in subject.lower():
            score += 2.0
            break

    # Boost for image types that stress encoder
    img_types = metadata.get('img_type', '[]')
    if isinstance(img_types, str):
        img_types = img_types.strip("[]'\"").split(',')
    for img_type in img_types:
        img_type = img_type.strip().strip("'\"")
        for stress_type in HIGH_STRESS_IMG_TYPES:
            if stress_type.lower() in img_type.lower():
                score += 1.5
                break

    # Boost for medium/hard difficulty
    difficulty = metadata.get('topic_difficulty', 'Medium')
    if difficulty == 'Hard':
        score += 1.0
    elif difficulty == 'Medium':
        score += 0.5

    # Boost for samples the model got WRONG
    # Wrong answers suggest the encoder struggled
    if acc == 0.0:
        score += 1.0

    return score


def load_mmmu_reviews(reviews_path: str) -> Dict[str, Dict]:
    """Load MMMU review files, keyed by sample_id."""
    import json
    from pathlib import Path

    reviews_dir = Path(reviews_path)
    sample_scores = {}

    for review_file in sorted(reviews_dir.glob('*.jsonl')):
        with open(review_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample_id = record.get('sample_id') or record.get('index')
                acc = record['sample_score']['score']['value'].get('acc', 0.0)
                metadata = record['sample_score'].get('sample_metadata', {})
                sample_scores[sample_id] = {
                    'acc': float(acc),
                    'metadata': metadata,
                }

    return sample_scores


def select_mmmu_probe_indices(
    reviews_path: str,
    prune_ratio: float = 0.3,
) -> List[int]:
    """
    Select MMMU sample indices that best stress the image encoder.
    Returns list of sample_ids to keep.
    """
    sample_scores = load_mmmu_reviews(reviews_path)

    scored = []
    for sample_id, data in sample_scores.items():
        stress_score = compute_visual_stress_score(data['metadata'], data['acc'])
        scored.append((sample_id, stress_score))

    # Sort by stress score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    n_keep = max(1, int(len(scored) * prune_ratio))
    kept = [s[0] for s in scored[:n_keep]]

    print(f'MMMU probe: keeping {len(kept)}/{len(scored)} samples ({len(kept)/len(scored):.1%})')
    return kept


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Image Encoder Probe)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description='Image-encoder stress probe for MMMU. Selects samples where visual processing is most critical.',
        dataset_id='MMMU/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=MULT_CHOICE_PROMPT,
        extra_params={
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (0.0-1.0)',
                'value': 0.3
            },
            'reviews_path': {
                'type': 'str',
                'description': 'Path to MMMU review files for scoring',
                'value': 'Evals/MMMU/reviews/glm-4.5v-fp8'
            },
        },
    )
)
class MMMUPrunedAdapter(MMMUAdapter):
    """
    MMMU adapter that selects image-encoder-stressing samples.
    Inherits all evaluation logic from MMMUAdapter.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
        self.reviews_path = self.extra_params.get('reviews_path', 'Evals/MMMU/reviews/glm-4.5v-fp8')
        self._kept_indices = None

    def _get_kept_indices(self) -> List:
        if self._kept_indices is None:
            self._kept_indices = select_mmmu_probe_indices(
                reviews_path=self.reviews_path,
                prune_ratio=self.prune_ratio,
            )
        return self._kept_indices

    def sample_filter(self, sample) -> bool:
        kept = self._get_kept_indices()
        return sample.index in kept or getattr(sample, 'sample_id', None) in kept