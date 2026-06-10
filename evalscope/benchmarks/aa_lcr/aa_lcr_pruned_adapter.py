from typing import List

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.pruning.variance_pruner import prune_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter, PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description='Variance-pruned version of AA-LCR. Keeps samples where models disagree most.',
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files.',
                'value': None
            },
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy. Currently supports: variance',
                'value': 'variance'
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (0.0-1.0)',
                'value': 0.3
            },
            'reviews_path': {
                'type': 'str',
                'description': 'Path to folder containing .jsonl review files for pruning',
                'value': 'Evals/Part 1/reviews'
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """
    Pruned AA-LCR adapter.
    Inherits all evaluation logic from AALCRAdapter,
    only overrides sample_filter to apply variance-based pruning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
        self.reviews_path = self.extra_params.get('reviews_path', 'Evals/Part 1/reviews')
        self._kept_indices = None

    def _get_kept_indices(self) -> List[int]:
        if self._kept_indices is None:
            self._kept_indices = prune_benchmark(
                reviews_path=self.reviews_path,
                prune_ratio=self.prune_ratio,
                score_key='acc',
                file_prefix='aa_lcr',
            )
        return self._kept_indices

    def sample_filter(self, sample) -> bool:
        kept = self._get_kept_indices()
        return sample.index in kept