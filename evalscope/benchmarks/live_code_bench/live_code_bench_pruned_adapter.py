from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.pruning.variance_pruner import prune_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter

_PRUNED_INDICES: List[int] = []


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description='Variance-pruned version of LiveCodeBench. Keeps samples where models disagree most.',
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['release_v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template='### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy to use. Currently supports: variance',
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
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """
    Pruned LiveCodeBench adapter.
    Inherits all evaluation logic from LiveCodeBenchAdapter,
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
                score_key='pass',
                file_prefix='live_code_bench',
            )
        return self._kept_indices

    def sample_filter(self, sample) -> bool:
        # first apply parent date filter
        if not super().sample_filter(sample):
            return False
        # then apply pruning filter
        kept = self._get_kept_indices()
        return sample.index in kept