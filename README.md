# evalscope (Cerebras Fork)

**Commit SHA developed against:** `e1b4d09aa4a5bbdab7fa5eeaf567d74c8469a6e7`

## What's added

Variance-based benchmark pruning for LCB, AA-LCR, and MMMU image encoder probe.

## Setup

```bash
git clone https://github.com/Royniel/evalscope.git
cd evalscope
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running the pruners

**LCB pruned:**
```bash
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "variance", "prune_ratio": 0.3, "reviews_path": "Evals/Part 1/reviews"}'
```

**AA-LCR pruned:**
```bash
evalscope eval --model <model> --datasets aa_lcr_pruned \
    --dataset-args '{"pruning_strategy": "variance", "prune_ratio": 0.3, "reviews_path": "Evals/Part 1/reviews"}'
```

**MMMU probe:**
```bash
evalscope eval --model <model> --datasets mmmu_pruned \
    --dataset-args '{"prune_ratio": 0.3, "reviews_path": "Evals/MMMU/reviews/glm-4.5v-fp8"}'
```