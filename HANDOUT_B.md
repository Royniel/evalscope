# Handout B — Why This Matters and How to Use It

## What Changes for the Customer Conversation

Before these pruners, answering "is this model good enough for our workload?" meant running hundreds of expensive benchmark samples. Now:

- **LCB**: Run 94 samples instead of 315 — same ranking signal, 70% cheaper
- **AA-LCR**: Run 30 samples instead of 100 — same ranking signal, 70% cheaper
- **MMMU probe**: Run ~200 targeted samples instead of 12,000 — surfaces encoder problems fast

For a customer evaluating 3-4 candidate models, this cuts evaluation cost by 70% while preserving the go/no-go answer they need.

## How to Run It Tomorrow

A sales engineer or deployment lead can run this inside evalscope with two commands:

```bash
# Coding capability
evalscope eval --model <candidate_model> --datasets live_code_bench_pruned \
    --dataset-args '{"prune_ratio": 0.3, "reviews_path": "Evals/Part 1/reviews"}'

# Long-context reasoning
evalscope eval --model <candidate_model> --datasets aa_lcr_pruned \
    --dataset-args '{"prune_ratio": 0.3, "reviews_path": "Evals/Part 1/reviews"}'
```

The output is a single accuracy score. Compare it to the known scores of gpt-oss-120b, kimi-k2.5, and minimax-m2.5 to immediately position the new model.

## What the Multimodal Probe Gives You That Random Sampling Cannot

Random sampling picks easy visual questions alongside hard ones. A model with a degraded image encoder still scores well on text-deducible questions — random sampling misses the problem.

Our probe specifically picks questions where **the image is the answer**. A degraded encoder will fail these disproportionately, surfacing the problem with far fewer samples.

Think of it like this: if you want to test someone's eyesight, you give them an eye chart — not a general knowledge quiz with some pictures.

## Why a Customer-Facing PM Should Care

- **Faster decisions**: 70% cheaper evaluation means faster turnaround on customer questions
- **Defensible answers**: "Our model scores X on this benchmark" is now a statement you can make in hours, not days
- **Targeted multimodal signal**: If a customer is evaluating multimodal next quarter, you can give them a quick encoder health check before committing to a full evaluation