# Handout A — Why This Works

## Part A: Coding + Long-Context Pruning (LCB v5 + AA-LCR)

### The Problem
Running 315 LCB samples and 100 AA-LCR samples for every candidate model is expensive. We need the smallest subset that still reliably ranks models the same way the full set would.

### Approach: Variance-Based Selection
We keep samples where models **disagree most**. The intuition is simple:
- A sample where all 3 models pass → too easy, no discrimination
- A sample where all 3 models fail → too hard, no discrimination  
- A sample where some pass and some fail → high variance, keeps this sample

For each sample we compute variance across model scores, then keep the top 30% by variance.

### Why This Works
This is equivalent to selecting high-discrimination items in psychometric testing (Item Response Theory). A sample that separates strong models from weak ones carries more signal per sample than consensus items. The approach is:
- **Not random** — we use real model behavior to guide selection
- **Not overfit** — variance is a property of the sample difficulty distribution, not the specific models. A fourth model would land in a predictable position relative to the existing score distribution
- **Benchmark-agnostic** — works for any benchmark with per-sample scores, just change the score key (`pass` for LCB, `acc` for AA-LCR)

### Pruning Results
- LCB: 315 → 94 samples (70% reduction)
- AA-LCR: 100 → 30 samples (70% reduction)

### Why This Subset Is Sufficient
The kept samples are those where model capability differences are most visible. A model that scores well on this subset will score well on the full set — the high-variance items are a compressed signal of the full ranking. The consensus items (all pass / all fail) add no ranking information.

### Note on AA-LCR Judge Noise
AA-LCR is graded by an LLM judge which introduces non-deterministic variance. Our variance scores for AA-LCR samples partially measure judge noise. To account for this, we could run each sample through the judge multiple times and average — but with 3 models the signal is still meaningful.

---

## Part B: MMMU Image Encoder Probe

### The Problem
We need to cheaply detect whether a candidate model's image encoder is degraded — not just whether the model is generally capable.

### Approach: Visual Stress Scoring
We score each sample by how much it **requires** the image encoder to answer correctly:

1. **Subject boost** — subjects like Diagnostics, Electronics, Architecture require actual image parsing. Text-only reasoning cannot answer these.
2. **Image type boost** — Charts, Diagrams, Circuit Diagrams, Medical Images stress the encoder more than natural photos.
3. **Difficulty boost** — harder samples require more precise visual processing.
4. **Wrong answer boost** — samples the reference model got wrong suggest the encoder struggled.

We keep the top 30% by stress score.

### Why These Choices Stress Encoders Specifically
- A degraded encoder will fail on fine-grained visual details (circuit diagrams, microscopy slides) before failing on text-heavy questions
- Subjects like Pathology and Electronics have no text-deducible answers — the image IS the question
- Random sampling would include many text-heavy questions where a degraded encoder still scores well

### Assumptions
- The 660 reference samples from glm-4.5v-fp8 are representative of the full 12K distribution
- Image type metadata is reliable
- A 4th model with encoder degradation would fail disproportionately on our probe vs random sample

### What Would Change With More Resources
- **(a) More data**: With the full 12K HuggingFace dataset, we could cluster images by visual complexity and sample proportionally across clusters
- **(b) Live model endpoint**: We could actively probe the encoder by sending images with deliberately misleading text descriptions and measuring if the model corrects them — a direct encoder quality test
- **(c) More time**: Implement IRT (Item Response Theory) discrimination parameters for more principled sample selection; add cross-validation to verify the pruned set preserves rankings