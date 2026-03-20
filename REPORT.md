# Research Report: Is Copying More Calibrated if It's Specialized?

## 1. Executive Summary

We tested whether a small Transformer trained specifically for copy decisions produces better-calibrated copy probabilities than a joint pointer-generator model that learns generation and copy decisions simultaneously. **Our results show a nuanced answer**: both models achieve remarkably low calibration error (ECE < 0.01), with no statistically significant difference in ECE (p=0.27). However, the specialized model dramatically outperforms the joint model in copy decision **accuracy** (F1: 0.985 vs 0.969) and **Brier score** (0.013 vs 0.024), meaning specialization primarily improves the sharpness and accuracy of copy predictions rather than their calibration per se. The practical implication is that specialization yields more reliable copy decisions overall, even though both approaches produce well-calibrated probability estimates.

## 2. Goal

**Hypothesis**: Training a small Transformer specifically to decide when to copy and from where will result in more accurate copying decisions than using a general language model trained for the same task.

**Why this matters**: Copy mechanisms are fundamental to NLP tasks like summarization, dialogue, and code generation. Models must decide when to copy from input vs. generate from vocabulary. If these decisions are poorly calibrated or inaccurate, downstream systems make suboptimal choices. Understanding whether task specialization improves copy decision quality has implications for NLP system design.

**Gap in existing work**: No prior work measures the calibration of copy decisions. All existing copy mechanisms (Pointer-Generator, CopyNet, Pointer Sentinel) embed the copy decision within the full generation model. The question of whether a separately trained, specialized copy-decision model would perform better has not been explored.

## 3. Data Construction

### Dataset Description
We used a **synthetic copy task** with controllable ground-truth copy labels, enabling precise measurement of calibration without the noise of real-world data.

- **Source**: Synthetically generated
- **Size**: 10,000 train / 2,000 validation / 2,000 test sequences
- **Vocabulary**: 500 tokens
- **Source length**: 40 tokens, **Target length**: 20 tokens
- **Copy ratio**: ~46.5% of target tokens are copies (empirical mean across test sets)

### Copy Decision Rule
The probability of copying at each output position depends on three factors:
1. **Token type** (tokens 0-249 are "entity-like", 70% copy tendency vs 30%)
2. **Position** (edges of input have 60% copy tendency vs 40%)
3. **Local repetition** (repeated nearby tokens increase copy tendency)

Plus Gaussian noise (σ=0.1) for realistic difficulty. This creates a learnable but non-trivial decision boundary.

### Example Samples
| Output Position | Source Token ID | Token Type | Position Region | Copy? |
|:-:|:-:|:-:|:-:|:-:|
| 3 | 142 | Entity-like | Edge | Yes (copied from src pos 6) |
| 7 | 387 | Function-like | Middle | No (generated token 201) |
| 12 | 88 | Entity-like | Middle | Yes (copied from src pos 24) |

### Data Quality
- No missing values (fully synthetic)
- Balanced copy/generate split (~46.5% copy, ~53.5% generate)
- Reproducible via random seed

### Train/Val/Test Splits
- Random generation with different seeds per split
- No overlap between splits (independent generation)
- 10K/2K/2K split (5:1:1 ratio)

## 4. Experiment Description

### Methodology

#### High-Level Approach
We train two models on the same synthetic copy task:
1. **Joint Pointer-Generator**: A small Transformer encoder-decoder with a pointer-generator mechanism. It learns to generate tokens AND make copy decisions simultaneously. The p_gen gate (following See et al., 2017) serves as the copy probability.
2. **Specialized Copy-Decider**: A small Transformer that takes encoder/decoder representations from the trained joint model and predicts ONLY copy decisions (binary copy/generate + pointer distribution).

We then compare the calibration and accuracy of their copy probability estimates.

#### Why This Method?
- **Synthetic data** gives us ground-truth copy labels, enabling precise calibration measurement
- **Shared encoder representations** isolates the effect of specialization (both models see the same features)
- **Multiple seeds** ensures results are robust
- **Temperature scaling baseline** tests whether any calibration gap is easily correctable

### Implementation Details

#### Tools and Libraries
| Library | Version |
|---------|---------|
| Python | 3.12.8 |
| PyTorch | 2.10.0+cu128 |
| NumPy | 2.4.3 |
| SciPy | 1.15.3 |
| matplotlib | 3.10.3 |

#### Hardware
- GPU: NVIDIA RTX A6000 (49GB VRAM)
- Training device: single GPU (CUDA)

#### Model Architectures

**Joint Pointer-Generator** (824K parameters):
- Embedding: 500 tokens -> d=128
- Positional encoding (sinusoidal)
- Transformer encoder: 2 layers, 4 heads, d_ff=256
- Transformer decoder: 2 layers, 4 heads, d_ff=256
- Generation head: linear(128, 500)
- p_gen gate: linear(384, 1) with sigmoid (from concatenated decoder state, context, input embedding)
- Pointer attention: query/key projections from decoder/encoder

**Specialized Copy-Decider** (472K parameters):
- Input projections for encoder/decoder representations
- 2 Transformer decoder layers (cross-attending to encoder)
- Copy decision head: MLP(128 -> 64 -> 1) with sigmoid
- Pointer attention: query/key projections

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| d_model | 128 | Literature-guided |
| n_heads | 4 | Standard for d=128 |
| n_layers | 2 | Small model design |
| d_ff | 256 | 2x d_model |
| dropout | 0.1 | Default |
| batch_size | 128 | GPU memory |
| learning_rate | 3e-4 | Adam default |
| epochs | 30 | Convergence monitoring |
| optimizer | AdamW | Standard |
| scheduler | CosineAnnealing | Smooth decay |
| grad_clip | 1.0 | Stability |

#### Training Procedure
1. Train joint model end-to-end on generation + copy + pointer losses
2. Extract encoder/decoder representations from trained joint model
3. Train specialized model on frozen representations (copy + pointer losses only)
4. Evaluate both on held-out test set

### Experimental Protocol

#### Reproducibility Information
- **Number of runs**: 5 (seeds: 42, 123, 456, 789, 1024)
- **Random seeds**: Set for Python, NumPy, PyTorch, CUDA
- **Execution time**: ~2 minutes per seed, ~15 minutes total including ablation

#### Evaluation Metrics
- **ECE (Expected Calibration Error)**: Average gap between predicted probability and observed frequency across 15 bins. Lower is better. Primary calibration metric.
- **MCE (Maximum Calibration Error)**: Worst-case bin calibration gap. Lower is better.
- **Brier Score**: Mean squared error of probability estimates. Decomposes into calibration + refinement. Lower is better.
- **Accuracy/F1**: Binary classification accuracy and F1 for copy decision at threshold 0.5.

### Raw Results

#### Main Results (5 seeds, mean ± std)

| Model | ECE | MCE | Brier | Accuracy | F1 |
|-------|:---:|:---:|:-----:|:--------:|:--:|
| Joint | 0.0065±0.0017 | 0.332±0.173 | 0.0239±0.0021 | 0.970±0.004 | 0.969±0.004 |
| Specialized | 0.0078±0.0012 | 0.214±0.068 | **0.0128±0.0012** | **0.986±0.001** | **0.985±0.001** |
| Temp-Scaled | **0.0063±0.0021** | 0.311±0.129 | 0.0238±0.0020 | 0.970±0.004 | 0.969±0.004 |

#### Model Size Ablation (single seed=42)

| Size | Joint ECE | Spec ECE | Joint F1 | Spec F1 | Joint Params | Spec Params |
|------|:---------:|:--------:|:--------:|:-------:|:------------:|:-----------:|
| Tiny (1L, d=64) | 0.0192 | 0.0179 | 0.918 | 0.968 | 157K | 69K |
| Small (2L, d=128) | 0.0106 | 0.0100 | 0.957 | 0.982 | 824K | 472K |
| Medium (4L, d=256) | 0.0053 | 0.0092 | 0.979 | 0.983 | 5.7M | 3.5M |

#### Statistical Tests

| Comparison | t-statistic | p-value | Significant (α=0.05)? |
|------------|:-----------:|:-------:|:---------------------:|
| Joint vs Specialized (ECE) | -1.27 | 0.272 | No |
| Temp-Scaled vs Specialized (ECE) | -1.01 | 0.371 | No |
| Cohen's d (Joint vs Spec ECE) | -0.64 | — | Medium effect |

#### Output Locations
- Full results: `results/all_results.json`
- Reliability diagrams: `results/plots/reliability_diagrams.png`
- Combined comparison: `results/plots/combined_reliability.png`
- Metric comparison bars: `results/plots/metric_comparison.png`
- Size ablation: `results/plots/size_ablation.png`

## 5. Result Analysis

### Key Findings

**Finding 1: Both models are well-calibrated, with no significant calibration difference.**
The joint model (ECE=0.0065) and specialized model (ECE=0.0078) both achieve excellent calibration. The difference is not statistically significant (p=0.27). The reliability diagrams show both models track the diagonal closely.

**Finding 2: The specialized model is dramatically more accurate at copy decisions.**
The specialized model achieves F1=0.985 vs 0.969 for the joint model — a 52% reduction in error rate. This is the clearest advantage of specialization.

**Finding 3: The specialized model has a substantially better Brier score.**
Brier score (0.0128 vs 0.0239) favors the specialized model by ~46%. Since the Brier score decomposes into calibration + sharpness, and calibration is similar, the advantage comes from *sharpness* — the specialized model produces more confident AND more correct predictions.

**Finding 4: Temperature scaling provides minimal benefit.**
The optimal temperature was consistently near 1.0 (range: 1.03-1.11), confirming both models are already well-calibrated. Temperature scaling marginally improves the joint model's ECE but doesn't affect accuracy.

**Finding 5: The specialized model has lower worst-case calibration error (MCE).**
MCE is 0.214 for specialized vs 0.332 for joint. The joint model has higher variance in MCE (std=0.173 vs 0.068), suggesting it occasionally produces very poorly calibrated bins, while the specialized model is more consistently calibrated.

**Finding 6: Size ablation reveals an interaction effect.**
At tiny size, specialized has slightly better ECE. At medium size, the joint model has better ECE. This suggests that as model capacity increases, the joint model's calibration benefits from the regularizing effect of the generation task, while the specialized model may overfit slightly to the copy decision.

### Hypothesis Testing Results

- **H1 (Specialized has lower ECE)**: **Not supported.** ECE difference is not significant (p=0.27). Both models are well-calibrated.
- **H2 (Specialized reliability diagram closer to diagonal)**: **Mixed.** Both are close. Specialized has lower MCE but slightly higher ECE.
- **H3 (Temperature scaling partially closes gap)**: **Not applicable** — there is no calibration gap to close.
- **H4 (Calibration advantage holds across sizes)**: **Not supported.** At larger sizes, the joint model is actually better calibrated.

### Surprises and Insights

1. **The joint model's generation loss acts as an implicit regularizer for calibration.** Training on the harder generation task may prevent the copy decision head from becoming overconfident, naturally producing well-calibrated probabilities.

2. **Specialization helps accuracy far more than calibration.** The 52% error reduction in F1 is striking. By focusing all capacity on copy decisions, the specialized model learns a much sharper decision boundary.

3. **The Brier score decomposition tells the real story.** Equal calibration + much better accuracy = much better Brier score. This means specialization improves the *quality* of copy decisions even though it doesn't improve their *calibration*.

4. **MCE vs ECE tells different stories.** The specialized model has better worst-case calibration despite slightly worse average calibration. This suggests the joint model has a few "miscalibrated pockets" in probability space.

### Limitations

1. **Synthetic data**: The copy decision rule is simpler than real-world copying patterns. Results may not transfer to natural language tasks.
2. **Shared encoder**: The specialized model uses representations from the joint model's encoder. In practice, one might train a separate encoder, which could change the dynamics.
3. **Small scale**: Models are very small (472K-824K params). Larger models common in practice (100M+) might show different calibration behavior.
4. **Single task**: Only one synthetic task structure was tested. Different copy patterns (more/less structured, different copy ratios) might yield different results.
5. **No real-world validation**: We did not validate on CNN/Daily Mail or other real summarization datasets due to the computational cost of training full pointer-generators from scratch.

## 6. Conclusions

### Summary
Training a specialized small Transformer for copy decisions does **not** produce better-calibrated probability estimates than a joint pointer-generator model — both achieve excellent calibration (ECE < 0.01). However, specialization produces **dramatically more accurate** copy decisions (F1: 0.985 vs 0.969, 52% error reduction) and better overall probabilistic predictions (Brier: 0.013 vs 0.024). The benefit of specialization lies in decision accuracy and sharpness, not calibration per se.

### Implications
- **For practitioners**: If you need a reliable copy decision, train a specialized model. You get much better accuracy with naturally good calibration.
- **For the calibration literature**: Joint training on a harder task (generation) appears to implicitly regularize calibration, matching or exceeding a specialized model.
- **For architecture design**: The copy decision module should be separate from the generator for accuracy, but does not need special calibration treatment.

### Confidence in Findings
**Moderate-to-high** for the calibration finding (no significant difference), supported by 5 seeds and consistent results. **High** for the accuracy finding (large, consistent F1 gap across all seeds). The synthetic setting provides clean measurements but limits generalizability.

## 7. Next Steps

### Immediate Follow-ups
1. **Real-world validation**: Train pointer-generator and specialized copy-decider on CNN/Daily Mail to verify findings transfer to natural language.
2. **Independent encoder**: Train the specialized model with its own encoder (not shared) to test whether end-to-end specialization changes calibration dynamics.
3. **Harder copy tasks**: Create synthetic tasks with more complex, context-dependent copy rules to test calibration under greater difficulty.

### Alternative Approaches
- Test with modern pre-trained models (BART, T5) as the base, extracting copy decisions from cross-attention patterns
- Compare against CopyNet's shared softmax approach
- Use focal loss for the specialized model to further improve calibration

### Open Questions
1. Does the implicit calibration regularization from generation training hold at larger scale?
2. Would a specialized model trained end-to-end (with its own encoder) show better or worse calibration?
3. How does the copy ratio affect the calibration dynamics? (Very low or very high copy ratios may change the picture.)
4. Can the specialized model's accuracy advantage be combined with the joint model's calibration advantage?

## References

1. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. *NeurIPS*.
2. Gu, J., Lu, Z., Li, H., & Li, V. O. (2016). Incorporating Copying Mechanism in Sequence-to-Sequence Learning. *ACL*.
3. See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. *ACL*.
4. Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer Sentinel Mixture Models. *ICLR*.
5. Gulcehre, C., Ahn, S., Nallapati, R., Zhou, B., & Bengio, Y. (2016). Pointing the Unknown Words. *ACL*.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.
7. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
