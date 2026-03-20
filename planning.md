# Research Plan: Is Copying More Calibrated if It's Specialized?

## Motivation & Novelty Assessment

### Why This Research Matters
Copy mechanisms are fundamental to modern NLP (summarization, dialogue, code generation). Models must decide *when* to copy from input vs. generate from vocabulary. If these copy probability estimates are poorly calibrated, downstream systems make suboptimal decisions. Understanding whether specialization improves calibration has practical implications for building more reliable NLP systems.

### Gap in Existing Work
Per the literature review, **no existing work measures calibration of copy decisions**. Papers report ROUGE, perplexity, and accuracy, but never evaluate whether the copy/generate probability (p_gen) is well-calibrated. Furthermore, all existing copy mechanisms are embedded within larger models sharing representations — no work has explored a separately trained, specialized copy-decision model.

### Our Novel Contribution
We directly test whether a small Transformer trained exclusively for copy decisions produces better-calibrated copy probabilities than an equivalent joint model. We introduce calibration metrics (ECE, Brier score, reliability diagrams) to the copy mechanism literature for the first time.

### Experiment Justification
- **Experiment 1 (Synthetic Copy Task)**: Controlled environment with known ground-truth copy decisions. Allows precise calibration measurement without confounds from noisy real-world labels. This is the most rigorous test of the hypothesis.
- **Experiment 2 (Model Size Ablation)**: Tests whether calibration advantage scales with model size, isolating the effect of specialization from capacity.
- **Experiment 3 (Temperature Scaling Baseline)**: Tests whether simple post-hoc calibration of the joint model eliminates any advantage, establishing whether the benefit is fundamental or easily correctable.

## Research Question
Does a small Transformer trained specifically to decide when to copy and from where produce more calibrated (well-calibrated probability estimates) copy decisions than a general Transformer trained jointly for generation and copying?

## Hypothesis Decomposition
1. **H1**: The specialized copy-decision model has lower ECE than the joint pointer-generator model.
2. **H2**: The specialized model's reliability diagram is closer to the diagonal (perfect calibration).
3. **H3**: Post-hoc temperature scaling of the joint model partially but not fully closes the calibration gap.
4. **H4**: The calibration advantage holds across different model sizes.

## Proposed Methodology

### Approach
We create a synthetic sequence-to-sequence task with known ground-truth copy labels. We train two architectures on this task:
- **Joint Model**: Small Transformer with pointer-generator mechanism (learns generation + copy decisions simultaneously)
- **Specialized Model**: Small Transformer that takes encoder representations and predicts only copy decisions (binary copy/generate + pointer distribution)

Both models share the same encoder. The joint model is trained end-to-end. The specialized model is trained only on the copy decision loss, using encoder representations from the joint model.

### Experimental Steps
1. Generate synthetic copy dataset (10K train, 2K val, 2K test sequences)
2. Train joint pointer-generator model (small Transformer, 2 layers, d=128)
3. Extract encoder representations from trained joint model
4. Train specialized copy-decision model on same representations
5. Evaluate calibration of both models' copy probabilities
6. Apply temperature scaling to joint model as baseline
7. Ablate model sizes (1, 2, 4 layers)

### Baselines
1. **Joint Pointer-Generator**: Standard approach (See et al., 2017 adapted to Transformer)
2. **Temperature-Scaled Joint Model**: Post-hoc calibration of the joint model's p_gen
3. **Random Baseline**: Uniform random copy probabilities (calibration floor)

### Evaluation Metrics
- **ECE (Expected Calibration Error)**: Primary metric. Measures average gap between predicted probability and observed frequency.
- **Brier Score**: Proper scoring rule combining calibration and refinement.
- **Copy Decision F1**: Accuracy of binary copy/generate decision.
- **Reliability Diagrams**: Visual assessment of calibration.
- **MCE (Maximum Calibration Error)**: Worst-case calibration.

### Statistical Analysis Plan
- Run each experiment with 5 random seeds
- Report mean ± std for all metrics
- Paired t-test comparing ECE of specialized vs. joint model (α = 0.05)
- Bootstrap confidence intervals for ECE differences

## Expected Outcomes
- **Supporting hypothesis**: Specialized model has ECE < 0.05, joint model has ECE > 0.10. Temperature scaling helps but doesn't fully close the gap.
- **Refuting hypothesis**: Both models have similar ECE, or temperature scaling eliminates any difference.

## Timeline and Milestones
1. Data generation + environment setup: 15 min
2. Joint model implementation + training: 30 min
3. Specialized model implementation + training: 30 min
4. Evaluation + calibration analysis: 20 min
5. Ablation experiments: 30 min
6. Documentation + report: 30 min

## Potential Challenges
- Synthetic task may not capture real-world complexity → mitigate by designing realistic copy patterns
- Models may converge poorly → use learning rate warmup and careful hyperparameter selection
- Small model capacity may limit both models equally → test multiple sizes

## Success Criteria
1. Clear, statistically significant difference in ECE between models (p < 0.05)
2. Reliability diagrams show visible calibration difference
3. Results are reproducible across random seeds
