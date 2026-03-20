# Is Copying More Calibrated if It's Specialized?

A research study comparing the calibration and accuracy of copy decisions between a specialized Transformer copy-decider and a joint pointer-generator model.

## Key Findings

- **Both models are well-calibrated** (ECE < 0.01) — no significant calibration difference (p=0.27)
- **Specialization dramatically improves copy decision accuracy**: F1 0.985 vs 0.969 (52% error reduction)
- **Brier score strongly favors the specialized model** (0.013 vs 0.024), driven by better accuracy, not calibration
- **The joint model's generation loss implicitly regularizes calibration**, matching the specialized model
- **At larger model sizes, the joint model's calibration advantage increases** (ablation study)

## Conclusion

Specialization helps copy *accuracy* far more than copy *calibration*. Both approaches produce well-calibrated probabilities, but the specialized model makes substantially better binary copy/generate decisions.

## Repository Structure

```
├── REPORT.md                    # Full research report with methodology and results
├── planning.md                  # Research plan and experimental design
├── src/
│   └── experiment.py            # Main experiment script (models, training, evaluation)
├── results/
│   ├── all_results.json         # Complete numerical results
│   ├── plots/
│   │   ├── reliability_diagrams.png    # Per-model reliability diagrams
│   │   ├── combined_reliability.png    # Overlay comparison
│   │   ├── metric_comparison.png       # Bar charts with error bars
│   │   └── size_ablation.png           # Model size ablation
│   └── models/                  # Saved model checkpoints
├── literature_review.md         # Background literature survey
├── resources.md                 # Catalog of datasets, papers, code
├── papers/                      # Downloaded research papers
├── datasets/                    # Pre-downloaded datasets
└── code/                        # Reference implementations
```

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch numpy matplotlib scikit-learn scipy tqdm

# Run experiments (~15 min with GPU)
CUDA_VISIBLE_DEVICES=0 python src/experiment.py
```

Requires Python 3.10+ and a CUDA-capable GPU. Results are saved to `results/`.

## Details

See [REPORT.md](REPORT.md) for the full research report including methodology, statistical analysis, ablation studies, and discussion.
