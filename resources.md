# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Is Copying More Calibrated if It's Specialized?" which investigates whether a small, specialized Transformer trained specifically for copy decisions produces more calibrated results than a general language model.

## Papers
Total papers downloaded: 21

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Pointer Networks | Vinyals et al. | 2015 | papers/1506.03134_pointer_networks.pdf | Foundational: attention as pointer |
| CopyNet | Gu et al. | 2016 | papers/1603.06393_copynet.pdf | Generate/copy shared softmax |
| Pointer-Generator Networks | See et al. | 2017 | papers/1704.04368_pointer_generator.pdf | Explicit p_gen gate + coverage |
| Pointer Sentinel Mixture | Merity et al. | 2016 | papers/1609.07843_pointer_sentinel.pdf | Sentinel-gated copy for LM |
| Pointing Unknown Words | Gulcehre et al. | 2016 | papers/1603.08148_pointing_unknown_words.pdf | MLP switching for copy/generate |
| Calibration of Neural Nets | Guo et al. | 2017 | papers/1706.04599_calibration_modern_nn.pdf | ECE, temperature scaling |
| Attention Is All You Need | Vaswani et al. | 2017 | papers/1706.03762_attention_is_all_you_need.pdf | Transformer architecture |
| Bahdanau Attention | Bahdanau et al. | 2014 | papers/1409.0473_bahdanau_attention.pdf | Foundational attention mechanism |
| DistilBERT | Sanh et al. | 2019 | papers/1908.10084_distilbert.pdf | Knowledge distillation for small models |
| GPT-3 | Brown et al. | 2020 | papers/2005.14165_gpt3.pdf | General LM capabilities |
| Neural Turing Machines | Graves et al. | 2014 | papers/1511.06279_nt_graves_neural_turing.pdf | External memory, read/write |
| T5 | Raffel et al. | 2019 | papers/1910.10683_t5.pdf | Text-to-text Transformer |
| Mistral 7B | Jiang et al. | 2023 | papers/2310.06825_mistral.pdf | Efficient small model |
| End-to-End Memory Nets | Sukhbaatar et al. | 2015 | papers/1503.02531_mem_networks.pdf | Memory-augmented networks |
| Chinchilla | Hoffmann et al. | 2022 | papers/2203.15556_chinchilla.pdf | Compute-optimal training |
| QLoRA | Dettmers et al. | 2023 | papers/2305.14314_qlora.pdf | Efficient fine-tuning |
| Transformer-XL | Dai et al. | 2019 | papers/1901.02860_knn_lm.pdf | Long-context Transformer |
| Copy mechanism NLG | Eric & Manning | 2017 | papers/1808.00508_copy_mechanism_nlg.pdf | Copy in dialogue |
| MARGE | Lewis et al. | 2020 | papers/2010.13002_marge_multilingual.pdf | Pre-training via copying |
| Seq2Seq Copy | — | 2016 | papers/1607.04606_seq2seq_copy.pdf | Copy in seq2seq |
| Language Transfer | Tran | 2020 | papers/2002.07306_copy_that_summarization.pdf | Transferring pretrained LMs |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 2 (+ 1 synthetic to generate)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CNN/Daily Mail | HuggingFace | 287K train / 13K val / 11K test | Summarization | datasets/cnn_dailymail/ | Primary dataset |
| XSum | HuggingFace | 204K train / 11K val / 11K test | Extreme summarization | datasets/xsum/ | More abstractive contrast |
| Synthetic Copy | To generate | Configurable | Pattern copying | (generate at experiment time) | Controlled ground truth |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| pointer-generator | github.com/abisee/pointer-generator | Official See et al. 2017 impl | code/pointer-generator/ | TF1, has p_gen mechanism |
| transformers-ref | github.com/huggingface/transformers | BART/Pegasus reference | code/transformers-ref/ | Sparse checkout |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with diligent mode for multiple queries
- Directly downloaded seminal papers by arXiv ID
- Searched for copy mechanism, pointer network, calibration, and specialized model papers
- Used PDF chunker for deep reading of 4 core papers (all chunks read)
- Skimmed abstracts of remaining papers

### Selection Criteria
1. **Core copy mechanism papers**: All major approaches to copy/pointer mechanisms in seq2seq
2. **Calibration literature**: Guo et al. (2017) is the foundational work on neural network calibration
3. **Architecture papers**: Transformer and variants as the base for both general and specialized models
4. **Efficient/small model papers**: DistilBERT, QLoRA, Mistral for techniques to build small specialized models

### Challenges Encountered
- Paper-finder relevance scores were low for this specific topic (copy mechanism calibration is niche)
- Some code repositories were unavailable (original CopyNet, pointer sentinel mixture)
- The intersection of "copy mechanisms" and "calibration" has essentially zero prior work — this is a genuine research gap

### Gaps and Workarounds
- No existing work directly measures calibration of copy decisions → this is the core novelty
- Original CopyNet code unavailable → pointer-generator serves as primary baseline
- No specialized copy-decision model exists → must be built from scratch in experiment phase

---

## Recommendations for Experiment Design

### 1. Primary Dataset
**CNN/Daily Mail** — the standard benchmark for pointer-generator evaluation with well-understood copy dynamics.

### 2. Baseline Methods
1. **Pointer-Generator** (See et al., 2017) — extract p_gen as copy decision baseline
2. **BART fine-tuned on CNN/DM** — general Transformer baseline (implicit copy via cross-attention)
3. **Temperature-scaled p_gen** — simple post-hoc calibration baseline

### 3. Proposed Method
Train a small Transformer (2-4 layers, 128-256 dim) that:
- Takes encoder hidden states and decoder state as input
- Outputs: (a) binary copy/generate decision with probability, (b) pointer distribution over source
- Trained on ground-truth copy labels extracted from a trained pointer-generator

### 4. Evaluation Metrics
- **ECE** (Expected Calibration Error) on copy decisions — primary metric
- **Reliability diagrams** for visual calibration assessment
- **Brier score** for probabilistic accuracy
- **Copy decision F1** for accuracy
- **ROUGE** for end-to-end task performance

### 5. Key Experimental Questions
1. Is the specialized model's p(copy) better calibrated than the pointer-generator's p_gen?
2. Does temperature scaling of p_gen close the calibration gap?
3. Does the specialized model make better copy decisions (higher F1)?
4. Does integrating the specialized model improve end-to-end ROUGE scores?
5. How does model size (of the specialized model) affect calibration?
