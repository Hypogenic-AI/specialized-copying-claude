# Cloned Repositories

## Repo 1: pointer-generator (See et al., 2017)
- **URL**: https://github.com/abisee/pointer-generator
- **Purpose**: Reference implementation of the Pointer-Generator Network with coverage mechanism
- **Location**: `code/pointer-generator/`
- **Language**: Python (TensorFlow 1.x)
- **Key files**:
  - `model.py`: Core pointer-generator model with p_gen computation
  - `attention_decoder.py`: Attention mechanism with coverage
  - `batcher.py`: Data loading for CNN/Daily Mail
  - `run_summarization.py`: Training and evaluation entry point
- **Notes**: This is the official implementation from the paper authors. The p_gen mechanism in `model.py` is the primary comparison target for our specialized copy-decision model. The model uses TF1.x and may need adaptation for modern frameworks. Key insight: p_gen is computed as sigmoid(w_h*h* + w_s*s + w_x*x + b) — a simple linear combination of context vector, decoder state, and decoder input.

## Repo 2: transformers-ref (HuggingFace)
- **URL**: https://github.com/huggingface/transformers
- **Purpose**: Reference implementations of BART and Pegasus for modern abstractive summarization
- **Location**: `code/transformers-ref/` (sparse checkout: BART and Pegasus models only)
- **Key files**:
  - `src/transformers/models/bart/`: BART model with copy-like generation
  - `src/transformers/models/pegasus/`: Pegasus summarization model
- **Notes**: These are modern Transformer-based summarization models that handle copying implicitly through cross-attention. They serve as the "general model" baseline — large Transformers trained for the full task that make implicit copy decisions through their attention patterns rather than an explicit copy mechanism.

## Recommended Additional Repositories (not cloned due to access)

### CopyNet
- Search for PyTorch reimplementations on GitHub (original Theano code is not available)
- Key search: "copynet pytorch" or "copy mechanism pytorch"

### Pointer Sentinel
- The original code for Merity et al. (2016) used the AWD-LSTM codebase
- Search: "pointer sentinel mixture pytorch"

## How These Repos Support the Research

1. **pointer-generator**: Extract p_gen values from trained model → these become training labels for the specialized copy-decision model
2. **transformers-ref**: Provides modern Transformer baselines (BART) where cross-attention implicitly handles copying
3. The experiment should:
   - Train a pointer-generator on CNN/DM
   - Extract p_gen values and attention distributions as ground truth
   - Train a small specialized Transformer to predict these
   - Compare calibration (ECE, reliability diagrams) of both
