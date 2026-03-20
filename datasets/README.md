# Downloaded Datasets

This directory contains datasets for the research project "Is Copying More Calibrated if It's Specialized?"

Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: CNN/Daily Mail (Primary)

### Overview
- **Source**: HuggingFace `cnn_dailymail` (version 3.0.0)
- **Size**: 287,113 train / 13,368 validation / 11,490 test examples
- **Format**: HuggingFace Dataset (Arrow format)
- **Task**: Abstractive text summarization
- **Avg article length**: ~781 tokens
- **Avg summary length**: ~56 tokens (3.75 sentences)
- **License**: Apache 2.0

### Why This Dataset
- Standard benchmark for pointer-generator networks (See et al., 2017)
- Articles are long enough that copy decisions are non-trivial
- Clear copy-vs-generate dynamics: names, numbers, and quotes tend to be copied
- Large enough for meaningful calibration analysis

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
dataset.save_to_disk("datasets/cnn_dailymail")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/cnn_dailymail")
```

### Fields
- `article`: Full news article text
- `highlights`: Multi-sentence summary
- `id`: Unique article identifier

---

## Dataset 2: XSum (Secondary - More Abstractive)

### Overview
- **Source**: HuggingFace `EdinburghNLP/xsum`
- **Size**: 204,045 train / 11,332 validation / 11,334 test examples
- **Format**: HuggingFace Dataset (Arrow format)
- **Task**: Extreme abstractive summarization (single-sentence summaries)
- **License**: MIT

### Why This Dataset
- More abstractive than CNN/DM — summaries require more generation vs copying
- Tests whether the specialized copy model correctly avoids copying when generation is needed
- Good contrast dataset for calibration: model should have lower p_gen values here

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("EdinburghNLP/xsum")
dataset.save_to_disk("datasets/xsum")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/xsum")
```

### Fields
- `document`: Full BBC article text
- `summary`: Single-sentence summary
- `id`: Unique article identifier

---

## Dataset 3: Synthetic Copy Tasks (To Be Generated)

### Overview
For controlled experiments, the experiment runner should generate synthetic copy tasks
following the protocol from CopyNet (Gu et al., 2016):

1. Generate random transformation rules with variables (x, y)
2. Create instances by replacing variables with random subsequences
3. Five rule types: x→∅, x→x, x→xx, xy→x, xy→xy

### Why This Dataset
- Ground truth copy decisions are known exactly
- Controlled evaluation without confounds from language understanding
- Can systematically vary copy difficulty

### Generation Instructions
```python
# See CopyNet paper Section 5.1 for full protocol
# Vocabulary size: 1000 symbols
# Rule length: 5-20 symbols
# Variable replacement length: 1-15 symbols
# 200 rules × 200 instances each, 50/50 train/test split
```

---

## Notes
- CNN/Daily Mail is the primary evaluation dataset
- XSum provides a contrast condition (high abstraction → less copying)
- Synthetic tasks provide controlled ground-truth evaluation
- All datasets are locally available in this directory after download
