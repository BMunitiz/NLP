
# Character‑Level LSTM Text Generation 

This repository contains an improved Jupyter notebook that trains a character‑level LSTM model on Nietzsche’s writings and generates new text with adjustable “temperature” (randomness). The improvements over a basic implementation focus on memory efficiency, training stability, code clarity, and robust sampling.

## Overview

The notebook:

- Downloads Nietzsche’s text (`nietzsche.txt`) from a public URL.
- Converts the text to lowercase and builds character‑level vocabulary.
- Uses **integer encoding** and an **Embedding layer** (instead of one‑hot encoding) to drastically reduce memory usage.
- Implements a **generator** that yields batches on the fly – no need to store the whole dataset in RAM.
- Builds an LSTM model with **dropout**, **recurrent dropout**, and **gradient clipping** for stable training.
- Saves the best model during training via `ModelCheckpoint` and stops early if loss does not improve.
- Generates text at the end of each epoch with multiple temperatures for monitoring.
- Fixes a common bug where the seed text was not reset between temperature runs.
- Provides a final generation function that uses `np.random.choice` to avoid numerical precision errors.

## Key Improvements

| Area | Improvement |
|------|--------------|
| **Memory** | Integer encoding + Embedding layer; data generator. |
| **Stability** | Dropout (0.2), recurrent dropout (0.2), gradient clipping (norm=1.0). |
| **Training** | Single `fit` call with callbacks; Adam optimizer; early stopping. |
| **Generation** | Resets seed for each temperature; uses `np.random.choice` for robust sampling. |
| **Code Quality** | Functions and classes; clear markdown explanations; typos fixed. |

## Requirements

Install the required packages (preferably in a virtual environment):

```bash
pip install tensorflow numpy
```

The notebook uses only `tensorflow` (>=2.0) and `numpy`. All other modules are part of the Python standard library.

## How to Run

1. Clone this repository or download the notebook `character_level_text_LSTM.ipynb`.
2. Open the notebook in Jupyter Lab / Notebook, or run it in Google Colab.
3. Execute the cells in order. The first run will download the Nietzsche text (~600 KB).
4. Training will start automatically. It will run for up to 60 epochs, but early stopping may halt it earlier if loss stops improving.
5. Generated text will appear both during training (end of each epoch) and after training (final generation cell).

### Example Output (after 30 epochs, temperature 0.5)

```
 of all that women write about "woman," we may well have
constatis wat they nave beenson on pheroodd---that it have baidd and chaidd at loadn ...
```

## Customisation

- **Change the text source**: Replace the `url` in the `get_file` call with any plain‑text file.
- **Adjust model size**: Modify `embedding_dim` and `lstm_units`.
- **Change generation length**: Update the `generate_len` parameter in `TextGenerator` or the `length` argument in `generate_text`.
- **Tune hyperparameters**: `maxlen` (sequence length), `batch_size`, `learning_rate`, `dropout` rates, etc.

## Important Note on Sampling

The sampling function uses temperature reweighting:

```python
probs = exp(preds / temperature) / sum(exp(preds / temperature))
```

- **Low temperature** (e.g., 0.2) → sharp distribution, more predictable text.
- **High temperature** (e.g., 1.2) → flatter distribution, more creative/chaotic output.
- The implementation adds a small epsilon (`1e-10`) to avoid `log(0)` and uses `np.random.choice` to avoid `multinomial` precision issues.

## Results and Discussion

After about 20–30 epochs the model starts producing character sequences that resemble English words and Nietzsche‑like sentence fragments. Longer training (40–60 epochs) yields even better coherence, but may overfit. The combination of dropout and gradient clipping prevents the loss spikes observed in the original implementation.

The notebook provides a solid foundation for experimenting with:

- Deeper LSTM architectures (e.g., stacked LSTMs).
- Bidirectional LSTM layers.
- Different sampling strategies (top‑k, top‑p / nucleus sampling).
- Keeping case sensitivity (removing the `.lower()` call).

