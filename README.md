# Character-Level LSTM Text Generation

A deep learning project that implements a character-level LSTM (Long Short-Term Memory) model for generating text in the style of Friedrich Nietzsche's writings.

## Overview

This project demonstrates text generation using a recurrent neural network with LSTM layers. The model learns the patterns and style of Nietzsche's philosophical texts and can generate new text that mimics his writing style.

## Features

- **Character-Level Text Generation**: Processes text at the character level rather than word level
- **Temperature Sampling**: Implements temperature-based probability distribution for controlled text generation
- **Real-time Training Monitoring**: Generates sample text during training to observe model improvement
- **Multiple Temperature Settings**: Tests text generation at different creativity levels (0.2, 0.5, 1.0, 1.2)

## Model Architecture

### Neural Network Structure
- **Input Layer**: One-hot encoded character sequences (60 characters)
- **LSTM Layer**: 256 units with return sequences
- **Output Layer**: Dense layer with softmax activation (57 units - one per character)

### Model Parameters
- **Total Parameters**: 336,185
- **Trainable Parameters**: 336,185
- **Input Shape**: (60, 57) - (sequence length, number of unique characters)

## Dataset

- **Source**: Nietzsche's writings from `nietzsche.txt`
- **Corpus Length**: 600,893 characters
- **Unique Characters**: 57 distinct characters
- **Preprocessing**: Text converted to lowercase

## Data Preparation

### Sequence Generation
- **Sequence Length**: 60 characters
- **Step Size**: 3 characters between sequences
- **Total Sequences**: 200,278 training examples

### Vectorization
- One-hot encoding of characters into binary arrays
- Input shape: `(len(sentences), maxlen, len(chars))`
- Output shape: `(len(sentences), len(chars))`

## Key Functions

### `reweight_distribution(original_distribution, temperature=0.5)`
Adjusts probability distribution using temperature parameter:
- Lower temperature (0.2): More deterministic, conservative text
- Higher temperature (1.2): More creative, random text

### `sample(preds, temperature=1.0)`
Samples the next character from model predictions using temperature-controlled randomness

### Text Generation Loop
- Trains model for 60 epochs
- Generates 400-character samples at different temperatures after each epoch
- Uses random seed texts from the original corpus

## Training Configuration

- **Optimizer**: RMSprop with learning rate 0.01
- **Loss Function**: Categorical crossentropy
- **Batch Size**: 128
- **Epochs**: 60
- **Training Time**: ~170 seconds per epoch

## Text Generation Process

1. **Seed Selection**: Random 60-character sequence from source text
2. **Character Prediction**: Model predicts next character probabilities
3. **Temperature Sampling**: Adjusts randomness of character selection
4. **Iterative Generation**: Builds text character by character (400 characters per sample)

## Temperature Effects

- **0.2**: Very conservative, closely follows training patterns
- **0.5**: Balanced between creativity and coherence
- **1.0**: Standard sampling, moderate creativity
- **1.2**: High creativity, more experimental text

## Dependencies

```python
import numpy as np
import keras
from keras import layers
import random
import sys
```

## Usage

1. **Data Loading**: Automatically downloads Nietzsche's texts
2. **Preprocessing**: Converts text to sequences and one-hot encoding
3. **Model Training**: Runs for 60 epochs with batch size 128
4. **Text Generation**: Produces samples at various temperatures during training

## Model Performance

- **Initial Loss**: ~0.82 (epoch 1)
- **Final Loss**: ~0.73 (epoch 30+)
- **Training Stability**: Consistent improvement over 60 epochs
- **Text Quality**: Gradual improvement in coherence and style matching

## Sample Output

The model generates text that:
- Mimics Nietzsche's philosophical style
- Maintains grammatical structure
- Shows thematic coherence with training data
- Varies in creativity based on temperature setting

## Technical Notes

- The model operates at character level, allowing it to learn punctuation and formatting
- Temperature parameter controls the trade-off between creativity and coherence
- Training includes real-time generation to monitor progress
- Model can be extended with multiple LSTM layers or different architectures

## Potential Improvements

- Adding multiple LSTM layers for deeper learning
- Implementing beam search for better text generation
- Adding attention mechanisms
- Training on larger datasets
- Implementing early stopping based on validation loss

## Applications

- Creative writing assistance
- Style imitation
- Text completion
- Educational tools for studying writing styles

This project demonstrates the power of character-level RNNs for understanding and generating text while maintaining the unique style of the training data.
