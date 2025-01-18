# Text Generation Using a Transformer Model

This repository demonstrates how to build, train, and use a simple Transformer model to generate text using TensorFlow. The project involves training the model on text data, saving the trained model and tokenizer, and using them to generate new text sequences based on a seed text.

## Features

- A custom Transformer model (`SimpleTransformer`) implemented using TensorFlow.
- Training on a provided text dataset (e.g., from a PDF file).
- Saving and loading the trained model and tokenizer.
- Generating new text sequences based on seed input with customizable randomness (temperature).

---

## Files and Structure

### Key Python Files

1. **`train.py`**
   - Loads a dataset (PDF format) and preprocesses it.
   - Trains a custom Transformer model (`SimpleTransformer`).
   - Saves the trained model, tokenizer, and training history for future use.

2. **`predict.py`**
   - Loads the trained Transformer model and tokenizer.
   - Generates new text sequences based on a seed input.
   - Allows experimentation with different temperature values to control randomness.

3. **`model.py`**
   - Contains the definition of the `SimpleTransformer` class and its building blocks.
   - Includes positional encoding and Transformer blocks.

### Output Directories

- `result_model/`: Directory where the trained model, tokenizer, and training history are saved.

---

## Setup

### Requirements

- Python 3.8 or later
- TensorFlow 2.11 or later
- NumPy
- PyPDF2
- Pickle (standard library)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/berkanium/LM.git
   cd LM
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Training the Model

To train the model on a PDF file:

1. Create `data` directory and reeplace the path to your PDF file in `train.py`:
   ```python
   pdf_path = "data/YourPDFFile.pdf"
   ```

2. Run the `train.py` script:
   ```bash
   python train.py
   ```

This will:
- Extract and preprocess text from the PDF.
- Train the Transformer model on the text.
- Save the trained model and tokenizer in the `result_model/` directory.

### 2. Generating Text

After training, you can generate text using the trained model:

1. Run the `predict.py` script:
   ```bash
   python predict.py
   ```

2. Modify the seed texts and temperatures in `predict.py` to test different configurations:
   ```python
   # Seed texts to start generating from
   seed_texts = [
       "Herkesin bildiği bir gerçektir ki",
       "Elizabeth Bennet, pek de güzel ",
       "Bay Darcy uzun boylu ",
   ]
   temperatures = [0.2, 0.3, 0.5]
   ```

This script will load the saved model and tokenizer, generate text for each seed input, and display the results in the console.

---

## How It Works

### 1. Model Architecture

The custom `SimpleTransformer` is defined in `model.py`. It includes:
- **Embedding Layer**: Converts token IDs to dense vectors.
- **Positional Encoding**: Adds positional information to embeddings.
- **Transformer Blocks**: Consists of multi-head self-attention and feed-forward networks with residual connections and layer normalization.
- **Final Dense Layer**: Outputs logits for each token in the vocabulary.

### 2. Training Pipeline

- **Data Preprocessing**: Tokenizes the text data, converts it into sequences, and prepares training inputs and targets.
- **Training**: Uses the `SparseCategoricalCrossentropy` loss and the Adam optimizer. The model is trained for 10 epochs by default, with validation splitting and model checkpointing.
- **Saving**: The trained model and tokenizer are saved to disk for later use.

### 3. Text Generation

- **Seed Text**: Provided by the user, tokenized into sequences.
- **Prediction Loop**: For each word to be generated:
  1. Predict the next word's probabilities using the model.
  2. Scale logits using the temperature parameter to control randomness.
  3. Sample a word from the probability distribution.
  4. Append the sampled word to the generated sequence.
- The process continues for a specified number of words.

---

## Parameters

### Model Parameters

- **`vocab_size`**: Size of the vocabulary used during tokenization (default: 10,000).
- **`d_model`**: Dimensionality of the embedding and model layers (default: 128).
- **`num_heads`**: Number of attention heads in each Transformer block (default: 4).
- **`num_layers`**: Number of Transformer blocks (default: 3).

### Text Generation Parameters

- **`seed_text`**: Initial text used to start the generation process.
- **`num_words`**: Number of words to generate (default: 50).
- **`temperature`**: Controls the randomness of predictions:
  - Low values (e.g., 0.2): More deterministic, repetitive results.
  - High values (e.g., 1.0): More random, creative results.

---

## Example Output

### Seed Text: \"Herkesin bildiği bir gerçektir ki\"

#### Temperature 0.2:
```
Herkesin bildiği bir gerçektir ki hayat her zaman bir mücadele ve bu mücadele içinde kazanmak için çok çalışmak gerekir...
```

#### Temperature 0.5:
```
Herkesin bildiği bir gerçektir ki insanlar hayatta mutluluğu bulmak için cesaretle adımlar atmalı ve yeni deneyimler...
```

#### Temperature 1.0:
```
Herkesin bildiği bir gerçektir ki yıldızların altında yapılan danslar, eski zaman masallarında unutulan melodilerle...
```

---

## Notes

- This project uses a basic Transformer implementation for demonstration purposes and is not optimized for large-scale datasets.
- Experiment with different hyperparameters (e.g., model depth, learning rate) to achieve better results for your specific use case.

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- TensorFlow Documentation: [Text Generation with Transformers](https://www.tensorflow.org/tutorials/text/transformer)

---

## License

This project is licensed under the MIT License.
