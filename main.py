import tensorflow as tf
from model import SimpleTransformer
import numpy as np
import PyPDF2
import re
import pickle
import os

# Function to read text from a PDF file
def read_pdf(pdf_path):
    text = ""
    # Open the PDF file in binary read mode
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        # Extract text from all pages of the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    # Clean up the text by removing excessive spaces and non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s.,!?-]', '', text)  
    return text.strip()

# Function to preprocess text and tokenize it
def preprocess_text(text, vocab_size=1000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>",  # Token to represent out-of-vocabulary words
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'  # Characters to be removed during tokenization
    )
    
    # Convert the text to lowercase and split into words
    words = text.lower().split()
    
    # Fit the tokenizer on the text
    tokenizer.fit_on_texts([' '.join(words)])
    
    # Convert words to sequences of integers
    sequences = tokenizer.texts_to_sequences([' '.join(words)])[0]
    return sequences, tokenizer

# Function to create training data from the sequences
def create_training_data(sequences, seq_length=50):
    input_sequences = []
    # Generate sequences of fixed length for training
    for i in range(len(sequences) - seq_length):
        input_sequences.append(sequences[i:i + seq_length + 1])
    
    # Split the sequences into input (x) and target (y) data
    x = np.array([seq[:-1] for seq in input_sequences])
    y = np.array([seq[1:] for seq in input_sequences])
    return x, y

def main():
    # Directory to save model and related files
    save_dir = "result_model"
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Path to the PDF file
    pdf_path = "data/JaneAustenPrideAndPrejudice.pdf"
    try:
        # Attempt to read the PDF
        text = read_pdf(pdf_path)
        print(f"Successfully loaded PDF. Text length: {len(text)} characters")
    except Exception as e:
        # Handle errors if PDF cannot be read
        print(f"Error loading PDF: {e}")
        return
    
    # Hyperparameters for model and training
    vocab_size = 10000   # Size of vocabulary
    seq_length = 50      # Length of each input sequence
    d_model = 128        # Dimension of model (hidden size)
    num_heads = 4        # Number of attention heads
    num_layers = 3       # Number of transformer layers
    
    # Preprocess the text and get the tokenized sequences
    sequences, tokenizer = preprocess_text(text, vocab_size)
    # Create training data from sequences
    x_train, y_train = create_training_data(sequences, seq_length)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    # Initialize the transformer model
    model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers)
    
    # Compile the model with an optimizer and loss function
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Path to save model checkpoints during training
    checkpoint_path = os.path.join(save_dir, "model_checkpoint.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,  # Save the full model, not just weights
        save_best_only=True,      # Only save the model with the best validation loss
        monitor='val_loss',
        verbose=1
    )
    
    # Train the model on the data
    history = model.fit(
        x_train, 
        y_train, 
        epochs=10, 
        batch_size=32,
        validation_split=0.1,  # Use 10% of data for validation
        callbacks=[checkpoint_callback]
    )
 
    # Save the trained model to disk
    final_model_path = os.path.join(save_dir, "final_model.keras")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
 
    # Save the tokenizer used for text preprocessing
    tokenizer_path = os.path.join(save_dir, "tokenizer.pickle")
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_path}")
 
    # Save the training history (e.g., loss and accuracy) for later analysis
    history_path = os.path.join(save_dir, "training_history.pickle")
    with open(history_path, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Training history saved to {history_path}")

# Entry point of the script
if __name__ == "__main__":
    main()
