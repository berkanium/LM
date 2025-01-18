import tensorflow as tf
import pickle
import os
import numpy as np

# Function to load the trained model and tokenizer from the disk
def load_model_and_tokenizer():
    save_dir = "result_model"  # Directory where the model and tokenizer are saved
     
    # Load the trained model
    model_path = os.path.join(save_dir, "final_model.keras")
    model = tf.keras.models.load_model(model_path)
     
    # Load the tokenizer from a pickle file
    tokenizer_path = os.path.join(save_dir, "tokenizer.pickle")
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer

# Function to generate text based on a seed text and the trained model
def generate_text(model, tokenizer, seed_text, num_words=50, temperature=0.7):
    # Convert seed text to lowercase and prepare it for tokenization
    seed_text = seed_text.lower()
     
    # Convert the seed text into a sequence of token IDs
    current_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    generated_text = seed_text
    
    # Reverse the tokenizer's word index to map back from token IDs to words
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # Generate the next num_words words based on the seed text
    for _ in range(num_words): 
        # Ensure the current sequence is of length 50 by padding if necessary
        padded_sequence = current_sequence[-50:]   
        if len(padded_sequence) < 50:
            padded_sequence = [0] * (50 - len(padded_sequence)) + padded_sequence
             
        # Reshape the sequence to match the input shape of the model
        padded_sequence = np.array(padded_sequence)[np.newaxis, :]
             
        # Predict the next word based on the current sequence
        predictions = model.predict(padded_sequence, verbose=0)[0]
         
        # Get the logits for the next word
        next_word_logits = predictions[-1]
         
        # Apply temperature scaling to the logits to control the randomness of predictions
        scaled_logits = next_word_logits / temperature 
        scaled_logits = scaled_logits - np.max(scaled_logits)  # Prevent overflow
        exp_logits = np.exp(scaled_logits)  # Convert logits to probabilities
        probabilities = exp_logits / np.sum(exp_logits)  # Normalize to sum to 1
         
        # Clip probabilities to avoid extreme values
        probabilities = np.clip(probabilities, 1e-7, 1.0)
        probabilities = probabilities / np.sum(probabilities)
        
        try: 
            # Sample the next word based on the probabilities
            predicted_id = np.random.choice(len(probabilities), p=probabilities)
             
            # Skip the <OOV> token or padding token
            if predicted_id == 0 or predicted_id == tokenizer.word_index.get('<OOV>', 0):
                continue
                 
            # Convert the predicted token ID back to a word
            predicted_word = reverse_word_index.get(predicted_id, '')
            if predicted_word:
                generated_text += ' ' + predicted_word
                current_sequence.append(predicted_id)
                
        except ValueError as e:
            # Handle any errors during sampling
            print(f"Error in sampling: {e}")
            continue
    
    return generated_text

# Main function to load model, tokenizer and generate text with different temperatures
if __name__ == "__main__": 
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
     
    # Seed texts to start generating from
    seed_texts = [
        "Herkesin bildiği bir gerçektir ki",
        "Elizabeth Bennet, pek de güzel ",
        "Bay Darcy uzun boylu ",
    ]
     
    # Different temperatures to experiment with (controls creativity/randomness)
    temperatures = [0.2, 0.3, 0.5]
    
    # Generate text for each combination of temperature and seed text
    for temp in temperatures:
        print(f"\nGenerating with temperature {temp}:")
        print("-" * 50)
        for seed_text in seed_texts:
            print(f"\nSeed: {seed_text}")
            generated_text = generate_text(
                model, 
                tokenizer, 
                seed_text, 
                num_words=30,  # Number of words to generate
                temperature=temp  # Set the temperature for text generation
            )
            print(f"Generated: {generated_text}\n")
            print("-" * 50)
