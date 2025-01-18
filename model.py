import tensorflow as tf
import numpy as np

# Define the SimpleTransformer model class which inherits from tf.keras.Model
class SimpleTransformer(tf.keras.Model):
    
    def __init__(self, vocab_size, d_model=64, num_heads=2, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize the parameters of the model
        self.vocab_size = vocab_size  # Vocabulary size
        self.d_model = d_model  # Dimensionality of the model (hidden size)
        self.num_heads = num_heads  # Number of attention heads
        self.num_layers = num_layers  # Number of transformer layers
        
        # Embedding layer to convert input tokens to dense vectors
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        # Positional encoding layer to add position information to the embedding
        self.pos_encoding = self.positional_encoding(vocab_size, d_model)
        
        # A list of transformer blocks to be stacked on top of each other
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ]
        
        # Final dense layer to produce output logits for each vocabulary token
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    # This method returns a configuration of the model which can be used for saving/loading
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        })
        return config
    
    # Class method to instantiate a model from a given configuration
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    # Method to compute positional encoding
    def positional_encoding(self, position, d_model):
        # Generate a matrix of shape (position, d_model)
        angles = np.arange(position)[:, np.newaxis] / np.power(
            10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model
        )
        
        # Apply sine and cosine to odd and even indices of the angles
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        # Add an extra dimension for batch size
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    # The main forward pass of the transformer model
    def call(self, x):
        seq_len = tf.shape(x)[1]  # Sequence length from input tensor
        
        # Pass input through embedding layer
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scale by the sqrt of model dim
        x += self.pos_encoding[:, :seq_len, :]  # Add positional encoding
        
        # Pass through all transformer blocks sequentially
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Final dense layer to output predictions for each token in the sequence
        return self.final_layer(x)

# Define a TransformerBlock which performs self-attention and feed-forward processing
class TransformerBlock(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        # Feed-forward neural network (FFN)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),  # First dense layer with ReLU
            tf.keras.layers.Dense(d_model)  # Second dense layer
        ])
        
        # Layer normalization layers for stability
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    # This method returns the configuration of the transformer block for saving/loading
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config
    
    # Class method to instantiate a transformer block from configuration
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    # The main forward pass of the transformer block
    def call(self, x):
        # Perform multi-head attention
        attention_output = self.attention(x, x)
        # Apply the first layer normalization
        x1 = self.layernorm1(x + attention_output)
        
        # Pass through the feed-forward network
        ffn_output = self.ffn(x1)
        # Apply the second layer normalization
        return self.layernorm2(x1 + ffn_output)
