from src.blocks import positional_encoding, self_attention

input_matrix = positional_encoding.generate_positional_encoding(2, 2)

attention = self_attention.SelfAttention(input_matrix, 4)
print(attention.compute_attention_weights())
