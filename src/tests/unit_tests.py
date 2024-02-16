from src.blocks import positional_encoding, self_attention
import torch

input_embeddings = torch.rand((5, 3))
positional_econding_vectors = positional_encoding.generate_positional_encoding(
    sequence_length=input_embeddings.size()[0],
    vector_dim=input_embeddings.size()[1],
)
input_matrix = torch.mul(input_embeddings, positional_econding_vectors)
attention = self_attention.SelfAttention(input_matrix, 1)
attention_weights = attention.compute_attention_weights()
