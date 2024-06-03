from src.blocks import positional_encoding, self_attention
from src.blocks.positional_encoding import PositionalEncoding
from sentence_transformers import SentenceTransformer
import torch


model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=4)
docs = ["man", "dog"]
input_matrix = torch.from_numpy(model.encode(docs))

positional_encoding_class = PositionalEncoding(input_matrix)
positiona_encoding_matrix = (
    positional_encoding_class.generate_positional_encoding_matrix()
)

attention = self_attention.SelfAttention(input_matrix, 2)
print(attention.scaled_dot_product_attention())
