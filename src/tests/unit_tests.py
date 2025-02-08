from src.blocks.positional_encoding import PositionalEncoding
from sentence_transformers import SentenceTransformer
import torch


model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=4)
docs = ["the", "dog", "walks"]
input_matrix = torch.from_numpy(model.encode(docs))
print(input_matrix)

positional_encoding_class = PositionalEncoding(input_matrix)
positiona_encoding_matrix = (
    positional_encoding_class.generate_positional_encoding_matrix()
)
