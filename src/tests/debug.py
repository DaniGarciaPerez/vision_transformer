import torch
from sentence_transformers import SentenceTransformer
from src.blocks.multihead_attention import MultiHeadAttention
from src.blocks.positional_encoding import PositionalEncoding
from src.blocks.patch_projection import PatchLinearProjection
from pathlib import Path
import os

# model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=4)
# text = "The feline cat"
# docs = text.split(" ")
# input_matrix = torch.from_numpy(model.encode(docs))
image_path = os.path.join(
    Path(__file__).parent.parent.parent, "data", "test", "dogs.jpg"
)
image_patches_class = PatchLinearProjection(num_patches=9, d_model=4)
input_matrix = image_patches_class.forward(image_path)
print(input_matrix.shape)

# positional_encoding = PositionalEncoding(input_matrix)
# positional_encoding_matrix = positional_encoding.forward()

# multihead_attention = MultiHeadAttention(
#     positional_encoding_matrix,
#     positional_encoding_matrix.shape[1],
#     3,
# ).forward()

# print(multihead_attention)
