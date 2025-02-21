import torch
from sentence_transformers import SentenceTransformer
from src.blocks.multihead_attention import MultiHeadAttention
from src.blocks.positional_encoding import PositionalEncoding
from src.blocks.patch_projection import PatchLinearProjection
from pathlib import Path
import os

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=4)
text = "The feline cat"
docs = text.split(" ")
input_matrix = torch.from_numpy(model.encode(docs))
print(input_matrix)
print(input_matrix.dtype)
print(input_matrix.shape)
print(type(input_matrix))


# image_patches_class = PatchLinearProjection()
# input_matrix = image_patches_class.split_image_patches(
#     os.path.join(Path(__file__).parent.parent.parent, "data", "test", "dogs.jpg")
# )
# print(input_matrix)
# print(input_matrix.dtype)
# print(input_matrix.shape)
# print(type(input_matrix))

positional_encoding = PositionalEncoding(input_matrix)
positional_encoding_matrix = positional_encoding.forward()

print(positional_encoding_matrix)

multihead_attention = MultiHeadAttention(
    positional_encoding_matrix,
    positional_encoding_matrix.shape[1],
    3,
).forward()
print(multihead_attention)
