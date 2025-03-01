import torch
from sentence_transformers import SentenceTransformer
from src.blocks.multihead_attention import MultiHeadAttention
from src.blocks.positional_encoding import PositionalEncoding
from src.blocks.patch_projection import PatchLinearProjection
from pathlib import Path
import os

d_model = 4

# model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=4)
# text = "The feline cat"
# docs = text.split(" ")
# input_matrix = torch.from_numpy(model.encode(docs))

image_path = os.path.join(
    Path(__file__).parent.parent.parent, "data", "test", "cats.jpg"
)
patch_size = 16
image_patches_class = PatchLinearProjection(patch_size=patch_size, d_model=d_model)
input_matrix = image_patches_class.forward(image_path)

positional_encoding_matrix = PositionalEncoding(input_matrix).forward()
multihead_attention = MultiHeadAttention(
    positional_encoding_matrix,
    positional_encoding_matrix.shape[1],
    3,
).forward()

print(multihead_attention)
