import os
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.model_components.positional_encoding import PositionalEncoding
from src.model_components.patch_projection import PatchLinearProjection
from src.encoder_block.encoder import ViTEncoder
from src.model_components.mlp_head import MLPHead


D_MODEL = 768
N_CLASSES = 3
PATCH_SIZE = 16
N_HEADS = 3
MLP_SIZE = 3072
N_LAYERS = 12
DROPOUT_RATIO = 0.1

# model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=D_MODEL)
# text = "The feline cat"
# docs = text.split(" ")
# input_matrix = torch.from_numpy(model.encode(docs))

image_path = os.path.join(
    Path(__file__).parent.parent.parent, "data", "test", "cat.jpg"
)

# Patch projection + Positional encoding block
x = PatchLinearProjection(patch_size=PATCH_SIZE, d_model=D_MODEL)(image_path)
x = PositionalEncoding(x).forward()

for layer in range(N_LAYERS):
    x = ViTEncoder(D_MODEL, N_HEADS, MLP_SIZE, DROPOUT_RATIO)(x)

# Global Average Pooling for classification instead of using a class token (GAP)
x = x.mean(dim=0)
mlp_head_output = MLPHead(D_MODEL, N_CLASSES)(x)

print(mlp_head_output)
