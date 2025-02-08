import torch
import plotly.express as px
from src.blocks.positional_encoding import PositionalEncoding


import plotly.express as px
import numpy as np


def create_positional_encoding_heatmap(
    input_matrix: torch.tensor,
    title: str = "Positional Encodings Heatmap",
    color_scale: str = "viridis",
):
    """ """
    # Generate positional encoding matrix
    positional_encodings = PositionalEncoding(
        input_matrix
    ).generate_positional_encoding_values()
    # Create heatmap using Plotly Express
    fig = px.imshow(
        positional_encodings,
        labels=dict(x="Encoding Dimension", y="Position", color="Encoding Value"),
        title=title,
        color_continuous_scale=color_scale,
    )
    return fig


# Example usage:
if __name__ == "__main__":
    # Generate and display the heatmap
    fig = create_positional_encoding_heatmap(torch.randn(100, 300, dtype=torch.float64))
    fig.show()
