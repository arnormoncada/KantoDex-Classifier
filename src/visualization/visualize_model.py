import logging

import torch
from torch import nn
from torchviz import make_dot


def visualize_model_structure(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> None:
    """
    Visualize the model architecture using torchviz and hiddenlayer.

    Args:
        model (nn.Module): The PyTorch model to visualize.
        input_size (tuple, optional): The size of the dummy input tensor. Defaults to (1, 3, 224, 224).

    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    try:
        # Using torchviz
        logging.info("Generating model visualization with torchviz...")
        y = model(dummy_input)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render("model_architecture_torchviz", format="png")
        logging.info("Model visualization saved as 'model_architecture_torchviz.png'.")
    except Exception as e:
        logging.exception(f"Failed to generate model visualization with torchviz: {e}")
