"""Backbone network utilities."""
import torch
import torch.nn as nn
from stable_pretraining.backbone.utils import from_huggingface
from typing import Tuple


# Known backbone embedding dimensions
BACKBONE_DIMS = {
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024,
    "facebook/dinov2-giant": 1536,
    "google/vit-base-patch16-224": 768,
    "google/vit-large-patch16-224": 1024,
    "vit_base_patch16": 768,
    "vit_large_patch16": 1024,
    "vit_huge_patch14": 1280,
}


def load_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, torch.device]:
    """Load a backbone network from HuggingFace.
    
    Args:
        backbone_name: Name of the backbone model
        pretrained: Whether to load pretrained weights
        
    Returns:
        Tuple of (backbone, device)
    """
    backbone = from_huggingface(backbone_name, pretrained=pretrained)
    
    # Ensure all parameters are trainable
    for p in backbone.parameters():
        p.requires_grad = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return backbone, device


def get_backbone_dim(backbone_name: str) -> int:
    """Get the embedding dimension for a backbone.
    
    Args:
        backbone_name: Name of the backbone model
        
    Returns:
        Embedding dimension
        
    Raises:
        ValueError: If backbone is unknown (falls back to 384)
    """
    if backbone_name in BACKBONE_DIMS:
        return BACKBONE_DIMS[backbone_name]
    else:
        print(f"Warning: Unknown backbone {backbone_name}, assuming embed_dim=384")
        return 384


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
