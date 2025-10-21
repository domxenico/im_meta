"""
IM-META: Influence Maximization with Node Metadata
"""

from .im_meta import IMMETA
from .network_inference import NetworkInference
from .coauthor_data import coauthor_data

__all__ = [
    "IMMETA",
    "NetworkInference", 
    "coauthor_data",
]