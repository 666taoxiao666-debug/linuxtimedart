#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core components of the Cabin model.

This module contains the main building blocks of the Cabin architecture:
- AmbientRepresentationModule (ARM)
- CollaborationAmbientVariables (CAV)
"""

import torch
import torch.nn as nn
from typing import Literal

from .base import GAR, KAN


class AmbientRepresentationModule(nn.Module):
    """
    Ambient Representation Module (ARM).
    
    Performs multi-dimensional feature extraction across sample, temporal, and feature dimensions
    to capture complex dependencies within input data.
    
    Args:
        window_size: Length of the input sequence.
        input_size: Number of input features/variables.
        dim: Dimension along which to apply softmax normalization.
             0 = sample axis, 1 = temporal axis, 2 = feature axis, -1 = no softmax.
        use_activate: Whether to apply ReLU activation.
        dropout: Dropout rate (currently unused, kept for compatibility).
    """
    
    def __init__(self, window_size: int, input_size: int, dim: int = 0,
                 use_activate: bool = True, dropout: float = 0.1):
        super(AmbientRepresentationModule, self).__init__()

        self.window_size = window_size
        self.input_size = input_size
        self.dim = dim
        self.use_activate = use_activate

        # Learnable weight matrix for element-wise scaling
        self.weight = nn.Parameter(torch.zeros(self.window_size, self.input_size))
        nn.init.uniform_(self.weight, a=0.01, b=0.1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of ARM.
        
        Args:
            x: Input tensor of shape [batch_size, window_size, input_size].
            
        Returns:
            Processed tensor with same shape as input.
        """
        if self.use_activate:
            x = torch.relu(x)

        # Element-wise multiplication with learnable weights
        out = x * self.weight

        # Apply dimension-specific softmax normalization
        if self.dim != -1:
            out = out.softmax(dim=self.dim)

        return out


class CollaborationAmbientVariables(nn.Module):
    """
    Collaboration of Ambient Variables (CAV) Module.
    
    Integrates extracted features from ARM using temporal convolutions and 
    Kolmogorov-Arnold Networks (KAN) to generate final predictions.
    
    Args:
        input_window_size: Length of input sequence.
        output_window_size: Length of output sequence (forecast horizon).
        input_size: Number of input features after ARM processing.
        hidden_size: Hidden dimension size for intermediate representations.
        output_size: Number of output variables to predict.
        use_activate: Whether to apply ReLU activations.
    """
    
    def __init__(self, input_window_size: int, output_window_size: int,
                 input_size: int, hidden_size: int, output_size: int, use_activate: bool = True):
        super(CollaborationAmbientVariables, self).__init__()

        self.use_activate = use_activate
        
        # Mapping unit: Linear transformation to unified representation
        self.mapping = nn.Linear(input_size, hidden_size)
        
        # Global Autoregression for temporal modeling
        self.ar = GAR(input_window_size, output_window_size)
        
        # Kolmogorov-Arnold Network for complex non-linear dependencies
        self.kan = KAN(output_window_size, hidden_size, output_window_size, output_size,
                      layers_hidden=[hidden_size // 2], patch_form=False)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of CAV.
        
        Args:
            x: Input tensor from ARM of shape [batch_size, window_size, input_size].
            
        Returns:
            Predictions of shape [batch_size, output_window_size, output_size].
        """
        # Mapping unit: transform to unified representation
        out = self.mapping(x)

        if self.use_activate:
            out = torch.relu(out)

        # Global autoregression for temporal patterns
        out = self.ar(out)

        if self.use_activate:
            out = torch.relu(out)

        # KAN for complex non-linear mapping
        out = self.kan(out)

        return out
