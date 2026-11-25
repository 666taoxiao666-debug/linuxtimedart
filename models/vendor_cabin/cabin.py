#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Cabin model implementation.

Cabin: A Collaborative and Adaptive Framework for Wind Power Forecasting 
Integrating Ambient Variables

Authors: Senzhen Wu, Yu Chen, Xinhao He, Zhijin Wang, Xiufeng Liu, Yonggang Fu
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from .components import AmbientRepresentationModule, CollaborationAmbientVariables


class Cabin(nn.Module):
    """
    Cabin: A Collaborative and Adaptive Framework for Wind Power Forecasting
    
    This model integrates historical wind power data with ambient variables through
    a collaborative and adaptive architecture featuring:
    - Ambient Representation Module (ARM) for multi-dimensional feature extraction
    - Collaboration of Ambient Variables (CAV) for sophisticated feature integration
    
    Args:
        input_window_size: Input sequence length (number of historical time steps).
        input_vars: Number of input variables (wind turbines/power series).
        output_window_size: Output sequence length (forecast horizon).
        output_vars: Number of output variables (typically same as input_vars).
        ex_retain_window_size: Input window size for exogenous variables (deprecated, kept for compatibility).
        ex_vars: Number of exogenous/ambient variables.
        hidden_size: Hidden dimension size for CAV layers.
        use_activation: Whether to use ReLU as activation function.
        dropout_rate: Dropout rate (currently unused, kept for compatibility).
        mode: Architecture variant - 'data_first', 'learning_first', or 'only_target'.
        use_temporal_softmax: Whether to use temporal softmax in ARM.
        use_feature_wise_softmax: Whether to use feature-wise softmax in ARM.
        use_sample_softmax: Whether to use sample-wise softmax in ARM.
    """

    def __init__(self, input_window_size: int = 0, input_vars: int = 0,
                 output_window_size: int = 1, output_vars: int = 1,
                 ex_retain_window_size: int = 0, ex_vars: int = 0,
                 hidden_size: int = 64, use_activation: bool = True, dropout_rate: float = 0.,
                 mode: Literal['data_first', 'learning_first', 'only_target'] = 'data_first',
                 use_temporal_softmax: bool = True,
                 use_feature_wise_softmax: bool = True,
                 use_sample_softmax: bool = True):
        super(Cabin, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars

        self.hidden_size = hidden_size
        self.mode = mode
        self.use_arm0 = use_sample_softmax
        self.use_arm1 = use_temporal_softmax
        self.use_arm2 = use_feature_wise_softmax

        # Initialize ARM and CAV modules based on the selected mode
        if self.mode == 'data_first':
            # Data-first: concatenate power and ambient data early, process jointly
            total_vars = self.input_vars + self.ex_vars
            
            # ARM modules for different softmax dimensions
            self.arm0 = AmbientRepresentationModule(
                self.input_window_size, total_vars, 0, use_activation, dropout_rate)
            self.arm1 = AmbientRepresentationModule(
                self.input_window_size, total_vars, 1, use_activation, dropout_rate)
            self.arm2 = AmbientRepresentationModule(
                self.input_window_size, total_vars, 2, use_activation, dropout_rate)

            # CAV module input size: original data + ARM outputs (if enabled)
            cav_input_size = total_vars * (1 + self.use_arm0 + self.use_arm1 + self.use_arm2)
            self.cav = CollaborationAmbientVariables(
                self.input_window_size, self.output_window_size, 
                cav_input_size, self.hidden_size, self.output_vars, use_activation)

        elif self.mode == 'learning_first':
            # Learning-first: separate processing for power and ambient data, then fusion
            
            # ARM modules for power data
            self.arm0 = AmbientRepresentationModule(
                self.input_window_size, self.input_vars, 0, use_activation, dropout_rate)
            self.arm1 = AmbientRepresentationModule(
                self.input_window_size, self.input_vars, 1, use_activation, dropout_rate)
            self.arm2 = AmbientRepresentationModule(
                self.input_window_size, self.input_vars, 2, use_activation, dropout_rate)

            # ARM modules for ambient data
            self.ex_arm0 = AmbientRepresentationModule(
                self.input_window_size, self.ex_vars, 0, use_activation, dropout_rate)
            self.ex_arm1 = AmbientRepresentationModule(
                self.input_window_size, self.ex_vars, 1, use_activation, dropout_rate)
            self.ex_arm2 = AmbientRepresentationModule(
                self.input_window_size, self.ex_vars, 2, use_activation, dropout_rate)

            # CAV module input size: combined features from both data types
            cav_input_size = (self.input_vars + self.ex_vars) * (1 + self.use_arm0 + self.use_arm1 + self.use_arm2)
            self.cav = CollaborationAmbientVariables(
                self.input_window_size, self.output_window_size,
                cav_input_size, self.hidden_size, self.output_vars, use_activation)

        elif self.mode == 'only_target':
            # Only-target: use only historical power data
            
            # ARM modules for power data only
            self.arm0 = AmbientRepresentationModule(
                self.input_window_size, self.input_vars, 0, use_activation, dropout_rate)
            self.arm1 = AmbientRepresentationModule(
                self.input_window_size, self.input_vars, 1, use_activation, dropout_rate)
            self.arm2 = AmbientRepresentationModule(
                self.input_window_size, self.input_vars, 2, use_activation, dropout_rate)

            # CAV module input size: power data features only
            cav_input_size = self.input_vars * (1 + self.use_arm0 + self.use_arm1 + self.use_arm2)
            self.cav = CollaborationAmbientVariables(
                self.input_window_size, self.output_window_size,
                cav_input_size, self.hidden_size, self.output_vars, use_activation)

        else:
            raise ValueError(f'Unknown mode: {self.mode}. Must be one of: data_first, learning_first, only_target')

    def forward(self, x: torch.Tensor, e: Optional[torch.Tensor] = None):
        """
        Forward pass of the Cabin model.
        
        Args:
            x: Historical wind power data of shape [batch_size, input_window_size, input_vars].
            e: Ambient variable data of shape [batch_size, input_window_size, ex_vars].
               Required for 'data_first' and 'learning_first' modes, ignored for 'only_target'.
               
        Returns:
            Predictions of shape [batch_size, output_window_size, output_vars].
        """

        if self.mode == 'data_first':
            if e is None:
                raise ValueError("Ambient variable data 'e' is required for 'data_first' mode")
            
            # Concatenate power and ambient data
            out = torch.cat([x, e], dim=2)

            # Apply ARM modules and collect outputs
            cat_list = []
            if self.use_arm0:
                out0 = self.arm0(out)
                cat_list.append(out0)
            if self.use_arm1:
                out1 = self.arm1(out)
                cat_list.append(out1)
            if self.use_arm2:
                out2 = self.arm2(out)
                cat_list.append(out2)
            cat_list.append(out)  # Original concatenated data

            # Concatenate all features
            out = torch.cat(cat_list, dim=2)

            # Generate predictions through CAV
            out = self.cav(out)

            return out

        elif self.mode == 'learning_first':
            if e is None:
                raise ValueError("Ambient variable data 'e' is required for 'learning_first' mode")
            
            # Process power data through ARM modules
            cat_list, ex_cat_list = [], []
            if self.use_arm0:
                out0 = self.arm0(x)
                ex_out0 = self.ex_arm0(e)
                cat_list.append(out0)
                ex_cat_list.append(ex_out0)
            if self.use_arm1:
                out1 = self.arm1(x)
                ex_out1 = self.ex_arm1(e)
                cat_list.append(out1)
                ex_cat_list.append(ex_out1)
            if self.use_arm2:
                out2 = self.arm2(x)
                ex_out2 = self.ex_arm2(e)
                cat_list.append(out2)
                ex_cat_list.append(ex_out2)
            
            # Add original data
            cat_list.append(x)
            ex_cat_list.append(e)

            # Concatenate all features from both data types
            out = torch.cat(cat_list + ex_cat_list, dim=2)

            # Generate predictions through CAV
            out = self.cav(out)

            return out

        elif self.mode == 'only_target':
            # Process only power data
            cat_list = []
            if self.use_arm0:
                out0 = self.arm0(x)
                cat_list.append(out0)
            if self.use_arm1:
                out1 = self.arm1(x)
                cat_list.append(out1)
            if self.use_arm2:
                out2 = self.arm2(x)
                cat_list.append(out2)
            cat_list.append(x)  # Original power data

            # Concatenate all power features
            out = torch.cat(cat_list, dim=2)

            # Generate predictions through CAV
            out = self.cav(out)

            return out

    def get_model_info(self):
        """
        Get information about the model configuration.
        
        Returns:
            Dictionary containing model configuration details.
        """
        return {
            'model_name': 'Cabin',
            'mode': self.mode,
            'input_window_size': self.input_window_size,
            'input_vars': self.input_vars,
            'output_window_size': self.output_window_size,
            'output_vars': self.output_vars,
            'ex_vars': self.ex_vars,
            'hidden_size': self.hidden_size,
            'use_sample_softmax': self.use_arm0,
            'use_temporal_softmax': self.use_arm1,
            'use_feature_wise_softmax': self.use_arm2,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
