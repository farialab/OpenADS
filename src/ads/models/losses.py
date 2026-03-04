"""
Hybrid loss function for ADS model training

This module implements the hybrid loss function described in the paper.
It combines generalized dice loss, balanced binary cross-entropy, and L1 regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Tuple, Dict

class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss as described in the paper
    
    This loss automatically handles class imbalance by weighting classes
    by the inverse of their volume.
    """
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Generalized Dice Loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
        
    def forward(self, 
               pred: torch.Tensor, 
               target: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Dice Loss
        
        Args:
            pred: Predicted segmentation (B, C, D, H, W) or (B, C, H, W)
            target: Target segmentation (B, C, D, H, W) or (B, C, H, W)
            
        Returns:
            Generalized Dice Loss value
        """
        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate class weights (inverse of squared volume)
        # w_l = 1 / (sum_l)^2
        class_weights = 1.0 / (torch.sum(target_flat, dim=1) ** 2 + self.smooth)
        
        # Calculate intersection and union
        intersection = torch.sum(pred_flat * target_flat, dim=1)
        union = torch.sum(pred_flat, dim=1) + torch.sum(target_flat, dim=1)
        
        # Apply class weights to intersection and union
        weighted_intersection = class_weights * intersection
        weighted_union = class_weights * union
        
        # Calculate generalized dice
        dice = (2.0 * torch.sum(weighted_intersection) + self.smooth) / (torch.sum(weighted_union) + self.smooth)
        
        # Return loss (1 - Dice)
        return 1.0 - dice

class BalancedBCELoss(nn.Module):
    """
    Balanced Binary Cross-Entropy Loss
    
    Weights the positive and negative classes to handle class imbalance.
    """
    def __init__(self, 
                pos_weight: Optional[float] = None, 
                reduction: str = 'mean'):
        """
        Initialize Balanced BCE loss
        
        Args:
            pos_weight: Weight for positive class. If None, calculated from target
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, 
               pred: torch.Tensor, 
               target: torch.Tensor) -> torch.Tensor:
        """
        Compute Balanced BCE Loss
        
        Args:
            pred: Predicted segmentation
            target: Target segmentation
            
        Returns:
            Balanced BCE loss value
        """
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate class weights if not provided
        if self.pos_weight is None:
            # Number of positive and negative samples
            num_pos = torch.sum(target_flat)
            num_neg = target_flat.numel() - num_pos
            
            # Avoid division by zero
            if num_pos == 0:
                num_pos = 1
            if num_neg == 0:
                num_neg = 1
                
            # Balance weights
            pos_weight = num_neg / num_pos
        else:
            pos_weight = self.pos_weight
            
        # Calculate balanced BCE loss
        loss = F.binary_cross_entropy_with_logits(
            pred_flat,
            target_flat,
            pos_weight=torch.tensor(pos_weight, device=pred.device),
            reduction=self.reduction
        )
        
        return loss

class L1RegularizationLoss(nn.Module):
    """
    L1 Regularization Loss
    
    Applies L1 regularization to the predicted values.
    """
    def __init__(self):
        """Initialize L1 Regularization Loss"""
        super().__init__()
        
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 Regularization Loss
        
        Args:
            pred: Predicted segmentation
            
        Returns:
            L1 regularization loss value
        """
        return torch.mean(torch.abs(pred))

class HybridLoss(nn.Module):
    """
    Hybrid Loss Function combining multiple losses
    
    Combines Generalized Dice Loss, Balanced BCE, and L1 regularization
    as described in the paper.
    """
    def __init__(self, 
                w_gds: float = 1.0, 
                w_bbc: float = 1.0,
                w_r: float = 1e-5,
                apply_l1_for_side: bool = False):
        """
        Initialize Hybrid Loss
        
        Args:
            w_gds: Weight for Generalized Dice Loss
            w_bbc: Weight for Balanced BCE Loss
            w_r: Weight for L1 regularization
            apply_l1_for_side: Whether to apply L1 reg for side outputs
        """
        super().__init__()
        self.w_gds = w_gds
        self.w_bbc = w_bbc
        self.w_r = w_r
        self.apply_l1_for_side = apply_l1_for_side
        
        # Initialize component losses
        self.gds_loss = GeneralizedDiceLoss()
        self.bbc_loss = BalancedBCELoss()
        self.l1_loss = L1RegularizationLoss()
        
    def fuse_loss(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for fused output
        
        Args:
            pred: Predicted segmentation
            target: Target segmentation
            
        Returns:
            Combined loss for fused output
        """
        gds = self.gds_loss(pred, target)
        bbc = self.bbc_loss(pred, target)
        l1_reg = self.l1_loss(pred)
        
        return self.w_gds * gds + self.w_bbc * bbc + self.w_r * l1_reg
    
    def side_loss(self, 
                pred: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for side outputs
        
        Args:
            pred: Predicted segmentation
            target: Target segmentation
            
        Returns:
            Combined loss for side output
        """
        gds = self.gds_loss(pred, target)
        bbc = self.bbc_loss(pred, target)
        
        loss = self.w_gds * gds + self.w_bbc * bbc
        
        # Add L1 regularization if enabled
        if self.apply_l1_for_side:
            l1_reg = self.l1_loss(pred)
            loss += self.w_r * l1_reg
            
        return loss
    
    def forward(self, 
               preds: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
               target: torch.Tensor) -> torch.Tensor:
        """
        Compute final hybrid loss
        
        Args:
            preds: Predicted segmentation (fused output and optional side outputs)
                  If tuple, first element is fused output, rest are side outputs
            target: Target segmentation
            
        Returns:
            Final hybrid loss
        """
        # Handle different input formats
        if isinstance(preds, tuple):
            fused_output = preds[0]
            side_outputs = preds[1:]
        else:
            fused_output = preds
            side_outputs = []
        
        # Compute fused loss
        loss = self.fuse_loss(fused_output, target)
        
        # Add side losses if available
        for side_output in side_outputs:
            loss += self.side_loss(side_output, target)
            
        return loss