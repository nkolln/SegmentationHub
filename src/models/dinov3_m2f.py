"""
DINOv3 + Mask2Former Hybrid Architecture.

Combines the powerful DINOv3 ViT-L backbone with a full Mask2Former-style
transformer decoder for improved segmentation performance.

Architecture:
    DINOv3 Backbone → Pixel Decoder (FPN) → Transformer Decoder → Masks + Classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelLevelModule,
    Mask2FormerTransformerModule,
    Mask2FormerMaskedAttentionDecoder,
    Mask2FormerLoss,
)
from typing import Optional, List, Tuple, Dict
from scipy.optimize import linear_sum_assignment


class PixelDecoder(nn.Module):
    """
    FPN-style pixel decoder that fuses multi-scale features from DINOv3.
    Produces pixel embeddings for the transformer decoder.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 256, num_feature_levels: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        
        # Lateral connections (1x1 conv to reduce channel dim)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
            ) for _ in range(num_feature_levels)
        ])
        
        # Output convolutions (3x3 conv for spatial mixing)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_feature_levels)
        ])
        
        # Final mask features projection
        self.mask_features = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            features: List of 4 feature maps from DINOv3, each (B, C, H, W)
            
        Returns:
            mask_features: (B, hidden_dim, H, W) - highest resolution
            multi_scale_features: List of (B, hidden_dim, H_i, W_i)
        """
        # Process from coarse to fine (reverse order for FPN)
        lateral_features = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            lateral_features.append(lateral_conv(feat))
        
        # Top-down pathway with lateral connections
        out_features = []
        prev_feat = None
        
        for i in range(len(lateral_features) - 1, -1, -1):
            lat_feat = lateral_features[i]
            
            if prev_feat is not None:
                # Upsample and add
                prev_upsampled = F.interpolate(
                    prev_feat, size=lat_feat.shape[-2:], mode='bilinear', align_corners=False
                )
                lat_feat = lat_feat + prev_upsampled
            
            out_feat = self.output_convs[i](lat_feat)
            out_features.insert(0, out_feat)
            prev_feat = out_feat
        
        # Mask features from highest resolution
        mask_feats = self.mask_features(out_features[0])
        
        return mask_feats, out_features


class TransformerDecoderLayer(nn.Module):
    """Single layer of the Mask2Former transformer decoder."""
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention to pixel features (masked attention)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, queries: torch.Tensor, pixel_features: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: (B, N, C) - N learnable queries
            pixel_features: (B, HW, C) - flattened pixel features
            attn_mask: Optional attention mask for masked attention
        """
        # Self-attention
        q = self.self_attn_norm(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        
        # Cross-attention to pixels
        q = self.cross_attn_norm(queries)
        if attn_mask is not None:
            queries = queries + self.cross_attn(q, pixel_features, pixel_features, attn_mask=attn_mask)[0]
        else:
            queries = queries + self.cross_attn(q, pixel_features, pixel_features)[0]
        
        # FFN
        queries = queries + self.ffn(self.ffn_norm(queries))
        
        return queries


class TransformerDecoder(nn.Module):
    """
    Mask2Former-style transformer decoder.
    Uses learnable queries that cross-attend to pixel features.
    """
    def __init__(self, hidden_dim: int = 256, num_queries: int = 100, 
                 num_heads: int = 8, num_layers: int = 6, num_classes: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        
        # Level embedding for multi-scale features
        self.level_embed = nn.Embedding(4, hidden_dim)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, mask_features: torch.Tensor, multi_scale_features: List[torch.Tensor]):
        """
        Args:
            mask_features: (B, C, H, W) from pixel decoder
            multi_scale_features: List of (B, C, H_i, W_i)
            
        Returns:
            all_class_logits: List of (B, N, num_classes+1) for each layer
            all_mask_logits: List of (B, N, H, W) for each layer
        """
        B = mask_features.shape[0]
        H, W = mask_features.shape[-2:]
        
        # Initialize queries
        queries = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, N, C)
        
        # Flatten mask features for cross-attention
        pixel_feats = mask_features.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        
        # Store predictions from all layers for auxiliary losses
        all_class_logits = []
        all_mask_logits = []
        
        # Process through decoder layers
        for layer in self.layers:
            queries = layer(queries, pixel_feats)
            
            # Predict at this layer
            class_logits = self.class_embed(queries)  # (B, N, num_classes+1)
            mask_embed = self.mask_embed(queries)     # (B, N, C)
            
            # Compute mask predictions via dot product with mask_features
            # mask_logits[b, n, h, w] = sum_c(mask_embed[b, n, c] * mask_features[b, c, h, w])
            mask_logits = torch.bmm(mask_embed, mask_features.flatten(2)).view(B, self.num_queries, H, W)
            
            all_class_logits.append(class_logits)
            all_mask_logits.append(mask_logits)
        
        return all_class_logits, all_mask_logits


class DINOv3Mask2Former(nn.Module):
    """
    Hybrid model combining DINOv3 backbone with Mask2Former decoder.
    
    This model:
    1. Extracts multi-scale features from DINOv3
    2. Fuses them via FPN-style pixel decoder
    3. Uses transformer decoder with learnable queries
    4. Produces per-query class predictions and masks
    5. Aggregates to semantic segmentation
    """
    
    def __init__(self, num_classes: int, config: dict = None,
                 model_type: str = 'facebook/dinov3-vitl16-pretrain-lvd1689m',
                 hidden_dim: int = 256, num_queries: int = 100, 
                 num_decoder_layers: int = 6, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Override from config if provided
        if config:
            model_type = config['model'].get('encoder_name', model_type)
            hidden_dim = config['model'].get('hidden_dim', hidden_dim)
            num_queries = config['model'].get('num_queries', num_queries)
            num_decoder_layers = config['model'].get('num_decoder_layers', num_decoder_layers)
            freeze_backbone = config['training'].get('freeze_encoder', freeze_backbone)
        
        print(f"Loading DINOv3 backbone: {model_type}...")
        
        # 1. Load DINOv3 backbone
        self.backbone_config = AutoConfig.from_pretrained(model_type)
        self.backbone = AutoModel.from_pretrained(model_type)
        self.patch_size = self.backbone_config.patch_size
        self.embed_dim = self.backbone_config.hidden_size
        
        print(f"✓ Backbone loaded: patch_size={self.patch_size}, embed_dim={self.embed_dim}")
        
        # 2. Pixel Decoder (FPN)
        self.pixel_decoder = PixelDecoder(
            in_channels=self.embed_dim,
            hidden_dim=hidden_dim,
            num_feature_levels=4
        )
        
        # 3. Transformer Decoder
        self.transformer_decoder = TransformerDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_heads=8,
            num_layers=num_decoder_layers,
            num_classes=num_classes
        )
        
        # Loss weights (from config or defaults)
        self.class_weight = 2.0
        self.mask_weight = 5.0
        self.dice_weight = 5.0
        
        if config and 'loss' in config:
            self.class_weight = float(config['loss'].get('class_weight', 2.0))
            self.mask_weight = float(config['loss'].get('mask_weight', 5.0))
            self.dice_weight = float(config['loss'].get('dice_weight', 5.0))
        
        print(f"🎯 Loss weights: class={self.class_weight}, mask={self.mask_weight}, dice={self.dice_weight}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ DINOv3 backbone frozen")
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from DINOv3 backbone."""
        B, C, H, W = x.shape
        
        # Handle padding for patch size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        H_pad, W_pad = x.shape[-2:]
        ph, pw = H_pad // self.patch_size, W_pad // self.patch_size
        
        # Get intermediate layers
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Use last 4 layers for multi-scale features
        features = []
        for feat in hidden_states[-4:]:
            # Remove CLS + register tokens (first 5 for DINOv3)
            spatial_feat = feat[:, 5:, :]
            # Reshape to spatial
            spatial_feat = spatial_feat.permute(0, 2, 1).reshape(B, self.embed_dim, ph, pw)
            features.append(spatial_feat)
        
        return features, (H, W), (H_pad, W_pad)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            labels: Ground truth masks (B, H, W) for training
            
        Returns:
            dict with 'loss' (if labels provided), 'class_logits', 'mask_logits'
        """
        B = x.shape[0]
        
        # 1. Extract backbone features
        features, (orig_h, orig_w), (pad_h, pad_w) = self._extract_features(x)
        
        # 2. Pixel decoder
        mask_features, multi_scale_features = self.pixel_decoder(features)
        
        # 3. Transformer decoder (returns predictions from all layers)
        all_class_logits, all_mask_logits = self.transformer_decoder(mask_features, multi_scale_features)
        
        # 4. Upsample all mask predictions to original resolution
        all_mask_logits_upsampled = []
        for mask_logits in all_mask_logits:
            mask_logits_up = F.interpolate(mask_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            all_mask_logits_upsampled.append(mask_logits_up)
        
        # Use final layer for inference
        outputs = {
            'class_logits': all_class_logits[-1],  # (B, N, num_classes+1)
            'mask_logits': all_mask_logits_upsampled[-1],  # (B, N, H, W)
        }
        
        # 5. Compute loss if labels provided (using all layers)
        if labels is not None:
            loss = self._compute_loss_with_aux(all_class_logits, all_mask_logits_upsampled, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def _compute_loss_single(self, class_logits: torch.Tensor, mask_logits: torch.Tensor,
                             labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Mask2Former-style loss with Hungarian matching for a single layer.
        
        Args:
            class_logits: (B, N, num_classes+1)
            mask_logits: (B, N, H, W)
            labels: (B, H, W)
        """
        B, N, num_cls_with_bg = class_logits.shape
        device = class_logits.device
        
        total_loss = 0.0
        
        for b in range(B):
            gt_mask = labels[b]  # (H, W)
            pred_masks = mask_logits[b]  # (N, H, W)
            pred_classes = class_logits[b]  # (N, num_classes+1)
            
            # Get unique classes in GT (excluding ignore index)
            unique_classes = torch.unique(gt_mask)
            unique_classes = unique_classes[unique_classes != 255]
            
            if len(unique_classes) == 0:
                continue
            
            # Create binary masks for each GT class
            gt_binary_masks = []
            gt_class_ids = []
            for cls in unique_classes:
                gt_binary_masks.append((gt_mask == cls).float())
                gt_class_ids.append(cls.long())
            
            gt_binary_masks = torch.stack(gt_binary_masks, dim=0)  # (K, H, W)
            gt_class_ids = torch.stack(gt_class_ids)  # (K,)
            K = len(gt_class_ids)
            
            # ===== HUNGARIAN MATCHING =====
            # Build cost matrix: (N queries) x (K GT segments)
            pred_masks_sigmoid = pred_masks.sigmoid()  # (N, H, W)
            pred_probs = F.softmax(pred_classes, dim=-1)  # (N, num_classes+1)
            
            # Cost 1: Classification cost
            # cost_class[n, k] = -prob(pred_n is gt_class_k)
            cost_class = -pred_probs[:, gt_class_ids].T  # (K, N) -> transpose to (N, K)
            cost_class = cost_class.T  # Now (N, K)
            
            # Cost 2: Mask BCE cost
            pred_flat = pred_masks.flatten(1)  # (N, HW) - use logits
            gt_flat = gt_binary_masks.flatten(1)  # (K, HW)
            
            # BCE cost: compute for each (query, gt) pair
            # Expand to (N, K, HW)
            pred_expanded = pred_flat.unsqueeze(1).expand(-1, K, -1)  # (N, K, HW)
            gt_expanded = gt_flat.unsqueeze(0).expand(N, -1, -1)  # (N, K, HW)
            
            cost_mask = F.binary_cross_entropy_with_logits(
                pred_expanded, gt_expanded, reduction='none'
            ).mean(dim=-1)  # (N, K)
            
            # Cost 3: Dice cost
            pred_sigmoid_flat = pred_masks_sigmoid.flatten(1)  # (N, HW)
            intersection = torch.mm(pred_sigmoid_flat, gt_flat.T)  # (N, K)
            pred_area = pred_sigmoid_flat.sum(dim=1, keepdim=True)  # (N, 1)
            gt_area = gt_flat.sum(dim=1, keepdim=True).T  # (1, K)
            union = pred_area + gt_area
            dice = 2 * intersection / (union + 1e-6)
            cost_dice = 1 - dice  # (N, K)
            
            # Total cost
            cost = (
                self.class_weight * cost_class +
                self.mask_weight * cost_mask +
                self.dice_weight * cost_dice
            )  # (N, K)
            
            # Hungarian algorithm (find optimal assignment)
            # linear_sum_assignment expects (rows=workers, cols=jobs)
            # We want to assign N queries to K GT segments
            cost_np = cost.detach().cpu().numpy()
            query_idx, gt_idx = linear_sum_assignment(cost_np)
            
            # Convert to tensors
            assigned_queries = torch.tensor(query_idx, device=device, dtype=torch.long)
            assigned_gt = torch.tensor(gt_idx, device=device, dtype=torch.long)
            
            # ===== COMPUTE LOSSES =====
            if len(assigned_queries) > 0:
                # Class loss (CE)
                pred_cls = pred_classes[assigned_queries]  # (M, num_classes+1)
                target_cls = gt_class_ids[assigned_gt]  # (M,)
                class_loss = F.cross_entropy(pred_cls, target_cls)
                
                # Mask loss (BCE)
                pred_m = pred_masks[assigned_queries]  # (M, H, W)
                target_m = gt_binary_masks[assigned_gt]  # (M, H, W)
                mask_loss = F.binary_cross_entropy_with_logits(pred_m, target_m)
                
                # Dice loss
                pred_m_sigmoid = pred_m.sigmoid()
                intersection = (pred_m_sigmoid * target_m).sum(dim=(1, 2))
                union = pred_m_sigmoid.sum(dim=(1, 2)) + target_m.sum(dim=(1, 2))
                dice = 2 * intersection / (union + 1e-6)
                dice_loss = (1 - dice).mean()
                
                # Background loss: unassigned queries should predict "no-object"
                assigned_set = set(query_idx)
                unassigned = [i for i in range(N) if i not in assigned_set]
                
                if len(unassigned) > 0:
                    unassigned_idx = torch.tensor(unassigned, device=device)
                    bg_pred = pred_classes[unassigned_idx]  # (U, num_classes+1)
                    bg_target = torch.full(
                        (len(unassigned),), self.num_classes, 
                        device=device, dtype=torch.long
                    )
                    bg_loss = F.cross_entropy(bg_pred, bg_target)
                else:
                    bg_loss = 0.0
                
                batch_loss = (
                    self.class_weight * (class_loss + bg_loss * 0.1) +
                    self.mask_weight * mask_loss +
                    self.dice_weight * dice_loss
                )
                total_loss += batch_loss
        
        return total_loss / B
    
    def _compute_loss_with_aux(self, all_class_logits: List[torch.Tensor],
                               all_mask_logits: List[torch.Tensor],
                               labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with auxiliary losses from all decoder layers.
        
        Args:
            all_class_logits: List of (B, N, num_classes+1) from each layer
            all_mask_logits: List of (B, N, H, W) from each layer
            labels: (B, H, W)
        """
        num_layers = len(all_class_logits)
        total_loss = 0.0
        
        # Final layer gets full weight, intermediate layers get 0.5 weight
        for i, (class_logits, mask_logits) in enumerate(zip(all_class_logits, all_mask_logits)):
            layer_weight = 1.0 if i == num_layers - 1 else 0.5
            layer_loss = self._compute_loss_single(class_logits, mask_logits, labels)
            total_loss += layer_weight * layer_loss
        
        return total_loss
    
    def post_process_semantic_segmentation(self, outputs: dict, target_sizes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """
        Convert model outputs to semantic segmentation maps.
        Implemented like official Mask2Former (no einsum).
        
        Args:
            outputs: Dict with 'class_logits' and 'mask_logits'
            target_sizes: List of (H, W) tuples for each image
            
        Returns:
            List of (H, W) semantic segmentation tensors
        """
        class_logits = outputs['class_logits']  # (B, N, num_classes+1)
        mask_logits = outputs['mask_logits']     # (B, N, H, W)
        
        B = class_logits.shape[0]
        results = []
        
        for b in range(B):
            h, w = target_sizes[b]
            
            # Resize masks to target size
            masks = F.interpolate(
                mask_logits[b].unsqueeze(0),  # (1, N, H, W)
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (N, h, w)
            
            # Get class predictions (exclude no-object class)
            # class_logits: (N, num_classes+1)
            # We want argmax over classes (0 to num_classes-1)
            class_probs = F.softmax(class_logits[b], dim=-1)  # (N, num_classes+1)
            
            # Get predicted class for each query (excluding background)
            pred_classes = class_probs[:, :-1].argmax(dim=-1)  # (N,)
            pred_confidences = class_probs[:, :-1].max(dim=-1).values  # (N,)
            mask_probs = masks.sigmoid()  # (N, h, w)
            
            # Initialize semantic map
            semantic_map = torch.zeros((h, w), dtype=torch.long, device=masks.device)
            
            # For each query, paint its mask with its predicted class
            # Confidence threshold to filter out low-confidence predictions
            conf_threshold = 0.5
            
            for query_idx in range(masks.shape[0]):
                # Skip if confidence too low
                if pred_confidences[query_idx] < conf_threshold:
                    continue
                
                # Get mask for this query
                query_mask = mask_probs[query_idx]  # (h, w)
                
                # Get class prediction
                pred_class = pred_classes[query_idx]
                
                # Paint pixels where mask > 0.5 with this class
                semantic_map[query_mask > 0.5] = pred_class
            
            results.append(semantic_map)
        
        return results
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ DINOv3 backbone unfrozen for fine-tuning")
