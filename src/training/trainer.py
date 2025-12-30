import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.logger import Logger
import wandb
from torchmetrics import JaccardIndex
import numpy as np
from src.training.losses import CombinedLoss,DiceLoss
from src.training.focal_loss import FocalLoss
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    """
    Trainer class for managing the training and validation loops.
    Supports encoder freezing with gradual unfreezing.
    """
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Encoder freezing configuration
        self.freeze_encoder = config['training'].get('freeze_encoder', False)
        self.unfreeze_epoch = config['training'].get('unfreeze_epoch', 5)
        self.encoder_lr_mult = config['training'].get('encoder_lr_mult', 0.1)
        self.encoder_frozen = False
        
        # Apply initial encoder freeze if configured
        if self.freeze_encoder:
            self._freeze_encoder()
            print(f"🔒 Encoder frozen. Will unfreeze at epoch {self.unfreeze_epoch}")
        
        # Create optimizer with parameter groups (encoder vs decoder)
        base_lr = float(config['training']['learning_rate'])
        self.optimizer = self._create_optimizer(base_lr, config)
        
        # Handle class weights if provided, convert to tensor
        class_weights = config['loss'].get('class_weights', None)
        if class_weights:
             class_weights = torch.tensor(class_weights).float().to(self.device)

        # Loss function - support for focal loss
        loss_type = config['loss'].get('type', 'cross_entropy')
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=config['loss'].get('focal_alpha', 0.25),
                gamma=config['loss'].get('focal_gamma', 2.0),
                ignore_index=255
            )
        elif loss_type == 'combined':
            self.criterion = CombinedLoss(weight=class_weights)
        elif loss_type == 'dice':
            self.criterion = DiceLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        self.logger = Logger(config)
        self.global_step = 0
        
        # Metrics
        self.num_classes = config['model']['num_classes']
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.num_classes).to(self.device)
        
        # Scheduler - OneCycleLR for better convergence
        scheduler_type = config['training'].get('scheduler', 'cosine')
        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config['training'].get('max_lr', 1e-4),
                epochs=config['training']['num_epochs'],
                steps_per_epoch=len(train_loader),
                pct_start=config['training'].get('warmup_ratio', 0.3),
                anneal_strategy='cos'
            )
            self.step_per_batch = True  # OneCycle steps per batch
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config['training']['num_epochs'], 
                eta_min=1e-6
            )
            self.step_per_batch = False
        
        # Mixed precision training for speed
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        self.best_val_iou = 0.0
        self.patience_counter = 0
        self.should_stop = False
    
    def _get_encoder_decoder_params(self):
        """
        Separate model parameters into encoder and decoder groups.
        Works with different model architectures.
        """
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            # Common encoder layer names across architectures
            if any(enc in name.lower() for enc in ['encoder', 'backbone', 'resnet', 'segformer', 'mit', 'efficientnet']):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        # Fallback: if no encoder found, treat first 70% as encoder
        if len(encoder_params) == 0:
            all_params = list(self.model.parameters())
            split_idx = int(len(all_params) * 0.7)
            encoder_params = all_params[:split_idx]
            decoder_params = all_params[split_idx:]
        
        return encoder_params, decoder_params
    
    def _create_optimizer(self, base_lr, config):
        """Create optimizer with separate parameter groups for encoder/decoder."""
        encoder_params, decoder_params = self._get_encoder_decoder_params()
        
        param_groups = [
            {'params': decoder_params, 'lr': base_lr, 'name': 'decoder'},
        ]
        
        # Only add encoder params if not frozen
        if not self.encoder_frozen:
            param_groups.append({
                'params': encoder_params, 
                'lr': base_lr * self.encoder_lr_mult, 
                'name': 'encoder'
            })
        
        return torch.optim.AdamW(
            param_groups,
            weight_decay=float(config['training']['weight_decay'])
        )
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        encoder_params, _ = self._get_encoder_decoder_params()
        for param in encoder_params:
            param.requires_grad = False
        self.encoder_frozen = True
        
        frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_count = sum(1 for _ in self.model.parameters())
        print(f"   Frozen {frozen_count}/{total_count} parameters")
    
    def _unfreeze_encoder(self):
        """Unfreeze encoder parameters and update optimizer."""
        encoder_params, _ = self._get_encoder_decoder_params()
        for param in encoder_params:
            param.requires_grad = True
        self.encoder_frozen = False
        
        # Recreate optimizer with encoder params included
        base_lr = float(self.config['training']['learning_rate'])
        self.optimizer = self._create_optimizer(base_lr, self.config)
        
        # Note: scheduler will continue from current state
        print(f"🔓 Encoder unfrozen! Now training all parameters with encoder_lr_mult={self.encoder_lr_mult}")

    def train_epoch(self, epoch):
        # Check if we should unfreeze encoder this epoch
        if self.freeze_encoder and self.encoder_frozen and epoch >= self.unfreeze_epoch:
            self._unfreeze_encoder()
        
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # Get gradient accumulation steps from config (default to 1 if not set)
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        self.optimizer.zero_grad() # Initialize gradients once
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks.long())
                    loss = loss / accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                # Standard forward/backward
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.long())
                loss = loss / accumulation_steps
                loss.backward()
            
            # Perform optimization step every 'accumulation_steps'
            if (batch_idx + 1) % accumulation_steps == 0:
                # Calculate gradient norm (on the accumulated gradients)
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                # Optimizer step with mixed precision support
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Step scheduler per batch if OneCycleLR
                if self.step_per_batch:
                    self.scheduler.step()
                
                # Logging (log the accumulated/actual loss, so multiply back)
                if batch_idx % self.config['logging']['log_every_n_steps'] == 0:
                    self.logger.log_metrics({
                        'train_loss': loss.item() * accumulation_steps, 
                        'train_grad_norm': total_norm,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
                
                self.global_step += 1
            
            # Update pbar with current scaled loss * steps to look normal
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
            
        # Step scheduler at epoch end only if not per-batch
        if not self.step_per_batch:
            self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        self.iou_metric.reset()
        
        # Select one batch for visualization (first batch)
        vis_images, vis_masks, vis_preds = None, None, None
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.long())
                val_loss += loss.item()
                
                # Update IoU metric
                preds = torch.argmax(outputs, dim=1)
                self.iou_metric.update(preds, masks)
                
                # Capture variables for visualization from the first batch
                if batch_idx == 0:
                    vis_images = images.cpu()
                    vis_masks = masks.cpu()
                    vis_preds = preds.cpu()
        
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        mean_iou = self.iou_metric.compute().item()
        
        metrics = {
            'val_loss': avg_val_loss,
            'val_iou': mean_iou
        }
        
        # Log visualization (only if wandb is active)
        if self.config['logging']['use_wandb'] and vis_images is not None:
            # Take the first image in the batch
            image = vis_images[0] # Tensor (3, H, W)
            
            # Unnormalize: (image * std) + mean
            mean = torch.tensor([0.485, 0.456, 0.406]).to(image.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(image.device).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            gt_mask = vis_masks[0].numpy()
            pred_mask = vis_preds[0].numpy()
            
            # Create interactive mask overlay
            class_labels = {i: f"Class {i}" for i in range(self.num_classes)}
            
            # Debug: Print stats about what we are visualizing
            print(f"DEBUG Vis: GT values: {np.unique(gt_mask)}, Pred values: {np.unique(pred_mask)}")
            
            wandb_image = wandb.Image(image_np, masks={
                "predictions": {
                    "mask_data": pred_mask,
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": gt_mask,
                    "class_labels": class_labels
                }
            }, caption=f"Epoch {epoch} Prediction")
            
            metrics['val_prediction'] = wandb_image

        self.logger.log_metrics(metrics, step=self.global_step)
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val IoU: {mean_iou:.4f}")
        
        # Early stopping check
        if mean_iou > self.best_val_iou:
            self.best_val_iou = mean_iou
            self.patience_counter = 0
            print(f"  ✓ New best IoU: {mean_iou:.4f}")
            # TODO: Save best checkpoint here
        else:
            self.patience_counter += 1
            print(f"  No improvement for {self.patience_counter} epochs")
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                self.should_stop = True

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.train_epoch(epoch)
            self.validate(epoch)
            
            # Check early stopping
            if self.should_stop:
                print(f"Training stopped early at epoch {epoch + 1}")
                break
            
            # Save checkpoint logic here (omitted for brevity)
        
        print(f"\n🎯 Training completed! Best IoU: {self.best_val_iou:.4f}")
        self.logger.finish()
