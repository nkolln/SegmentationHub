import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.logger import Logger
import wandb
from torchmetrics import JaccardIndex, ConfusionMatrix
import numpy as np
from src.training.losses import CombinedLoss,DiceLoss
from src.training.focal_loss import FocalLoss
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image
import cv2 # For connected components
import seaborn as sns

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
        self.num_classes = config['model']['num_classes']
        
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
        # # class_weights = compute_class_weights(train_dataset)
        # class_sample_count = np.unique(train_loader.dataset.targets, return_counts=True)[1]
        # weight = 1. / class_sample_count
        # samples_weight = weight[target]
        # class_weights = torch.from_numpy(samples_weight)

        print(class_weights)
        # Loss function - Dynamic initialization
        loss_config = config.get('loss', {})
        
        # We use CombinedLoss as the primary entry point now
        self.criterion = CombinedLoss(
            loss_config=loss_config,
            num_classes=self.num_classes,
            weight=class_weights,
            ignore_index=config['loss'].get('ignore_index', -100)
        )
        
        print(f"✅ Loss initialized with: {list(self.criterion.loss_weights.keys())}")
        for l_name, l_weight in self.criterion.loss_weights.items():
            print(f"   - {l_name}: weight {l_weight}")
            
        # Optional Counting Loss
        self.use_counting_loss = config['training'].get('use_counting_loss', False)
        if self.use_counting_loss:
            self.counting_criterion = nn.MSELoss()
            self.counting_loss_weight = config['training'].get('counting_loss_weight', 0.1)
            print(f"✅ Auxiliary Window Counting Loss enabled (Weight: {self.counting_loss_weight})")
        else:
             self.use_counting_loss = False
             self.counting_loss_weight = 0.0
            
        self.logger = Logger(config)
        self.global_step = 0
        
        # Metrics
        if self.num_classes == 2:
            self.iou_metric = JaccardIndex(task="binary").to(self.device)
            self.conf_matrix = ConfusionMatrix(task="binary").to(self.device)
        else:
            self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.num_classes).to(self.device)
            self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(self.device)
        
        # Scheduler - OneCycleLR for better convergence
        scheduler_type = config['training'].get('scheduler', 'cosine')
        if scheduler_type == 'onecycle':
            max_lr_base = config['training'].get('max_lr', 1e-4)
            # Create a list of max_lrs for each param group
            # Group 0: Decoder, Group 1: Encoder
            max_lrs = [max_lr_base, max_lr_base * self.encoder_lr_mult]
            
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
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
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
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
        """Create optimizer with both encoder and decoder groups present from the start."""
        encoder_params, decoder_params = self._get_encoder_decoder_params()
        weight_decay = float(config['training']['weight_decay'])
        
        param_groups = [
            {'params': decoder_params, 'lr': base_lr, 'name': 'decoder', 'weight_decay': weight_decay},
            {
                'params': encoder_params, 
                'lr': base_lr * self.encoder_lr_mult, 
                'name': 'encoder',
                'weight_decay': 0.0 if self.freeze_encoder else weight_decay # Avoid weight decay on frozen params
            },
        ]
        
        return torch.optim.AdamW(param_groups)
    
    def _freeze_encoder(self):
        """Initial freeze of encoder parameters."""
        encoder_params, _ = self._get_encoder_decoder_params()
        for param in encoder_params:
            param.requires_grad = False
        self.encoder_frozen = True
        
        frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_count = sum(1 for _ in self.model.parameters())
        print(f"   Frozen {frozen_count}/{total_count} parameters")

    def _unfreeze_encoder(self):
        """Unfreeze encoder parameters and enable weight decay for its group."""
        encoder_params, _ = self._get_encoder_decoder_params()
        for param in encoder_params:
            param.requires_grad = True
        
        # Enable weight decay for the encoder group (it's at index 1)
        weight_decay = float(self.config['training']['weight_decay'])
        for group in self.optimizer.param_groups:
            if group.get('name') == 'encoder':
                group['weight_decay'] = weight_decay
        
        self.encoder_frozen = False
        print(f"🔓 Encoder unfrozen! Enabled gradients and weight decay for encoder group.")

    def train_epoch(self, epoch):
        # Check if we should unfreeze encoder this epoch
        if self.freeze_encoder and self.encoder_frozen and epoch >= self.unfreeze_epoch:
            self._unfreeze_encoder()
        
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # Get gradient accumulation steps from config (default to 1 if not set)
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        self.optimizer.zero_grad() # Initialize gradients once
        
        for batch_idx, batch_data in enumerate(pbar):
            # Dataset returns 3 items?
            if len(batch_data) == 3:
                images, masks, counts = batch_data
                images, masks, counts = images.to(self.device), masks.to(self.device), counts.to(self.device)
            else:
                images, masks = batch_data
                images, masks = images.to(self.device), masks.to(self.device)
                counts = None
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    
                    # Handle Aux Output
                    if isinstance(outputs, tuple):
                        seg_logits, count_pred = outputs
                    else:
                        seg_logits = outputs
                        count_pred = None
                        
                    loss = self.criterion(seg_logits, masks.long())
                    
                    if self.use_counting_loss and count_pred is not None and counts is not None:
                        count_loss = self.counting_criterion(count_pred, counts)
                        loss = loss + self.counting_loss_weight * count_loss
                    
                    loss = loss / accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                # Standard forward/backward
                outputs = self.model(images)
                
                # Handle Aux Output
                if isinstance(outputs, tuple):
                    seg_logits, count_pred = outputs
                else:
                    seg_logits = outputs
                    count_pred = None
                    
                loss = self.criterion(seg_logits, masks.long())
                
                if self.use_counting_loss and count_pred is not None and counts is not None:
                    count_loss = self.counting_criterion(count_pred, counts)
                    loss = loss + self.counting_loss_weight * count_loss
                    
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
                
                # Gradient Clipping for stability (especially important for Transformers)
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
                if self.global_step % self.config['logging'].get('log_every_n_steps', 10) == 0:
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
        self.conf_matrix.reset()
        
        # Domain Metrics Accumulators
        total_window_count_error = 0.0
        total_head_error = 0.0
        total_area_error = 0.0
        n_samples = 0

        # Select one batch for visualization (first batch)
        vis_images, vis_masks, vis_preds = None, None, None
        
        with torch.no_grad():
            vbar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch}", leave=False)
            for batch_idx, batch_data in enumerate(vbar):
                if len(batch_data) == 3:
                     images, masks, counts = batch_data
                     gt_counts = counts.to(self.device)
                else:
                     images, masks = batch_data
                     gt_counts = None
                
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Conditional TTA based on frequency
                use_tta = self.config['training'].get('use_tta', False)
                tta_freq = self.config['training'].get('tta_every_n_epochs', 1)
                do_tta = use_tta and (epoch % tta_freq == 0)

                if do_tta:
                    outputs = self._tta_inference(images)
                else:
                    outputs = self.model(images)
                
                if isinstance(outputs, tuple):
                    outputs, pred_count = outputs
                else:
                    pred_count = None
                    
                loss = self.criterion(outputs, masks.long())
                val_loss += loss.item()
                
                # Update IoU metric
                preds = torch.argmax(outputs, dim=1)
                self.iou_metric.update(preds, masks)
                self.conf_matrix.update(preds, masks)
                
                # --- Domain Metrics Calculation ---
                # Move to CPU for numpy operations
                batch_preds_np = preds.detach().cpu().numpy().astype(np.uint8)
                batch_masks_np = masks.detach().cpu().numpy().astype(np.uint8)
                
                for i in range(len(batch_preds_np)):
                    # 1. Window Count Error (Class 2)
                    # Use XML counts if available, otherwise fallback to connected components
                    gt_wins = (batch_masks_np[i] == 2).astype(np.uint8)
                    if gt_counts is not None:
                        n_gt = float(gt_counts[i].item())
                    else:
                        n_gt = float(cv2.connectedComponents(gt_wins)[0] - 1)
                        
                    # Predicted count from Segmentation Head (connected components)
                    pred_wins = (batch_preds_np[i] == 2).astype(np.uint8)
                    n_pred_mask = float(cv2.connectedComponents(pred_wins)[0] - 1)
                    
                    # Error for Segmentation Mask
                    total_window_count_error += abs(n_pred_mask - n_gt)
                    
                    # Error for Counting Head (if exists)
                    if pred_count is not None:
                         n_pred_head = float(pred_count[i].item())
                         total_head_error += abs(n_pred_head - n_gt)
                    
                    # 2. Area Error (Class 1 = Facade)
                    gt_area = np.sum(batch_masks_np[i] == 1)
                    pred_area = np.sum(batch_preds_np[i] == 1)
                    
                    # Avoid division by zero
                    if gt_area > 0:
                        err = abs(pred_area - gt_area) / gt_area
                    else:
                        err = 0.0 if pred_area == 0 else 1.0
                    total_area_error += err
                    n_samples += 1

                # Capture variables for visualization from the first batch
                if batch_idx == 0:
                    vis_images = images.cpu()
                    vis_masks = masks.cpu()
                    vis_preds = preds.cpu()
        
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        mean_iou = self.iou_metric.compute().item()
        
        avg_win_err = total_window_count_error / max(1, n_samples)
        avg_head_err = total_head_error / max(1, n_samples)
        avg_area_err = total_area_error / max(1, n_samples)
        
        metrics = {
            'val_loss': avg_val_loss,
            'val_iou': mean_iou,
            'val_window_count_error': avg_win_err,
            'val_head_error': avg_head_err,
            'val_area_error_pct': avg_area_err
        }
        
        print(f"  > Metrics: IoU={mean_iou:.4f} | WinErr={avg_win_err:.2f} | HeadErr={avg_head_err:.2f} | AreaErr={avg_area_err:.2%}")
        
        # Log visualization (only if wandb is active)
        if self.config['logging']['use_wandb'] and vis_images is not None:
            try:
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

                # Automated legible masks (using the loop_mask.py logic)
                metrics['legible_ground_truth'] = self._generate_legible_mask(gt_mask, f"Epoch {epoch} GT")
                metrics['legible_prediction'] = self._generate_legible_mask(pred_mask, f"Epoch {epoch} Pred")
            except Exception as e:
                print(f"  ⚠ Failed to create wandb image: {e}")

        # Log confusion matrix to WandB
        if self.config['logging']['use_wandb']:
            try:
                cm = self.conf_matrix.compute().cpu().numpy()
                class_names = [f"Class {i}" for i in range(self.num_classes)]
                
                # Also log a heatmap image of the confusion matrix
                metrics['confusion_matrix_plot'] = self._plot_confusion_matrix(cm, class_names, epoch)

            except Exception as e:
                print(f"  ⚠ Failed to log confusion matrix table: {e}")

        self.logger.log_metrics(metrics, step=self.global_step)
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val IoU: {mean_iou:.4f}")
        
        # Early stopping check
        if mean_iou > self.best_val_iou:
            self.best_val_iou = mean_iou
            self.patience_counter = 0
            print(f"  ✓ New best IoU: {mean_iou:.4f}")
            self._save_checkpoint(epoch, is_best=True)
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
            
            # Periodic checkpoint saving
            self._save_checkpoint(epoch, is_best=False)
        
        print(f"\n🎯 Training completed! Best IoU: {self.best_val_iou:.4f}")
        self.logger.finish()

    def _plot_confusion_matrix(self, cm, class_names, epoch):
        """
        Generates a professional heatmap plot using Seaborn.
        """
        # Normalize the confusion matrix: row-wise (actual class)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm) # Handle division by zero for rare classes
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Normalized Confusion Matrix - Epoch {epoch}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Buffer to WandB image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        wandb_img = wandb.Image(img, caption=f"Normalized Confusion Matrix Epoch {epoch}")
        plt.close()
        
        return wandb_img

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        save_dir = self.config['logging'].get('output_dir', 'outputs')
        if self.config.get('experiment_name', None):
            save_dir = os.path.join(save_dir, self.config.get('experiment_name'))
        os.makedirs(save_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_iou': self.best_val_iou,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(save_dir, 'best_model.pth')
            torch.save(state, path)
            print(f"  💾 Best model saved to {path}")
        
        # Periodic saving logic (e.g. every 5 epochs)
        save_freq = self.config['logging'].get('save_checkpoint_every_n_epochs', 5)
        if (epoch + 1) % save_freq == 0:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(state, path)
            print(f"  💾 Checkpoint saved to {path}")

    def _generate_legible_mask(self, mask_np, title):
        """Generates a color-mapped visualization of a mask, similar to loop_mask.py"""
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_np, cmap='tab20', vmin=0, vmax=max(11, np.max(mask_np)))
        plt.colorbar()
        plt.title(f"{title} (Values: {np.unique(mask_np)})")
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return wandb.Image(Image.open(buf), caption=title)

    def _tta_inference(self, images):
        """
        Multi-Scale Test-Time Augmentation (TTA).
        Scales: [0.75, 1.0, 1.25] + Horizontal Flip
        """
        scales = [0.75, 1.0, 1.25]
        flips = [False, True]
        
        # Output accumulation
        total_logits = 0
        n_augs = 0
        
        B, C, H, W = images.shape
        
        for scale in scales:
            # Resize image
            if scale != 1.0:
                raw_h, raw_w = H * scale, W * scale
                divisor = 14
                
                # Robust calculation: Ensure >= raw dims and divisible by 14
                scaled_h = int(math.ceil(raw_h / divisor) * divisor)
                scaled_w = int(math.ceil(raw_w / divisor) * divisor)
                
                scaled_images = F.interpolate(images, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
            else:
                scaled_images = images
                
            for flip in flips:
                if flip:
                    inp = torch.flip(scaled_images, dims=[3])
                else:
                    inp = scaled_images
                
                # Inference
                with torch.no_grad():
                    # Check for AMP
                    if self.use_amp:
                         with autocast('cuda'):
                            outputs = self.model(inp)
                    else:
                        outputs = self.model(inp)
                
                # Handle Aux Output (Tuple)
                # We only TTA the segmentation logits, ignoring counting head for now
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Undo flip
                if flip:
                    logits = torch.flip(logits, dims=[3])
                
                # Undo scaling (resize back to original H, W)
                if scale != 1.0:
                    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                
                total_logits += logits
                n_augs += 1
                
        return total_logits / n_augs
